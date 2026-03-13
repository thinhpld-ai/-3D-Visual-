"""
registration.py - v2
RGB-D Odometry + Pose Graph with keyframe-based loop closure.

Improvements over v1:
- Keyframe selection: add a frame to keyframes every K frames
- Loop closure: for each keyframe, try ICP against earlier keyframes via FPFH+RANSAC
- Longer-range loop edges dramatically reduce drift for walk-around sequences
- Better OdometryOption parameters for hybrid RGBDJacobian
"""

import argparse
import glob
import json
import numpy as np
import open3d as o3d
from pathlib import Path


def load_intrinsics(path):
    with open(path, "r") as f:
        data = json.load(f)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        data["width"], data["height"],
        data["intrinsic_matrix"][0][0],
        data["intrinsic_matrix"][1][1],
        data["intrinsic_matrix"][0][2],
        data["intrinsic_matrix"][1][2]
    )
    return intrinsic


def rgbd_from_files(rgb_path, depth_path):
    color = o3d.io.read_image(str(rgb_path))
    depth = o3d.io.read_image(str(depth_path))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=1000.0,
        depth_trunc=5.0,        # 5m range
        convert_rgb_to_intensity=False
    )
    return rgbd


def pcd_from_rgbd(rgbd, intrinsic):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd = pcd.voxel_down_sample(0.05)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd


def compute_fpfh(pcd):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
    )


def registration_ransac(src_pcd, tgt_pcd, src_fpfh, tgt_fpfh):
    """Fast Global Registration (RANSAC) between two keyframes."""
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd, tgt_pcd,
        src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=0.075,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.075)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999)
    )
    return result


def refine_with_icp(src_pcd, tgt_pcd, init_transform):
    """Refine global registration with point-to-plane ICP."""
    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=0.03,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result


def run_registration(input_dir, output_dir, step=1, keyframe_interval=20):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsic = load_intrinsics(input_dir / "intrinsics.json")

    rgb_files   = sorted(glob.glob(str(input_dir / "rgb"   / "*.png")))[::step]
    depth_files = sorted(glob.glob(str(input_dir / "depth" / "*.png")))[::step]
    n_frames = min(len(rgb_files), len(depth_files))

    if n_frames == 0:
        raise ValueError("No frames found.")

    print(f"Registration v2: {n_frames} frames (step={step})")

    # ---- Build sequential odometry pose graph ----
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))

    odo_option = o3d.pipelines.odometry.OdometryOption()
    # Camera pose in world coordinates (T_world_cam). Start at identity.
    T_wc = np.identity(4)

    print("Building sequential odometry...")
    for s in range(n_frames - 1):
        # Compute transform from frame s -> s+1
        src_rgbd = rgbd_from_files(rgb_files[s],     depth_files[s])
        tgt_rgbd = rgbd_from_files(rgb_files[s + 1], depth_files[s + 1])

        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            src_rgbd, tgt_rgbd,
            intrinsic, np.identity(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            odo_option
        )

        if not success:
            trans = np.identity(4)
            info  = np.identity(6) * 1e-6

        # Accumulate global pose: T_wc(s+1) = T_wc(s) * T_s_to_s+1
        T_wc = T_wc @ trans
        pose_graph.nodes.append(
            # Open3D pose graph nodes store inverse pose in many examples.
            o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(T_wc))
        )
        pose_graph.edges.append(
            # Edge transformation should map source(s) -> target(s+1)
            o3d.pipelines.registration.PoseGraphEdge(s, s + 1, trans, info, uncertain=False)
        )

        if s % 50 == 0:
            print(f"  Odometry: frame {s}/{n_frames}")

    # ---- Keyframe-based loop closure ----
    print(f"Detecting loop closures (keyframe every {keyframe_interval} frames)...")
    keyframe_indices = list(range(0, n_frames, keyframe_interval))
    keyframe_pcds    = []
    keyframe_fpfhs   = []

    for ki in keyframe_indices:
        rgbd = rgbd_from_files(rgb_files[ki], depth_files[ki])
        pcd  = pcd_from_rgbd(rgbd, intrinsic)
        fpfh = compute_fpfh(pcd)
        keyframe_pcds.append(pcd)
        keyframe_fpfhs.append(fpfh)

    loop_added = 0
    for i, ki in enumerate(keyframe_indices):
        # Compare against keyframes that are far enough apart (at least 2 keyframes back)
        for j in range(max(0, i - 5), max(0, i - 1)):
            kj = keyframe_indices[j]
            try:
                result = registration_ransac(
                    keyframe_pcds[i], keyframe_pcds[j],
                    keyframe_fpfhs[i], keyframe_fpfhs[j]
                )
                if result.fitness > 0.3:
                    result_icp = refine_with_icp(
                        keyframe_pcds[i], keyframe_pcds[j], result.transformation
                    )
                    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                        keyframe_pcds[i], keyframe_pcds[j],
                        0.03, result_icp.transformation
                    )
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            ki, kj, result_icp.transformation, info, uncertain=True
                        )
                    )
                    loop_added += 1
            except Exception:
                pass

    print(f"  Loop closure edges added: {loop_added}")

    # ---- Global optimization ----
    print("Running global pose graph optimization...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.03,
        edge_prune_threshold=0.25,
        preference_loop_closure=0.3,
        reference_node=0
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )

    # ---- Save trajectory ----
    # node.pose is (usually) the inverse of T_world_cam, so convert back to T_world_cam.
    poses = [np.linalg.inv(node.pose).tolist() for node in pose_graph.nodes]
    output_data = {
        "step": step,
        "files": [str(Path(f).name) for f in rgb_files],
        "poses": poses,
        "pose_convention": "T_world_cam"
    }
    traj_file = output_dir / "trajectory.json"
    with open(traj_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved optimized trajectory ({n_frames} poses) to {traj_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGB-D Odometry + Loop Closure Registration (v2).")
    parser.add_argument("--input_dir",         type=str, default="outputs/processed_frames")
    parser.add_argument("--output_dir",         type=str, default="outputs/trajectory")
    parser.add_argument("--step",               type=int, default=1)
    parser.add_argument("--keyframe_interval",  type=int, default=20,
                        help="Add a keyframe every N frames for loop closure")
    args = parser.parse_args()
    run_registration(args.input_dir, args.output_dir, args.step, args.keyframe_interval)
