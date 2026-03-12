import os
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
        data["width"],
        data["height"],
        data["intrinsic_matrix"][0][0], # fx
        data["intrinsic_matrix"][1][1], # fy
        data["intrinsic_matrix"][0][2], # cx
        data["intrinsic_matrix"][1][2]  # cy
    )
    return intrinsic

def generate_rgbd_frames(rgb_files, depth_files):
    frames = []
    for rgb_path, depth_path in zip(rgb_files, depth_files):
        color = o3d.io.read_image(rgb_path)
        depth = o3d.io.read_image(depth_path)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=4.0, convert_rgb_to_intensity=False
        )
        frames.append(rgbd)
    return frames

def build_pose_graph(rgbd_frames, intrinsic):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry_init = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry_init))
    
    trans_odometry = np.identity(4)
    
    option = o3d.pipelines.odometry.OdometryOption()
    
    for s in range(len(rgbd_frames) - 1):
        target = rgbd_frames[s]
        source = rgbd_frames[s + 1]
        
        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            source, target, intrinsic, odometry_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option
        )
        
        if not success:
            print(f"Warning: Odometry failed between frame {s} and {s+1}. Using identity.")
            trans = np.identity(4)
            info = np.identity(6)
            
        trans_odometry = np.dot(trans, trans_odometry)
        
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(trans_odometry)))
        
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s, s + 1, trans, info, uncertain=False)
        )
        
        # Simple loop closure tracking (comparing to a few earlier frames if possible)
        # For simplicity in this baseline script, we just connect to the explicit previous frame
        # advanced loop closure would involve FPFH features or similar to detect larger loops
        
    return pose_graph

def optimize_pose_graph(pose_graph):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.03,
        edge_prune_threshold=0.25,
        preference_loop_closure=0.1,
        reference_node=0
    )
    
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )
    return pose_graph

def run_registration(input_dir, output_dir, step=1):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    intrinsics_path = input_dir / "intrinsics.json"
    intrinsic = load_intrinsics(intrinsics_path)
    
    rgb_files = sorted(glob.glob(str(input_dir / "rgb" / "*.png")))[::step]
    depth_files = sorted(glob.glob(str(input_dir / "depth" / "*.png")))[::step]
    
    if len(rgb_files) == 0:
        raise ValueError("No frames found in input directory.")
        
    print(f"Loading {len(rgb_files)} frames...")
    rgbd_frames = generate_rgbd_frames(rgb_files, depth_files)
    
    print("Building pose graph (Odometry)...")
    pose_graph = build_pose_graph(rgbd_frames, intrinsic)
    
    print("Optimizing pose graph...")
    optimized_graph = optimize_pose_graph(pose_graph)
    
    poses = []
    for node in optimized_graph.nodes:
        poses.append(node.pose.tolist())
        
    # Save poses and frame mapping to disk
    output_data = {
        "step": step,
        "files": [str(Path(f).name) for f in rgb_files],
        "poses": poses
    }
    
    trajectory_file = output_dir / "trajectory.json"
    with open(trajectory_file, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Saved optimized trajectory to {trajectory_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform RGB-D Odometry and Pose Graph optimization.")
    parser.add_argument("--input_dir", type=str, default="data/processed", help="Path to processed frames")
    parser.add_argument("--output_dir", type=str, default="data/trajectory", help="Path to save trajectory")
    parser.add_argument("--step", type=int, default=1, help="Frame step size (use e.g. 5 to process every 5th frame for speed)")
    args = parser.parse_args()
    
    run_registration(args.input_dir, args.output_dir, args.step)
