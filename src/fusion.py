"""
fusion.py - v2
TSDF volume fusion with best-quality parameters.

Improvements over v1:
- Save BOTH raw (scene_raw.ply) and cleaned (scene_clean.ply) point clouds
- Smaller TSDF voxel for denser output
- Multi-stage cleaning: statistical + radius outlier removal + normal estimation
- Use full 5m depth range
"""

import argparse
import json
import glob
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


def reconstruct_scene(data_dir, trajectory_path, output_ply, voxel_size=0.008):
    data_dir       = Path(data_dir)
    trajectory_path = Path(trajectory_path)
    output_ply     = Path(output_ply)

    intrinsic = load_intrinsics(data_dir / "intrinsics.json")

    with open(trajectory_path, "r") as f:
        traj = json.load(f)

    filenames = traj["files"]
    poses     = traj["poses"]

    # Use smaller voxel_length for higher resolution output
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 5,      # trunc = 5× voxel (good heuristic)
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    print(f"Fusing {len(filenames)} frames (voxel={voxel_size*100:.1f}cm)...")
    failed = 0
    for idx, (filename, pose_list) in enumerate(zip(filenames, poses)):
        rgb_path   = data_dir / "rgb"   / filename
        depth_path = data_dir / "depth" / filename

        if not rgb_path.exists() or not depth_path.exists():
            failed += 1
            continue

        color = o3d.io.read_image(str(rgb_path))
        depth = o3d.io.read_image(str(depth_path))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=1000.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False
        )
        pose = np.array(pose_list)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

        if idx % 100 == 0:
            print(f"  Integrated {idx}/{len(filenames)} frames...")

    print(f"  (Skipped {failed} missing frames)")

    # --- Extract RAW cloud (no cleaning) ---
    print("\nExtracting RAW point cloud from TSDF volume...")
    pcd_raw = volume.extract_point_cloud()
    print(f"  Raw cloud: {len(pcd_raw.points):,} points")

    # Estimate normals for raw cloud (helps MeshLab rendering)
    pcd_raw.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd_raw.orient_normals_consistent_tangent_plane(100)

    raw_ply = output_ply.parent / "scene_raw.ply"
    o3d.io.write_point_cloud(str(raw_ply), pcd_raw)
    print(f"  Saved RAW cloud → {raw_ply}")

    # --- Multi-stage cleaning ---
    print("\nCleaning point cloud...")
    pcd_clean = pcd_raw

    # Stage 1: Statistical outlier removal (removes sporadic floating noise)
    pcd_clean, ind1 = pcd_clean.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    print(f"  After statistical filter: {len(pcd_clean.points):,} points")

    # Stage 2: Radius outlier removal (removes isolated clusters)
    pcd_clean, ind2 = pcd_clean.remove_radius_outlier(nb_points=20, radius=0.03)
    print(f"  After radius filter: {len(pcd_clean.points):,} points")

    # Stage 3: Voxel downsample to uniform density
    pcd_clean = pcd_clean.voxel_down_sample(voxel_size=voxel_size)
    print(f"  After voxel downsample: {len(pcd_clean.points):,} points")

    # Re-estimate normals on clean cloud
    pcd_clean.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd_clean.orient_normals_consistent_tangent_plane(100)

    o3d.io.write_point_cloud(str(output_ply), pcd_clean)
    print(f"  Saved CLEAN cloud → {output_ply}")

    return str(raw_ply), str(output_ply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSDF Fusion v2 — saves raw + clean PLY.")
    parser.add_argument("--input_dir",  type=str, default="outputs/processed_frames")
    parser.add_argument("--trajectory", type=str, default="outputs/trajectory/trajectory.json")
    parser.add_argument("--output_ply", type=str, default="outputs/pointclouds/scene_clean.ply",
                        help="Path to save CLEANED output (raw is saved alongside automatically)")
    parser.add_argument("--voxel_size", type=float, default=0.008,
                        help="TSDF voxel size in metres (0.008 = 8mm, default)")
    args = parser.parse_args()
    Path(args.output_ply).parent.mkdir(parents=True, exist_ok=True)
    reconstruct_scene(args.input_dir, args.trajectory, args.output_ply, args.voxel_size)
