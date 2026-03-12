import os
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
        data["width"],
        data["height"],
        data["intrinsic_matrix"][0][0],
        data["intrinsic_matrix"][1][1],
        data["intrinsic_matrix"][0][2],
        data["intrinsic_matrix"][1][2]
    )
    return intrinsic

def reconstruct_scene(data_dir, trajectory_path, output_ply, voxel_size=0.01):
    data_dir = Path(data_dir)
    trajectory_path = Path(trajectory_path)
    
    intrinsics_path = data_dir / "intrinsics.json"
    intrinsic = load_intrinsics(intrinsics_path)
    
    with open(trajectory_path, "r") as f:
        trajectory_data = json.load(f)
        
    filenames = trajectory_data["files"]
    poses = trajectory_data["poses"]
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    print(f"Fusing {len(filenames)} frames into TSDF volume...")
    for idx, (filename, pose_list) in enumerate(zip(filenames, poses)):
        rgb_path = str(data_dir / "rgb" / filename)
        depth_path = str(data_dir / "depth" / filename)
        
        color = o3d.io.read_image(rgb_path)
        depth = o3d.io.read_image(depth_path)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=4.0, convert_rgb_to_intensity=False
        )
        
        pose = np.array(pose_list)
        
        # Integrate into volume
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        
        if idx > 0 and idx % 100 == 0:
            print(f"Integrated {idx} frames...")
            
    print("Extracting point cloud from volume...")
    pcd = volume.extract_point_cloud()
    
    print("Cleaning Point Cloud...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    
    o3d.io.write_point_cloud(str(output_ply), pcd)
    print(f"Saved fused point cloud to {output_ply}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse registered frames into a global point cloud.")
    parser.add_argument("--input_dir", type=str, default="data/processed", help="Path to processed frames")
    parser.add_argument("--trajectory", type=str, default="data/trajectory/trajectory.json", help="Path to poses json")
    parser.add_argument("--output_ply", type=str, default="outputs/pointclouds/scene_clean.ply", help="Path to output point cloud")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size for TSDF fusion")
    args = parser.parse_args()
    
    Path(args.output_ply).parent.mkdir(parents=True, exist_ok=True)
    reconstruct_scene(args.input_dir, args.trajectory, args.output_ply, args.voxel_size)
