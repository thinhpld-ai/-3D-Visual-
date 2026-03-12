import argparse
import open3d as o3d
import numpy as np
import json
from pathlib import Path

def normalize_coordinates(pcd, floor_plane):
    """
    Align the point cloud so that the floor plane becomes Z=0 and its normal becomes the Z-axis.
    """
    a, b, c, d = floor_plane
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    # We want this normal to align with the global Z axis [0, 0, 1]
    target_normal = np.array([0, 0, 1])
    
    # Ensure normal points upwards
    if normal[2] < 0:
        normal = -normal
        
    v = np.cross(normal, target_normal)
    s = np.linalg.norm(v)
    c_val = np.dot(normal, target_normal)
    
    if s < 1e-6:
        # Already aligned or completely anti-aligned
        if c_val < 0:
             R = np.diag([-1, -1, 1])
        else:
             R = np.eye(3)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_val) / (s ** 2))
        
    pcd_normalized = pcd.rotate(R, center=(0,0,0))
    
    # Translate so the lowest point on the plane is at Z=0.
    # Actually, a point on the plane satisfies ax+by+cz+d=0.
    # The normal distance to origin is d / ||normal||.
    # We rotated the cloud, so we should just shift down by the height of the lowest points.
    # To be robust, let's just use the 1st percentile of Z to set Z=0 after rotation.
    points = np.asarray(pcd_normalized.points)
    if len(points) > 0:
        z_min = np.percentile(points[:, 2], 1)
        pcd_normalized.translate((0, 0, -z_min))
        
    return pcd_normalized

def calculate_metrics(pcd, voxel_size=0.01):
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return {}
        
    # Percentile-based Height
    z_low = np.percentile(points[:, 2], 1)
    z_high = np.percentile(points[:, 2], 99)
    height = float(z_high - z_low)
    
    # Extents and width
    x_min, x_max = np.percentile(points[:, 0], 1), np.percentile(points[:, 0], 99)
    y_min, y_max = np.percentile(points[:, 1], 1), np.percentile(points[:, 1], 99)
    width_x = float(x_max - x_min)
    width_y = float(y_max - y_min)
    max_width = max(width_x, width_y)
    
    # Voxel occupancy volume
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    num_voxels = len(voxel_grid.get_voxels())
    voxel_volume = float(num_voxels * (voxel_size ** 3))
    
    # Bounding Box
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    
    metrics = {
        "height_percentile_1_to_99": height,
        "width_x": width_x,
        "width_y": width_y,
        "max_width": max_width,
        "voxel_volume": voxel_volume,
        "num_voxels": num_voxels,
        "voxel_resolution_m": voxel_size,
        "bounding_box_extent": {
            "x": float(extent[0]),
            "y": float(extent[1]),
            "z": float(extent[2])
        },
        "num_points": len(points)
    }
    
    return metrics

def run_metrics_pipeline(input_ply, floor_info_path, output_json, output_norm_ply):
    pcd = o3d.io.read_point_cloud(input_ply)
    
    if Path(floor_info_path).exists():
        with open(floor_info_path, "r") as f:
            floor_data = json.load(f)
            floor_plane = floor_data.get("plane_model")
            
        print("Normalizing coordinates based on floor plane...")
        pcd_norm = normalize_coordinates(pcd, floor_plane)
        
        Path(output_norm_ply).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(output_norm_ply, pcd_norm)
        pcd = pcd_norm
    else:
        print("Warning: Floor plane info not found. Metrics may not represent true physical vertical axes.")
        
    print("Calculating metrics...")
    metrics = calculate_metrics(pcd)
    
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics saved to {output_json}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate tree metrics from point cloud.")
    parser.add_argument("--input_ply", type=str, default="outputs/pointclouds/tree_clean.ply", help="Path to clean tree point cloud")
    parser.add_argument("--floor_info", type=str, default="outputs/metrics/floor_plane.json", help="Path to floor plane json")
    parser.add_argument("--output_json", type=str, default="outputs/metrics/tree_metrics.json", help="Path to output metrics json")
    parser.add_argument("--output_norm_ply", type=str, default="outputs/pointclouds/tree_normalized.ply", help="Path to save normalized point cloud")
    args = parser.parse_args()
    
    run_metrics_pipeline(args.input_ply, args.floor_info, args.output_json, args.output_norm_ply)
