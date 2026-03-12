import argparse
import open3d as o3d
import numpy as np
import json
from pathlib import Path

def multi_plane_removal(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000, num_planes=3):
    remaining_cloud = pcd
    planes = []
    
    for i in range(num_planes):
        if len(remaining_cloud.points) < ransac_n * 10:
            break
            
        plane_model, inliers = remaining_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        [a, b, c, d] = plane_model
        print(f"Plane {i+1} equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        
        inlier_cloud = remaining_cloud.select_by_index(inliers)
        outlier_cloud = remaining_cloud.select_by_index(inliers, invert=True)
        
        planes.append((plane_model, inlier_cloud))
        remaining_cloud = outlier_cloud
        
    return remaining_cloud, planes

def crop_center_region(pcd, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), z_range=(0.0, 3.0)):
    points = np.asarray(pcd.points)
    
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    
    cropped = pcd.select_by_index(np.where(mask)[0])
    return cropped

def extract_largest_cluster(pcd, eps=0.05, min_points=100):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
        
    if len(labels) == 0:
        return None
        
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")
    
    if max_label < 0:
        return pcd
        
    # Find the largest cluster (excluding -1 noise)
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return pcd
        
    largest_cluster_label = unique_labels[np.argmax(counts)]
    print(f"Largest cluster label: {largest_cluster_label}")
    
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    return pcd.select_by_index(largest_cluster_indices)

def run_segmentation(input_ply, output_ply, num_planes=3, dbscan_eps=0.05, min_points=50, crop_radius=1.5, z_min=-0.5, z_max=3.0, floor_info_path="outputs/metrics/floor_plane.json"):
    pcd = o3d.io.read_point_cloud(input_ply)
    
    print("Downsampling for faster segmentation...")
    pcd_down = pcd.voxel_down_sample(voxel_size=0.01)
    
    # 1. Plane removal
    print(f"Removing up to {num_planes} dominant planes...")
    tree_candidates, planes = multi_plane_removal(pcd_down, distance_threshold=0.03, num_planes=num_planes)
    
    # Save the floor plane (assuming the first plane is the floor for now based on Z normal)
    # Actually, we should find the plane whose normal is most parallel to the global Z axis (or Y if Y is up)
    # Let's save the plane parameters so the next step can use it for normalization
    floor_plane = None
    for plane, _ in planes:
        # Check Z component of normal
        normal = np.array(plane[:3])
        if abs(normal[2]) > 0.8 or abs(normal[1]) > 0.8: # Could be Z-up or Y-up
            floor_plane = plane
            break
    
    if floor_plane is not None:
        Path(floor_info_path).parent.mkdir(parents=True, exist_ok=True)
        with open(floor_info_path, "w") as f:
            json.dump({"plane_model": list(floor_plane)}, f)
        print(f"Saved floor plane parameters to {floor_info_path}")
    
    # 2. Crop Region of Interest
    # The camera starts at origin, so the tree is likely between 0 and a few meters in front of it.
    print(f"Cropping point cloud within radius {crop_radius}m and Z[{z_min}, {z_max}]...")
    tree_candidates = crop_center_region(tree_candidates, 
                                        x_range=(-crop_radius, crop_radius), 
                                        y_range=(-crop_radius, crop_radius), 
                                        z_range=(z_min, z_max))
                                        
    # 3. DBSCAN
    print("Extracting the largest contiguous cluster (DBSCAN)...")
    tree_clean = extract_largest_cluster(tree_candidates, eps=dbscan_eps, min_points=min_points)
    
    if tree_clean is None or len(tree_clean.points) == 0:
        print("Warning: No valid tree cluster found.")
        return
        
    print("Saving clean tree point cloud...")
    o3d.io.write_point_cloud(output_ply, tree_clean)
    print(f"Saved {len(tree_clean.points)} points to {output_ply}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment tree point cloud from scene.")
    parser.add_argument("--input_ply", type=str, default="outputs/pointclouds/scene_clean.ply", help="Path to scene point cloud")
    parser.add_argument("--output_ply", type=str, default="outputs/pointclouds/tree_clean.ply", help="Path to save tree point cloud")
    parser.add_argument("--num_planes", type=int, default=3, help="Number of planes to remove")
    parser.add_argument("--eps", type=float, default=0.05, help="DBSCAN epsilon")
    parser.add_argument("--min_points", type=int, default=50, help="DBSCAN min points")
    parser.add_argument("--z_min", type=float, default=-0.5, help="Minimum Z for crop region")
    parser.add_argument("--z_max", type=float, default=3.0, help="Maximum Z for crop region")
    parser.add_argument("--floor_info_path", type=str, default="outputs/metrics/floor_plane.json", help="Path to save extracted floor plane")
    args = parser.parse_args()
    
    Path(args.output_ply).parent.mkdir(parents=True, exist_ok=True)
    run_segmentation(args.input_ply, args.output_ply, args.num_planes, args.eps, args.min_points, 1.5, args.z_min, args.z_max, args.floor_info_path)
