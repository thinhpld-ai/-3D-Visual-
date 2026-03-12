import argparse
import open3d as o3d
import json
import numpy as np

def visualize_results(cloud_path, metrics_path=None):
    pcd = o3d.io.read_point_cloud(cloud_path)
    
    geometries = [pcd]
    
    if metrics_path:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
        # Draw bounding box
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)
        geometries.append(aabb)
        
        # We could also render text or lines, but Open3D's basic visualizer 
        # is somewhat limited for text. We'll stick to geometry overlays.
        
        print("\n--- Tree Metrics ---")
        print(f"Height: {metrics.get('height_percentile_1_to_99', 0):.3f} m")
        print(f"Max Width: {metrics.get('max_width', 0):.3f} m")
        print(f"Voxel Volume: {metrics.get('voxel_volume', 0):.5f} m³")
        print("--------------------\n")

    # Add coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geometries.append(coord)
        
    print("Close the visualization window to continue...")
    o3d.visualization.draw_geometries(geometries, window_name="Tree Point Cloud Visualization", width=1280, height=720)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize tree point cloud and metrics.")
    parser.add_argument("--input_ply", type=str, default="outputs/pointclouds/tree_normalized.ply", help="Path to point cloud")
    parser.add_argument("--metrics", type=str, default="outputs/metrics/tree_metrics.json", help="Path to metrics JSON")
    args = parser.parse_args()
    
    visualize_results(args.input_ply, args.metrics)
