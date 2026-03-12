import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_step(script_name, args_list, step_desc):
    print(f"\n{'='*50}")
    print(f"Step: {step_desc}")
    print(f"{'='*50}")
    
    cmd = [sys.executable, f"src/{script_name}"] + args_list
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_desc}' failed with return code {result.returncode}.")
        sys.exit(result.returncode)
    print(f"[SUCCESS] Step '{step_desc}' completed.\n")

def main():
    parser = argparse.ArgumentParser(description="End-to-End Tree Point Cloud Extraction Pipeline")
    parser.add_argument("mkv_file", type=str, help="Path to input .mkv file from Azure Kinect")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base directory for all outputs")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--step", type=int, default=1, help="Frame step size for registration/fusion")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size for TSDF fusion")
    parser.add_argument("--dbscan_eps", type=float, default=0.05, help="DBSCAN epsilon for tree segmentation")
    parser.add_argument("--blur_threshold", type=float, default=100.0, help="Variance of Laplacian threshold for blur filtering")
    parser.add_argument("--z_min", type=float, default=-0.5, help="Minimum Z for crop region")
    parser.add_argument("--z_max", type=float, default=3.0, help="Maximum Z for crop region")
    parser.add_argument("--skip_extract", action="store_true", help="Skip frame extraction step and use existing raw frames")
    parser.add_argument("--visualize", action="store_true", help="Launch visualizer at the end")
    
    args = parser.parse_args()
    
    output_base = Path(args.output_dir)
    raw_dir = output_base / "raw_frames"
    processed_dir = output_base / "processed_frames"
    traj_dir = output_base / "trajectory"
    pc_dir = output_base / "pointclouds"
    metrics_dir = output_base / "metrics"
    
    for d in [raw_dir, processed_dir, traj_dir, pc_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    scene_ply = pc_dir / "scene_clean.ply"
    tree_ply = pc_dir / "tree_clean.ply"
    norm_ply = pc_dir / "tree_normalized.ply"
    metrics_json = metrics_dir / "tree_metrics.json"
    floor_info = metrics_dir / "floor_plane.json"

    # Step 1: Extract Frames
    if not args.skip_extract:
        args_extract = [args.mkv_file, "--output_dir", str(raw_dir)]
        if args.max_frames:
            args_extract.extend(["--max_frames", str(args.max_frames)])
        run_step("extract_frames.py", args_extract, "Extract RGB-D Frames")
    else:
        print("\n==================================================")
        print("Step: Using pre-extracted RGB-D Frames")
        print("==================================================")
        
    rgb_raw_count = len(list((raw_dir / "rgb").glob("*.png")))
    if rgb_raw_count == 0:
        print("[FATAL ERROR] Frame extraction failed: No RGB frames found.")
        sys.exit(1)
    
    # Step 2: Preprocess Frames
    args_preprocess = ["--input_dir", str(raw_dir), "--output_dir", str(processed_dir), 
                       "--blur_threshold", str(args.blur_threshold)]
    run_step("preprocess.py", args_preprocess, "Preprocess and Filter Frames")
    
    rgb_proc_count = len(list((processed_dir / "rgb").glob("*.png")))
    print(f"[INFO] Frames kept after preprocessing: {rgb_proc_count} / {rgb_raw_count}")
    if rgb_proc_count < 20:
        print("[FATAL ERROR] Preprocessing filtered out too many frames. Pipeline cannot continue reliably.")
        sys.exit(1)
    
    # Step 3: Registration
    args_registration = ["--input_dir", str(processed_dir), "--output_dir", str(traj_dir), "--step", str(args.step)]
    run_step("registration.py", args_registration, "Odometry and Pose Graph Registration")
    
    traj_json = traj_dir / "trajectory.json"
    if not traj_json.exists():
        print("[FATAL ERROR] Registration failed: trajectory.json was not created.")
        sys.exit(1)
    
    # Step 4: Fusion
    args_fusion = ["--input_dir", str(processed_dir), "--trajectory", str(traj_json), 
                   "--output_ply", str(scene_ply), "--voxel_size", str(args.voxel_size)]
    run_step("fusion.py", args_fusion, "TSDF Scene Fusion")
    
    if not scene_ply.exists():
        print("[FATAL ERROR] Fusion failed: scene_clean.ply was not created.")
        sys.exit(1)
    
    # Step 5: Segmentation
    args_seg = ["--input_ply", str(scene_ply), "--output_ply", str(tree_ply), 
                "--eps", str(args.dbscan_eps), "--z_min", str(args.z_min), 
                "--z_max", str(args.z_max), "--floor_info_path", str(floor_info)]
    run_step("segmentation.py", args_seg, "Tree Segmentation")
    
    if not tree_ply.exists() or not floor_info.exists():
        print("[FATAL ERROR] Segmentation failed: Could not produce tree_clean.ply or floor_plane.json.")
        sys.exit(1)
    
    # Step 6: Metrics
    args_metrics = ["--input_ply", str(tree_ply), "--floor_info", str(floor_info),
                    "--output_json", str(metrics_json), "--output_norm_ply", str(norm_ply)]
    run_step("metrics.py", args_metrics, "Coordinate Normalization and Metrics")
    
    print("\n==================================================")
    print("PIPELINE COMPLETE")
    print("==================================================")
    print(f"Final tree point cloud: {norm_ply}")
    print(f"Metrics output: {metrics_json}")
    
    if args.visualize:
        args_vis = ["--input_ply", str(norm_ply), "--metrics", str(metrics_json)]
        run_step("visualization.py", args_vis, "Visualization")

if __name__ == "__main__":
    main()
