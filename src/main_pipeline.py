"""
main_pipeline.py - v2
End-to-end pipeline for RGB-D tree point cloud extraction.

Outputs:
  scene_raw.ply    - full fused cloud before cleaning
  scene_clean.ply  - after outlier removal
  scene_segmented.ply  - full scene with tree highlighted green
  tree_clean.ply   - only the tree cluster
  tree_normalized.ply  - tree normalized to ground plane
  *_colored.ply    - height-colormap versions for MeshLab
"""

import argparse
import sys
from pathlib import Path
import subprocess


def run_step(script_name, args_list, step_desc):
    print(f"\n{'='*55}")
    print(f"  {step_desc}")
    print(f"{'='*55}")
    cmd = [sys.executable, f"src/{script_name}"] + args_list
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_desc}' failed (exit {result.returncode}).")
        sys.exit(result.returncode)
    print(f"  [OK] {step_desc}")


def main():
    parser = argparse.ArgumentParser(description="v2 Hi-Quality Tree Point Cloud Pipeline")
    parser.add_argument("mkv_file",          type=str, help="Azure Kinect .mkv file")
    parser.add_argument("--output_dir",      type=str, default="outputs")
    parser.add_argument("--max_frames",      type=int, default=None)
    parser.add_argument("--step",            type=int, default=1,
                        help="Frame step for registration (1 = every frame)")
    parser.add_argument("--voxel_size",      type=float, default=0.008,
                        help="TSDF voxel size in metres (0.008 = 8mm)")
    parser.add_argument("--blur_threshold",  type=float, default=50.0)
    parser.add_argument("--temporal_similarity_thresh", type=float, default=0.98,
                        help="Preprocess: skip frame if too similar to previous (higher = more skipping)")
    parser.add_argument("--min_valid_ratio", type=float, default=0.1,
                        help="Preprocess: reject if valid depth pixel ratio is below this threshold")
    parser.add_argument("--disable_temporal_filter", action="store_true",
                        help="Preprocess: keep frames even if very similar to previous")
    parser.add_argument("--dbscan_eps",      type=float, default=0.05)
    parser.add_argument("--z_min",           type=float, default=-0.5)
    parser.add_argument("--z_max",           type=float, default=4.0)
    parser.add_argument("--keyframe_interval", type=int, default=20,
                        help="Keyframe interval for loop closure detection")
    parser.add_argument("--skip_extract",    action="store_true")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_registration", action="store_true")
    parser.add_argument("--skip_fusion",     action="store_true")
    parser.add_argument("--visualize",       action="store_true",
                        help="Launch visualizer after pipeline")

    args = parser.parse_args()

    base       = Path(args.output_dir)
    raw_dir    = base / "raw_frames"
    proc_dir   = base / "processed_frames"
    traj_dir   = base / "trajectory"
    pc_dir     = base / "pointclouds"
    metrics_dir = base / "metrics"

    for d in [raw_dir, proc_dir, traj_dir, pc_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    raw_ply    = pc_dir / "scene_raw.ply"
    clean_ply  = pc_dir / "scene_clean.ply"
    tree_ply   = pc_dir / "tree_clean.ply"
    norm_ply   = pc_dir / "tree_normalized.ply"
    floor_info = metrics_dir / "floor_plane.json"
    metrics_json = metrics_dir / "tree_metrics.json"
    traj_json  = traj_dir / "trajectory.json"

    # Step 1: Extract
    if not args.skip_extract:
        step_args = [args.mkv_file, "--output_dir", str(raw_dir)]
        if args.max_frames:
            step_args += ["--max_frames", str(args.max_frames)]
        run_step("extract_frames.py", step_args, "Step 1 — Extract RGB-D Frames")
    else:
        print("\n[SKIP] Step 1 — using existing raw frames")

    rgb_raw_count = len(list((raw_dir / "rgb").glob("*.png")))
    if rgb_raw_count == 0:
        print("[FATAL] No raw RGB frames found."); sys.exit(1)
    print(f"  Raw frames: {rgb_raw_count}")

    # Step 2: Preprocess
    if not args.skip_preprocess:
        run_step("preprocess.py", [
            "--input_dir",      str(raw_dir),
            "--output_dir",     str(proc_dir),
            "--blur_threshold", str(args.blur_threshold),
            "--max_depth",      "6000",
            "--min_depth",      "100",
            "--temporal_similarity_thresh", str(args.temporal_similarity_thresh),
            "--min_valid_ratio", str(args.min_valid_ratio),
            *(
                ["--disable_temporal_filter"]
                if args.disable_temporal_filter
                else []
            ),
        ], "Step 2 — Preprocess & Filter Frames (v2)")
    else:
        print("\n[SKIP] Step 2 — using existing processed frames")

    rgb_proc = len(list((proc_dir / "rgb").glob("*.png")))
    print(f"  Frames after preprocess: {rgb_proc} / {rgb_raw_count}")
    if rgb_proc < 10:
        print("[FATAL] Too few frames after preprocessing."); sys.exit(1)

    # Step 3: Registration
    if not args.skip_registration:
        run_step("registration.py", [
            "--input_dir",          str(proc_dir),
            "--output_dir",         str(traj_dir),
            "--step",               str(args.step),
            "--keyframe_interval",  str(args.keyframe_interval),
        ], "Step 3 — Odometry + Loop Closure Registration (v2)")
    else:
        print("\n[SKIP] Step 3 — using existing trajectory")

    if not traj_json.exists():
        print("[FATAL] trajectory.json not found."); sys.exit(1)

    # Step 4: Fusion → scene_raw.ply + scene_clean.ply
    if not args.skip_fusion:
        run_step("fusion.py", [
            "--input_dir",  str(proc_dir),
            "--trajectory", str(traj_json),
            "--output_ply", str(clean_ply),
            "--voxel_size", str(args.voxel_size),
        ], "Step 4 — TSDF Fusion v2 (raw + clean)")
    else:
        print("\n[SKIP] Step 4 — using existing point clouds")

    if not clean_ply.exists():
        print("[FATAL] scene_clean.ply not created."); sys.exit(1)

    # Step 5: Segmentation
    run_step("segmentation.py", [
        "--input_ply",      str(clean_ply),
        "--output_ply",     str(tree_ply),
        "--eps",            str(args.dbscan_eps),
        "--z_min",          str(args.z_min),
        "--z_max",          str(args.z_max),
        "--floor_info_path", str(floor_info),
    ], "Step 5 — Tree Segmentation")

    # Step 6: Metrics
    if tree_ply.exists() and floor_info.exists():
        run_step("metrics.py", [
            "--input_ply",      str(tree_ply),
            "--floor_info",     str(floor_info),
            "--output_json",    str(metrics_json),
            "--output_norm_ply", str(norm_ply),
        ], "Step 6 — Metrics & Normalization")

    print(f"\n{'='*55}")
    print("  PIPELINE COMPLETE — v2")
    print(f"{'='*55}")
    for label, path in [
        ("Raw scene (before clean)", raw_ply),
        ("Clean scene",              clean_ply),
        ("Tree only",                tree_ply),
        ("Tree normalized",          norm_ply),
        ("Metrics",                  metrics_json),
    ]:
        status = "✓" if path.exists() else "✗ MISSING"
        print(f"  [{status}] {label}: {path}")

    # Visualization
    if args.visualize:
        vis_args = []
        if raw_ply.exists():
            vis_args += ["--raw", str(raw_ply)]
        if clean_ply.exists():
            vis_args += ["--clean", str(clean_ply)]
        if vis_args:
            run_step("visualize_ply.py", vis_args,
                     "Visualization — RAW vs CLEAN side by side")


if __name__ == "__main__":
    main()
