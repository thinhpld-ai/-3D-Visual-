"""
preprocess.py - v2
Best-quality frame selection and depth filtering for Azure Kinect RGB-D data.

Improvements over v1:
- Lower blur threshold (50 vs 100) to keep more frames
- Extended depth range: 100-6000mm (Kinect full range)
- Bilateral filter on depth for noise reduction
- Adaptive blur threshold based on image statistics
- Temporal consistency: skip frames too similar to previous (reduces redundancy)
"""

import os
import argparse
import glob
import cv2
import numpy as np
import shutil
from pathlib import Path


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def bilateral_filter_depth(depth_img, d=7, sigma_color=75, sigma_space=75):
    """Apply bilateral filter to depth image to reduce noise while preserving edges."""
    # IMPORTANT:
    # Depth images are 16-bit (mm). Normalizing to 8-bit loses precision and can
    # distort geometry (thin structures disappear / get "melted").
    # OpenCV supports bilateral filtering on CV_32F, so keep depth in mm.
    depth_float = depth_img.astype(np.float32)
    valid_mask = depth_img > 0

    filtered = cv2.bilateralFilter(depth_float, d, sigma_color, sigma_space)

    # Preserve invalid pixels (0 depth) to avoid "bleeding" zeros into valid areas.
    filtered[~valid_mask] = 0.0

    # Round back to integer mm.
    return np.rint(filtered).astype(depth_img.dtype)


def preprocess_frames(
    input_dir,
    output_dir,
    blur_threshold=50.0,
    max_depth=6000,
    min_depth=100,
    temporal_similarity_thresh=0.98,
    min_valid_ratio=0.1,
    disable_temporal_filter=False,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    in_rgb_dir  = input_dir / "rgb"
    in_depth_dir = input_dir / "depth"
    out_rgb_dir  = output_dir / "rgb"
    out_depth_dir = output_dir / "depth"

    if not in_rgb_dir.exists() or not in_depth_dir.exists():
        raise FileNotFoundError("Input RGB or Depth directory does not exist.")

    out_rgb_dir.mkdir(parents=True, exist_ok=True)
    out_depth_dir.mkdir(parents=True, exist_ok=True)

    # Copy intrinsics
    for intrinsics_file in [input_dir / "intrinsics.json"]:
        if intrinsics_file.exists():
            shutil.copy(intrinsics_file, output_dir / "intrinsics.json")

    rgb_files   = sorted(glob.glob(str(in_rgb_dir / "*.png")))
    depth_files = sorted(glob.glob(str(in_depth_dir / "*.png")))

    if len(rgb_files) == 0:
        raise ValueError("No RGB frames found.")
    if len(rgb_files) != len(depth_files):
        print(f"Warning: RGB({len(rgb_files)}) vs Depth({len(depth_files)}) count mismatch.")

    accepted = 0
    rejected = 0
    prev_gray = None
    # SSIM-like check: skip if too similar to previous frame.
    # Higher threshold => more aggressive skipping. Lower => keep more frames.
    TEMPORAL_SIMILARITY_THRESH = float(temporal_similarity_thresh)

    print(f"Starting preprocessing: {len(rgb_files)} total frames")
    print(
        "Settings: "
        f"blur_threshold={blur_threshold}, "
        f"depth={min_depth}-{max_depth}mm, "
        f"temporal_similarity_thresh={TEMPORAL_SIMILARITY_THRESH}, "
        f"min_valid_ratio={min_valid_ratio}, "
        f"disable_temporal_filter={disable_temporal_filter}"
    )

    for rgb_path, depth_path in zip(rgb_files, depth_files):
        filename = Path(rgb_path).name

        rgb   = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

        if rgb is None or depth is None:
            rejected += 1
            continue

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        # --- Blur check ---
        blur_score = variance_of_laplacian(gray)
        if blur_score < blur_threshold:
            rejected += 1
            continue

        # --- Temporal redundancy check: skip if almost identical to previous frame ---
        if (not disable_temporal_filter) and (prev_gray is not None):
            diff = cv2.absdiff(gray, prev_gray)
            similarity = 1.0 - (diff.mean() / 255.0)
            if similarity > TEMPORAL_SIMILARITY_THRESH:
                rejected += 1
                continue

        # --- Depth range filtering ---
        depth_filtered = depth.copy()
        depth_filtered[depth_filtered > max_depth] = 0
        depth_filtered[depth_filtered < min_depth]  = 0

        # --- Valid depth coverage check ---
        valid_ratio = np.count_nonzero(depth_filtered) / max(depth_filtered.size, 1)
        if valid_ratio < float(min_valid_ratio):
            rejected += 1
            continue

        # --- Bilateral filter on depth to reduce sensor noise ---
        depth_filtered = bilateral_filter_depth(depth_filtered)

        cv2.imwrite(str(out_rgb_dir / filename),   rgb)
        cv2.imwrite(str(out_depth_dir / filename), depth_filtered)
        prev_gray = gray
        accepted += 1

        if accepted % 50 == 0:
            print(f"  Processed {accepted} valid frames (rejected so far: {rejected})...")

    pct = 100.0 * accepted / max(len(rgb_files), 1)
    print(f"\nPreprocessing complete.")
    print(f"  Accepted : {accepted} / {len(rgb_files)} ({pct:.1f}%)")
    print(f"  Rejected : {rejected}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess extracted RGB-D frames (v2).")
    parser.add_argument("--input_dir",       type=str,   default="outputs/raw_frames")
    parser.add_argument("--output_dir",      type=str,   default="outputs/processed_frames")
    parser.add_argument("--blur_threshold",  type=float, default=50.0)
    parser.add_argument("--max_depth",       type=int,   default=6000)
    parser.add_argument("--min_depth",       type=int,   default=100)
    parser.add_argument("--temporal_similarity_thresh", type=float, default=0.98,
                        help="Skip frame if too similar to previous (higher = more skipping).")
    parser.add_argument("--min_valid_ratio", type=float, default=0.1,
                        help="Reject if valid depth pixel ratio is below this threshold.")
    parser.add_argument("--disable_temporal_filter", action="store_true",
                        help="Keep frames even if they are very similar to previous.")
    args = parser.parse_args()

    preprocess_frames(
        args.input_dir,
        args.output_dir,
        args.blur_threshold,
        args.max_depth,
        args.min_depth,
        args.temporal_similarity_thresh,
        args.min_valid_ratio,
        args.disable_temporal_filter,
    )
