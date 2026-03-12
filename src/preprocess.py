import os
import argparse
import glob
import cv2
import numpy as np
import shutil
from pathlib import Path

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def preprocess_frames(input_dir, output_dir, blur_threshold=100.0, max_depth=4000, min_depth=200):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    in_rgb_dir = input_dir / "rgb"
    in_depth_dir = input_dir / "depth"
    
    out_rgb_dir = output_dir / "rgb"
    out_depth_dir = output_dir / "depth"
    
    if not in_rgb_dir.exists() or not in_depth_dir.exists():
        raise FileNotFoundError("Input RGB or Depth directory does not exist.")
        
    out_rgb_dir.mkdir(parents=True, exist_ok=True)
    out_depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy intrinsics if exists
    intrinsics_file = input_dir / "intrinsics.json"
    if intrinsics_file.exists():
        shutil.copy(intrinsics_file, output_dir / "intrinsics.json")
        
    rgb_files = sorted(glob.glob(str(in_rgb_dir / "*.png")))
    depth_files = sorted(glob.glob(str(in_depth_dir / "*.png")))
    
    if len(rgb_files) != len(depth_files):
        print("Warning: Number of RGB and depth images do not match.")
        
    accepted = 0
    rejected = 0
    
    for rgb_path, depth_path in zip(rgb_files, depth_files):
        filename = Path(rgb_path).name
        
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        if rgb is None or depth is None:
            rejected += 1
            print(f"Skipping {filename}: Could not read image.")
            continue
            
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        blur_score = variance_of_laplacian(gray)
        
        if blur_score < blur_threshold:
            rejected += 1
            print(f"Rejecting {filename} (Blur Score: {blur_score:.2f} < {blur_threshold})")
            continue
            
        # Depth filtering
        depth_filtered = depth.copy()
        depth_filtered[depth_filtered > max_depth] = 0
        depth_filtered[depth_filtered < min_depth] = 0
        
        valid_depth_ratio = np.count_nonzero(depth_filtered) / depth_filtered.size
        if valid_depth_ratio < 0.1:
            rejected += 1
            print(f"Rejecting {filename} (Too few valid depth pixels)")
            continue
            
        cv2.imwrite(str(out_rgb_dir / filename), rgb)
        cv2.imwrite(str(out_depth_dir / filename), depth_filtered)
        accepted += 1
        
        if accepted > 0 and accepted % 100 == 0:
            print(f"Processed {accepted} valid frames...")
            
    print(f"Preprocessing complete. Accepted: {accepted}, Rejected: {rejected}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess extracted RGB-D frames.")
    parser.add_argument("--input_dir", type=str, default="data/extracted", help="Path to extracted frames")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Path to save processed frames")
    parser.add_argument("--blur_threshold", type=float, default=100.0, help="Minimum variance of Laplacian for blur detection")
    parser.add_argument("--max_depth", type=int, default=4000, help="Maximum valid depth in mm")
    parser.add_argument("--min_depth", type=int, default=200, help="Minimum valid depth in mm")
    args = parser.parse_args()
    
    preprocess_frames(args.input_dir, args.output_dir, args.blur_threshold, args.max_depth, args.min_depth)
