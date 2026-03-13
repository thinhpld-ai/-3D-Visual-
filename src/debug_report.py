import os
import json
import glob
import cv2
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

def analyze_depth_quantization(depth_dir):
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    if not depth_files:
        return None
    
    quant_steps = []
    # Sample 10 frames to get a median statistic
    sample_files = depth_files[::max(1, len(depth_files) // 10)]
    
    for f in sample_files:
        depth = cv2.imread(f, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            continue
        
        unique_vals = np.unique(depth)
        unique_vals = unique_vals[unique_vals > 0] # ignore zero
        
        if len(unique_vals) < 2:
            continue
            
        diffs = np.diff(unique_vals)
        quant_steps.append(np.min(diffs))
    
    if not quant_steps:
        return None
        
    return {
        "min": float(np.min(quant_steps)),
        "median": float(np.median(quant_steps)),
        "max": float(np.max(quant_steps))
    }

def analyze_trajectory_pca(trajectory_path):
    if not os.path.exists(trajectory_path):
        return None
    
    with open(trajectory_path, "r") as f:
        data = json.load(f)
    
    poses = np.array(data["poses"]) # (N, 4, 4)
    translations = poses[:, :3, 3]
    
    pca = PCA(n_components=3)
    pca.fit(translations)
    
    return pca.explained_variance_ratio_.tolist()

def main():
    print("=== Pipeline Debug Report ===\n")
    
    raw_depth_dir = "outputs/raw_frames/depth"
    proc_depth_dir = "outputs/processed_frames/depth"
    traj_path = "outputs/trajectory/trajectory.json"
    
    print("1. Depth Quantization Analysis (Lower is better, goal: ~1mm)")
    raw_stats = analyze_depth_quantization(raw_depth_dir)
    if raw_stats:
        print(f"  RAW Depth       : min={raw_stats['min']:.1f}mm, median={raw_stats['median']:.1f}mm")
    else:
        print("  RAW Depth       : No data")
        
    proc_stats = analyze_depth_quantization(proc_depth_dir)
    if proc_stats:
        print(f"  PROCESSED Depth : min={proc_stats['min']:.1f}mm, median={proc_stats['median']:.1f}mm")
        if proc_stats['median'] > 5:
            print("  [!] WARNING: Strong quantization detected in processed frames.")
    else:
        print("  PROCESSED Depth : No data")
        
    print("\n2. Trajectory PCA Analysis (Higher 2nd axis means more 'circular', goal: 2D-like)")
    pca_ratios = analyze_trajectory_pca(traj_path)
    if pca_ratios:
        print(f"  PCA explained variance ratios: PC1={pca_ratios[0]*100:.1f}%, PC2={pca_ratios[1]*100:.1f}%, PC3={pca_ratios[2]*100:.1f}%")
        if pca_ratios[0] > 0.85:
            print("  [!] WARNING: Trajectory is primarily 1D. High risk of 'stretching' artifacts.")
    else:
        print("  Trajectory      : No data (trajectory.json not found)")

if __name__ == "__main__":
    main()
