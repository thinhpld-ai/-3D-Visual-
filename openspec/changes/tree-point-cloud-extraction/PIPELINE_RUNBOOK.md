# Tree Point Cloud Extraction Pipeline: Runbook

This document describes how to operate the 3D Tree Extraction pipeline to achieve reliable point cloud extraction and measurement from Azure Kinect `.mkv` captures. 

---

## ⚠️ Prerequisites & Azure Kinect SDK Dependency

The `extract_frames.py` script relies on the **Microsoft Azure Kinect Sensor SDK (v1.4.1+)** to achieve mathematically perfect depth-to-color alignment using factory intrinsics. 
- **Windows**: The SDK `bin` folder must be in your system `PATH`.
- **Linux**: The SDK must be installed properly (`apt install k4a-tools libk4a...`) or available via `LD_LIBRARY_PATH`.
- **macOS**: Native hardware SDK is not officially supported.

To verify your environment is ready to extract frames, run:
```bash
python -c "import pykinect_azure as pykinect; print('PyKinect OK')"
```
If this fails, you cannot extract frames from `.mkv` using this pipeline. You will need to extract them on a compatible machine first.

---

## End-to-End Execution (Recommended)

The safest and most reliable way to run the pipeline is via `main_pipeline.py`. It orchestrates all steps, checks for failures between phases, and ensures all dependent files are successfully generated.

### Execution Command

```bash
python src/main_pipeline.py path/to/your_recording.mkv \
    --output_dir outputs \
    --step 5 \
    --voxel_size 0.01 \
    --dbscan_eps 0.05 \
    --blur_threshold 100.0 \
    --z_min -0.5 \
    --z_max 3.0 \
    --visualize
```

### Data Flow & Outputs

When using `main_pipeline.py`, the following dependency chain is handled automatically:
1. `extract_frames.py` -> Creates `outputs/raw_frames/rgb/` and `depth/`. It also generates `intrinsics.json`.
2. `preprocess.py` -> Filters raw frames into `outputs/processed_frames/`.
3. `registration.py` -> Connects frames and generates `outputs/trajectory/trajectory.json`.
4. `fusion.py` -> Fuses frames using the trajectory into `outputs/pointclouds/scene_clean.ply`.
5. `segmentation.py` -> Crops the scene, removing walls/floor. **This step outputs the isolated `outputs/pointclouds/tree_clean.ply` AND the fitted `outputs/metrics/floor_plane.json`**.
6. `metrics.py` -> **Uses `tree_clean.ply` AND `floor_plane.json`** to rotate the tree vertically. **This step generates `outputs/pointclouds/tree_normalized.ply`** and `outputs/metrics/tree_metrics.json`.

*Note: `main_pipeline.py` contains safety checks to abort immediately if any step fails to produce the expected output quantities.*

---

## Technical Parameter Guide

### Pose Graph Tuning (`--step`)
- `--step` decides frame decimation. 
- **`5` is the recommended balance**. It skips minor jitters and makes the pose graph faster and often more stable.
- Try `1` (all frames) if you walked very fast (low overlap). It's slower but has more data.
- Try `10` if movement was extremely slow/smooth and drift is occurring.

### Voxel Fusion (`--voxel_size`)
- `0.01` (1cm) is the **recommended default**.
- Try `0.005` (5mm) for extremely intricate branches, *if* your machine has enough RAM and time.
- Try `0.02` (2cm) for giant trees, lower spec machines, or if you only need rough volume. 

### Segmentation Cropping (`--z_min`, `--z_max`)
Instead of editing source code, restrict the tree extraction box via CLI.
- Ensure `--z_min` and `--z_max` contain the tree topology but clip the ceiling or deep sub-floor noise.

### Blur Threshold (`--blur_threshold`)
This controls the image sharpness filter (Variance of Laplacian).
- **Metric meaning:** A *higher* score means a *sharper* image. A score of `0` is completely blurry.
- **Filtering behavior:** If `score < threshold`, the frame is REJECTED.
- **Tuning:** If the camera moved very rapidly causing motion blur, *increase* to `150.0` to reject more blurry frames. If the environment is naturally low-texture and frames are incorrectly rejected, *decrease* to `50.0`.

### DBSCAN Segmentation (`--dbscan_eps`)
Relying solely on "largest cluster DBSCAN" is risky if trees touch walls/pots. 
- Default is `0.05`.
- If the tree branches break apart into separate clusters, *increase* to `0.08`.
- If the tree is fusing with a nearby wall or person, *decrease* to `0.03` or use tighter Z-crops.

---

## 🚑 Quick Debug Checklist

If the final result looks bad or fails:

### Point cloud is warped or drifted (Registration Failure)
- **Check overlap:** Ensure the camera moved slowly and looked at the tree smoothly.
- **Tune:** Increase `--step 5` to `8` to find wider baselines.
- **Blur:** Increase `--blur_threshold 120` to ensure no blurry frames confused the odometry.
- **Check Output:** Open `trajectory.json` and ensure there are actually poses calculated.

### Tree is cut into pieces or missing branches (Segmentation Failure)
- **Tune:** Increase `--dbscan_eps 0.08`. 
- **Check Fusion:** Did fusion actually capture the branches? If the branches are smaller than 1cm, decrease `--voxel_size 0.005`.
- **Check Crop:** Expand `--z_max 4.0` if the top of the taller tree was chopped off.

### Tree is attached to the floor/wall (Segmentation Failure)
- **Tune:** Decrease `--dbscan_eps`. 
- **Check Crop:** Make `--z_min` and `--z_max` tighter around the tree trunk.

### Registration/Fusion crashes or gives empty clouds
- Ensure depth data is scaled natively in millimeters (`1000.0` scale factor factor).
- `raw_frames/rgb` must be 8-bit `.png`, and `raw_frames/depth` must be 16-bit `.png` perfectly matched in pairs by filename.
- Verify `intrinsics.json` matches your camera.
