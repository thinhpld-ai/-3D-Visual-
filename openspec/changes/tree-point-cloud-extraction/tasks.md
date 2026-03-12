## 1. Setup and Ingestion

- [x] 1.1 Create `extract_frames.py` to convert `.mkv` to RGB/depth directories and save `intrinsics.json`.
- [x] 1.2 Create `preprocess.py` to filter depth images and discard blurry or invalid frames.

## 2. Point Cloud Reconstruction

- [x] 2.1 Implement `registration.py` to perform Open3D RGB-D odometry and build a pose graph between sequential frames.
- [x] 2.2 Add global pose graph optimization with loop closure detection.
- [x] 2.3 Create `fusion.py` to transform all frames into a global coordinate system and fuse them into `scene_clean.ply`.

## 3. Segmentation and Cleanup

- [x] 3.1 Implement `segmentation.py` with multi-plane RANSAC removal to delete floor and walls.
- [x] 3.2 Add spatial cropping and DBSCAN clustering to logically isolate the primary tree mesh into `tree_clean.ply`.

## 4. Measurement and Metrics

- [x] 4.1 Create a coordinate normalization function to align the tree vertically based on the extracted floor plane.
- [x] 4.2 Implement `metrics.py` to calculate percentile-based height, bounding box dimensions, and voxel occupancy volume.
- [x] 4.3 Export the calculated metrics to `tree_metrics.json`.

## 5. Visualization and Reporting

- [x] 5.1 Create `visualization.py` to render the segmented point cloud, bounding boxes, and measurement overlays using Open3D visualization tools.
- [x] 5.2 Build a main pipeline script `main_pipeline.py` to tie all modules together and manage configuration arguments.
