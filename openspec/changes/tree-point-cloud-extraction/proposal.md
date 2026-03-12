## Why
Extracting clean 3D point clouds of trees from Azure Kinect DK `.mkv` files is necessary for accurate and stable measurement of geometric parameters (height, width, volume). Relying on complex SLAM repositories often introduces difficult build and operational dependencies. A dedicated, controllable Python and Open3D-based pipeline ensures stable execution, easy debugging, and high-quality outputs suitable for quantitative reporting in research papers.

## What Changes
- Implement a robust 2-tier parameter extraction pipeline using Python and Open3D.
- **Tier 1**: Create a standard Python pipeline to read Azure Kinect data, build RGB-D frames, register and fuse point clouds, filter out the room background, isolate the tree, and compute precise metrics.
- **Tier 2**: (Optional/Future) Incorporate SLAM techniques (Open3D RGB-D odometry + pose graph or ORB-SLAM/BADSLAM) to enhance registration quality if the baseline pipeline requires improvement.
- Deliver 4 main outputs: `scene.ply`, `tree_clean.ply`, `tree_metrics.json/csv`, and visual 3D reports.

## Capabilities

### New Capabilities
- `tree-point-cloud-pipeline`: The end-to-end data processing pipeline for converting raw `.mkv` files into filtered 3D point clouds (`scene.ply` and `tree_clean.ply`).
- `tree-geometric-metrics`: The measurement module responsible for extracting and calculating physical tree parameters such as robust height (percentile-based), width, bounding boxes, and voxel occupancy volume.

### Modified Capabilities

## Impact
- Creates a new Python project structure under `src/` spanning extraction, preprocessing, registration, fusion, segmentation, metrics, and visualization.
- Introduces core Python dependencies: `open3d`, `opencv-python`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`.
