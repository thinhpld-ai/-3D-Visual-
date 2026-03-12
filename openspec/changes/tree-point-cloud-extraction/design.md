## Context
Extracting 3D geometric properties (height, width, volume) of trees from Azure Kinect `.mkv` video captures requires robust scanning and processing. Existing methods leveraging complex external SLAM frameworks can be difficult to build, operate, and debug. We need a stable, easily verifiable Python/Open3D pipeline that prioritizes obtaining a clean tree point cloud for reliable quantitative measurement.

## Goals / Non-Goals

**Goals:**
- Provide a standard Python + Open3D extraction pipeline to turn Azure Kinect `.mkv` files into synchronized RGB and depth frames.
- Reconstruct the visual scene by estimating transformations between frames, building a pose graph, and fusing the point clouds.
- Robustly segment and extract the tree from the surrounding room (floors, walls).
- Ensure the pipeline produces intermediate outputs (`scene.ply`, `tree_clean.ply`) that can be easily visualized and reported on.

**Non-Goals:**
- Creating a real-time SLAM system capable of running at 30 fps on resource-constrained devices.
- Fully automated zero-configuration segmentation (semi-automated checks and threshold tuning are acceptable to ensure research-quality outputs).
- Generating high-fidelity textured 3D meshes (the priority is strictly on the point cloud and its geometric measurements).

## Decisions
- **Python + Open3D Stack**: Chosen over C++ SLAM libraries for ease of debugging, rapid experimentation, and broad data science ecosystem integration (NumPy, SciPy, pandas).
- **Two-Tier Pipeline Strategy**: Develop a baseline focused on pure Python/Open3D RGB-D odometry and pose graph optimization. This allows for early results while keeping the option to use external SLAM trajectories open for future enhancements.
- **Multi-Stage Segmentation Pipeline**: Sequentially remove large planar surfaces (RANSAC) -> crop by bounding region -> apply DBSCAN clustering. This multi-step process is crucial to cleanly isolating the complex geometry of a tree from a standard room background.
- **Coordinate System Normalization before measurement**: Rotating the segmented point cloud so the floor plane is parallel to XY and Z is strictly vertical to ensure measurements represent true physical dimensions.

## Risks / Trade-offs
- [Registration Drift] -> Mitigated by using pose graph optimization, attempting loop closures if a full 360 rotation is captured, and aggressively filtering out blurry or low-overlap frames during preprocessing.
- [Segmentation Leakage (Tree attached to floor/wall)] -> Mitigated by implementing a semi-automated review step allowing users to visually verify and adjust crop boxes and DBSCAN cluster thresholds for challenging corner cases.
