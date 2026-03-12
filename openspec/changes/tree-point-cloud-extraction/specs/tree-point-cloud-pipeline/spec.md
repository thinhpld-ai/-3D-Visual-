## ADDED Requirements

### Requirement: MKV File Ingestion
The system SHALL extract synchronized RGB and depth frames from Azure Kinect `.mkv` files, including camera intrinsics.

#### Scenario: Valid MKV Extraction
- **WHEN** a valid `.mkv` file is processed
- **THEN** the system outputs a directory of aligned RGB and depth images alongside an `intrinsics.json` file.

### Requirement: Scene Registration and Fusion
The system SHALL estimate the camera trajectory and fuse the depth maps into a single global 3D point cloud.

#### Scenario: Odometry and Pose Graph Optimization
- **WHEN** a sequence of RGB-D frames is provided
- **THEN** the system builds a pose graph, optimizes camera trajectories, and outputs a fused `scene.ply` point cloud.

### Requirement: Tree Segmentation
The system SHALL isolate the tree structure from the room background (e.g., floor, walls, ceiling).

#### Scenario: Plane Removal and Clustering
- **WHEN** the fused `scene.ply` is analyzed
- **THEN** the system sequentially removes large planar surfaces, crops to a region of interest, applies DBSCAN clustering, and outputs `tree_clean.ply`.
