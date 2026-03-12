## ADDED Requirements

### Requirement: Coordinate Normalization
The system SHALL align the segmented tree point cloud such that the floor plane is parallel to the XY plane and the Z-axis represents the true vertical.

#### Scenario: Normalizing Tree Orientation
- **WHEN** `tree_clean.ply` and the floor plane normal are provided
- **THEN** the system rotates and translates the point cloud to be vertically aligned, originating from a logical Z=0 baseline.

### Requirement: Tree Height Calculation
The system SHALL calculate the robust height of the tree using Z-axis percentiles to ignore outlier points.

#### Scenario: Percentile Height Output
- **WHEN** a normalized tree point cloud is analyzed
- **THEN** the system computes the total height as the difference between the 99th and 1st percentiles of the Z coordinates.

### Requirement: Voxel Volume Calculation
The system SHALL estimate the geometric volume of the tree canopy using voxel occupancy counting, preferring this over convex hull approximations.

#### Scenario: Voxel Counting
- **WHEN** a tree point cloud and a specified voxel resolution are provided
- **THEN** the system computes the volume by counting occupied voxels and multiplying by the unit voxel volume.
