"""
visualize_ply.py - v2
Best-in-class point cloud visualizer using Open3D.

Features:
- Load one or two PLY files (before/after)
- Z-height colormap (hot/rainbow) for depth perception
- Normal-based shading
- Bounding box overlay
- Optional side-by-side view (two windows sequential)
- Also exports height-colored PLY for MeshLab
"""

import argparse
import sys
import numpy as np
import open3d as o3d
from pathlib import Path


COLORMAPS = {
    "rainbow": lambda t: np.column_stack([
        np.clip(1.5 - abs(t * 4 - 3), 0, 1),
        np.clip(1.5 - abs(t * 4 - 2), 0, 1),
        np.clip(1.5 - abs(t * 4 - 1), 0, 1),
    ]),
    "hot": lambda t: np.column_stack([
        np.clip(t * 3,           0, 1),
        np.clip(t * 3 - 1,       0, 1),
        np.clip(t * 3 - 2,       0, 1),
    ]),
    "cool": lambda t: np.column_stack([
        t,
        1.0 - t,
        np.ones_like(t),
    ]),
}


def apply_height_colormap(pcd, colormap_name="rainbow", axis=1):
    """Color points by height along given axis (default Y=1 up)."""
    pts = np.asarray(pcd.points)
    h   = pts[:, axis]
    h_min, h_max = h.min(), h.max()
    t = (h - h_min) / max(h_max - h_min, 1e-6)
    cmap_fn = COLORMAPS.get(colormap_name, COLORMAPS["rainbow"])
    colors  = cmap_fn(t).astype(np.float64)
    colors  = np.clip(colors, 0.0, 1.0)
    pcd_colored = o3d.geometry.PointCloud(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)
    return pcd_colored


def visualize_single(ply_path, window_name, colormap="rainbow", show_normals=False,
                     export_colored=True):
    print(f"\nLoading: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        print("  [WARNING] Empty point cloud, skipping.")
        return

    print(f"  Points: {len(pcd.points):,}")

    # Estimate normals if missing
    if not pcd.has_normals():
        print("  Estimating normals...")
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)

    # Apply height colormap
    pcd_vis = apply_height_colormap(pcd, colormap)

    # Export height-colored PLY for MeshLab
    if export_colored:
        p = Path(ply_path)
        out_path = p.parent / f"{p.stem}_colored.ply"
        o3d.io.write_point_cloud(str(out_path), pcd_vis)
        print(f"  Saved height-colored PLY → {out_path}")

    # Build geometry list
    geoms = [pcd_vis]

    # Bounding box
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0.5, 0)
    geoms.append(aabb)

    # Coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geoms.append(coord)

    print(f"  Opening '{window_name}'... (close window to continue)")
    o3d.visualization.draw_geometries(
        geoms,
        window_name=window_name,
        width=1600, height=900,
        point_show_normal=show_normals
    )


def main():
    parser = argparse.ArgumentParser(description="Best-quality point cloud visualizer v2")
    parser.add_argument("--raw",        type=str, default=None,
                        help="Path to RAW point cloud (before cleaning)")
    parser.add_argument("--clean",      type=str, default=None,
                        help="Path to CLEAN point cloud (after cleaning)")
    parser.add_argument("--input_ply",  type=str, default=None,
                        help="Single PLY file to view")
    parser.add_argument("--colormap",   type=str, default="rainbow",
                        choices=["rainbow", "hot", "cool"],
                        help="Height colormap to apply")
    parser.add_argument("--normals",    action="store_true",
                        help="Show normals in viewer")
    parser.add_argument("--no_export",  action="store_true",
                        help="Skip exporting height-colored PLY")
    parser.add_argument("--metrics",    type=str, default=None,
                        help="(Optional) metrics JSON to print tree stats")
    args = parser.parse_args()

    if args.metrics:
        import json
        try:
            with open(args.metrics) as f:
                m = json.load(f)
            print("\n--- Tree Metrics ---")
            for k, v in m.items():
                print(f"  {k}: {v}")
            print("--------------------")
        except Exception as e:
            print(f"Could not load metrics: {e}")

    export = not args.no_export

    if args.raw:
        visualize_single(args.raw,   "RAW  Point Cloud (before cleaning)",
                         args.colormap, args.normals, export)
    if args.clean:
        visualize_single(args.clean, "CLEAN Point Cloud (after cleaning)",
                         args.colormap, args.normals, export)
    if args.input_ply:
        visualize_single(args.input_ply, Path(args.input_ply).stem,
                         args.colormap, args.normals, export)


if __name__ == "__main__":
    main()
