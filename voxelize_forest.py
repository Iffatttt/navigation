# voxelize_forest.py
import open3d as o3d
import numpy as np

# INPUT_PATH = "forest_49_map_new/dataset_49_points.ply"
INPUT_PATH = INPUT_PATH = "forest_56_glim_map/forest_56_map/dataset_56_points.ply"
VOXEL_SIZE = 0.1

OUTPUT_PCD = "forest_56_voxel_0.1.pcd"
OUTPUT_NPY = "forest_56_voxels_0.1m.npy"


print(f"Loading: {INPUT_PATH}")

pcd = o3d.io.read_point_cloud(INPUT_PATH)

print(f"Loaded {len(pcd.points):,} points")


print(f"\nVoxelizing entire point cloud at {VOXEL_SIZE} m...")

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
    pcd,
    voxel_size=VOXEL_SIZE
)

voxels = voxel_grid.get_voxels()

print(f"Occupied voxels: {len(voxels):,}")

voxel_indices = np.array(
    [voxel.grid_index for voxel in voxels],
    dtype=np.int32
)

np.save(OUTPUT_NPY, voxel_indices)

print(f"Saved voxel indices:")
print(f"  {OUTPUT_NPY}")

downsampled = pcd.voxel_down_sample(
    voxel_size=VOXEL_SIZE
)

o3d.io.write_point_cloud(
    OUTPUT_PCD,
    downsampled
)

print(f"Saved downsampled PCD:")
print(f"  {OUTPUT_PCD}")

print("\nVoxel grid origin:")
print(voxel_grid.origin)

np.save(
    "forest_56_voxel_origin.npy",
    voxel_grid.origin
)

print("Saved:")
print("  forest_56_voxel_origin.npy")


print("\nDone.")