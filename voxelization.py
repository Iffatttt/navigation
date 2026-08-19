# import dendromatics as dm
# import open3d as o3d
# import numpy as np

# forest_pcd = o3d.io.read_point_cloud("forest_56_glim_map/dataset_56_points.pcd")
# pcd_arr = np.asarray(forest_pcd.points)
# vox_cloud, vox_to_cloud_ind, n_points_per_vox=dm.primitives.voxel.voxelate(
#     pcd_arr, 0.1, 0.1, n_digits=5, X_field=0, Y_field=1, Z_field=2, with_n_points=True, verbose=True)

# vox_coud = dm.stripe.verticality_clustering(vox_cloud, scale=0.1, vert_threshold=0.7, n_points=120, n_iter=3, resolution_xy=0.1, resolution_z=0.1, n_digits=5)
# vis_pcd = o3d.geometry.PointCloud()
# vis_pcd.points = o3d.utility.Vector3dVector(vox_cloud[:, :3])
# o3d.visualization.draw_geometries([vis_pcd])

import dendromatics as dm
import numpy as np
import open3d as o3d
import os

forest_pcd = o3d.io.read_point_cloud("forest_56_glim_map/dataset_56_points.pcd")
coords = np.asarray(forest_pcd.points)

dtm_cache = "dtm_cache.npy"
if os.path.exists(dtm_cache):
    dtm = np.load(dtm_cache)
    print("Loaded cached DTM")
else:
    dtm = dm.generate_dtm(coords, bSloopSmooth=True, cloth_resolution=0.5)
    dtm = dm.complete_dtm(dtm)
    np.save(dtm_cache, dtm)
    print("Computed and cached DTM")

z0_values = dm.normalize_heights(coords, dtm)
coords = np.append(coords, np.expand_dims(z0_values, axis=1), axis=1)

lower_limit = 0.8
upper_limit = 2.0
stripe = coords[(coords[:, 3] > lower_limit) & (coords[:, 3] < upper_limit), 0:4]
print("Stripe points:", stripe.shape)

stripe_stems = dm.stripe.verticality_clustering(
    stripe,
    scale=0.1,
    vert_threshold=0.7,
    n_points=120,
    n_iter=2,
    resolution_xy=0.1,
    resolution_z=0.1,
)

assigned_cloud, tree_vector, tree_heights = dm.individualize_trees(
    coords,
    stripe_stems,
    stripe_lower_limit=lower_limit,
    stripe_upper_limit=upper_limit,
    h_range=0.8,      # tightened back up: stem must span most of the 2m stripe (real trunks do; canopy fragments don't)
    min_points=30,    # tightened: filters tiny noise blobs from counting as "stems"
    d_max=1.5,        # tightened: real trunk radius, not a 1.5m sweep
    max_dev=20,       # stricter vertical-deviation cutoff, still only affects height-trust flag
)

# dendromatics does NOT reject tilted axes from point assignment — do it yourself
axis_deviation = tree_vector[:, 8]   # degrees from vertical, per tree
valid_tree_ids = tree_vector[tree_vector[:, 8] < 40, 0]   # keep only near-vertical axes

NO_ID = 100000
tree_id_col = 4
trunk_mask = np.isin(assigned_cloud[:, tree_id_col], valid_tree_ids)
trunk_points = assigned_cloud[trunk_mask]

print("Valid trees:", len(valid_tree_ids), "/", tree_vector.shape[0])
print("Trunk points:", trunk_points.shape[0])

dist_col = 5


# Color by tree ID (random color per tree)
unique_ids = np.unique(trunk_points[:, tree_id_col])
colors = np.random.rand(len(unique_ids), 3)
id_to_color = {tid: colors[i] for i, tid in enumerate(unique_ids)}
point_colors = np.array([id_to_color[tid] for tid in trunk_points[:, tree_id_col]])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(trunk_points[:, :3])  # x, y, z
pcd.colors = o3d.utility.Vector3dVector(point_colors)

o3d.visualization.draw_geometries([pcd])


