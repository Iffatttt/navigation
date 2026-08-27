import open3d as o3d
import numpy as np
import matplotlib.cm as cm
import dendromatics as dm
import os
from scipy import ndimage
import matplotlib.pyplot as plt
from dendromatics.sections import fit_circle
import matplotlib.patches as patches

#this gives a black voxel map of the forest similar to glim
forest_pcd = o3d.io.read_point_cloud("forest_56_glim_map/dataset_56_points.pcd")
pcd_arr = np.asarray(forest_pcd.points)

#create dtm and cache
dtm_cache = "dtm_cache.npy"
dtm = dm.generate_dtm(pcd_arr, bSloopSmooth=True, cloth_resolution=0.5)
dtm=dm.complete_dtm(dtm)
np.save(dtm_cache, dtm)

#view dtm by wrapping the nx3 as a point cloud
dtm_pcd = o3d.geometry.PointCloud()
dtm_pcd.points = o3d.utility.Vector3dVector(dtm[:, :3])
dtm_z = dtm[:, 2]
dtm_z_norm = (dtm_z - dtm_z.min()) / (dtm_z.max() - dtm_z.min())
dtm_pcd.colors = o3d.utility.Vector3dVector(cm.viridis(dtm_z_norm)[:, :3])
#o3d.visualization.draw_geometries([dtm_pcd])

#norm height above ground per point
z0= dm.normalize_heights(pcd_arr, dtm)

# if u want a colored voxel grid instead of black
z0_norm = (z0 - z0.min()) / (z0.max() - z0.min())
colors = cm.viridis(z0_norm)[:, :3]   #drop alpha channel
#forest_pcd.colors = o3d.utility.Vector3dVector(colors)

#voxel grid through dendromatics
# vox_cloud, vox_to_cloud_ind, n_points_per_vox=dm.primitives.voxel.voxelate(
#     pcd_arr, 0.1, 0.1, n_digits=5, X_field=0, Y_field=1, Z_field=2, with_n_points=True, verbose=True)
# voxel_grid = o3d.geometry.PointCloud()
# voxel_grid.points = o3d.utility.Vector3dVector(vox_cloud[:, :3])

#voxel grid through open3d
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(forest_pcd, voxel_size = 0.1)

#o3d.visualization.draw_geometries([voxel_grid]) 

#to check if each point lies within a voxel:
# inclusion = voxel_grid.check_if_included(o3d.utility.Vector3dVector(pcd_arr))
# for i in inclusion:
#     if i == False:
#         print(i) #prints only if a point is excluded from the grid

#height slab
z_min = 0.4
z_max = 1.37
voxels = voxel_grid.get_voxels()
slab_voxels=[]

#get voxel centers
voxel_centers = np.array([voxel_grid.get_voxel_center_coordinate(v.grid_index) for v in voxels])

#normalize height for voxel centers and check voxels that lie within that z range
slab_voxel_z0 = dm.normalize_heights(voxel_centers, dtm)
slab_mask = (slab_voxel_z0 >= z_min) & (slab_voxel_z0 <= z_max)
slab_voxels = [v for v, keep in zip(voxels, slab_mask) if keep]
#o3d.visualization.draw_geometries([height_voxel_grid]) 
#o3d.visualization.draw_geometries([voxel_grid, height_voxel_grid]) 

#xy projection - get xy indices for each voxel
voxel_indices = np.array([v.grid_index for v in slab_voxels])
xy_indices = voxel_indices[:, :2]

#create 2d grid
i_min, j_min = xy_indices.min(axis=0)
i_max, j_max = xy_indices.max(axis=0)

#fill grid
occupancy_2d = np.zeros((i_max - i_min + 1, j_max - j_min + 1))
#shift indices to start from 0
shifted = xy_indices - [i_min, j_min]
occupancy_2d[shifted[:, 0], shifted[:, 1]] = True

#show grid
# plt.figure(figsize=(10, 10))
# plt.imshow(occupancy_2d, cmap='gray', origin='lower')
# plt.title("2D Occupancy grid")
# plt.show()

closed_grid = ndimage.binary_closing(occupancy_2d, structure=np.ones((3, 3)), iterations=1)
labels_grid, n_components = ndimage.label(closed_grid, structure=np.ones((3, 3), dtype=int))
print(f"Connected components found: {n_components}")

# plt.figure(figsize=(10, 10))
# plt.imshow(closed_grid, cmap='gray', origin='lower')
# plt.title("2D Occupancy grid")
# plt.show()

#show labeled components
plt.figure(figsize=(10, 10))
plt.imshow(labels_grid, cmap='nipy_spectral', origin='lower')
plt.title(f"8-CCL result: {n_components} components")
plt.show()

#centroids for each labeled component
centroids = ndimage.center_of_mass(closed_grid, labels_grid, range(1, n_components + 1))
centroids = np.array(centroids)  # shape (n_components, 2), each row is (row, col)

print(f"Found {len(centroids)} trunk centroids")

#overlay centroids on the grid
# plt.figure(figsize=(10, 10))
# plt.imshow(labels_grid, cmap='nipy_spectral', origin='lower')
# plt.scatter(centroids[:, 1], centroids[:, 0], c='red', marker='x', s=100, label='Centroids')
# plt.title(f"Trunk centroids ({n_components} trees)")
# plt.legend()
# plt.show()

fitted_circles = []  #will hold (center_x_idx, center_y_idx, radius_px, label_id)
ck_rk =[]
for label_id in range(1, n_components + 1):
    ys, xs = np.where(labels_grid == label_id)  # row, col indices of this component's pixels
    
    if len(xs) < 3: #n_min
        continue  #can't fit a circle with fewer than 3 points, skip tiny noise blobs
    
    center, radius = fit_circle(xs.astype(float), ys.astype(float))
    ck_rk.append([center[0], center[1], radius])  # one row: [cx, cy, r]    
    fitted_circles.append((center[0], center[1], radius, label_id))

ck_rk = np.array(ck_rk)
print(f"Fitted {len(fitted_circles)} circles out of {n_components} components")
print(ck_rk)

# visualize: labeled grid + fitted circle outlines
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(labels_grid, cmap='nipy_spectral', origin='lower')

for cx, cy, r, label_id in fitted_circles:
    circle = patches.Circle((cx, cy), r, fill=False, edgecolor='white', linewidth=2)
    ax.add_patch(circle)
    ax.plot(cx, cy, 'r+', markersize=10)

ax.set_title(f"Fitted circles: {len(fitted_circles)} trunks")
plt.show()
