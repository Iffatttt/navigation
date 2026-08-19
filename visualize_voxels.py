#script to visualize the voxels
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

VOXEL_FILE = "forest_56_voxels_0.1m.npy"
ORIGIN_FILE = "forest_56_voxel_origin.npy"

VOXEL_SIZE = 0.1  # 10 cm

# ============================================================
# LOAD VOXELS
# ============================================================

voxels = np.load(VOXEL_FILE)
origin = np.load(ORIGIN_FILE)

print("Loaded:", len(voxels), "occupied voxels")
print("Voxel size:", VOXEL_SIZE, "m")
print("Origin:", origin)

# Convert voxel indices to physical coordinates.
# Use voxel centers, not voxel corners.
xyz = origin + (voxels + 0.5) * VOXEL_SIZE

x = xyz[:, 0]
y = xyz[:, 1]
z = xyz[:, 2]

print("\nForest extent represented by voxels:")
print(f"X: {x.min():.2f} -> {x.max():.2f} m")
print(f"Y: {y.min():.2f} -> {y.max():.2f} m")
print(f"Z: {z.min():.2f} -> {z.max():.2f} m")


# ============================================================
# HEIGHT COLOR
# ============================================================

z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)


# ============================================================
# 1. SIDE VIEW
# ============================================================

plt.figure(figsize=(15, 7))

plt.scatter(
    x,
    z,
    c=z_norm,
    cmap="viridis",
    s=0.15,
    marker=".",
    linewidths=0,
    rasterized=True
)

plt.xlabel("X (m)")
plt.ylabel("Height Z (m)")
plt.title("Entire Forest — 10 cm Voxel Map — Side View")

plt.axis("equal")
plt.tight_layout()

plt.savefig(
    "forest_voxel_side.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 2. TOP VIEW
# ============================================================

plt.figure(figsize=(12, 10))

plt.scatter(
    x,
    y,
    c=z_norm,
    cmap="viridis",
    s=0.15,
    marker=".",
    linewidths=0,
    rasterized=True
)

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Entire Forest — 10 cm Voxel Map — Top View")

plt.axis("equal")
plt.tight_layout()

plt.savefig(
    "forest_voxel_top.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 3. 3D VIEW
# ============================================================

# Matplotlib 3D becomes very slow with 1.47 million points.
# Therefore sample ONLY for visualization.
# The actual .npy remains complete.

MAX_POINTS_3D = 500_000

if len(xyz) > MAX_POINTS_3D:

    print(
        f"\nSampling {MAX_POINTS_3D:,} voxels "
        f"for 3D visualization only..."
    )

    rng = np.random.default_rng(42)

    selected = rng.choice(
        len(xyz),
        MAX_POINTS_3D,
        replace=False
    )

    xyz_plot = xyz[selected]
    color_plot = z_norm[selected]

else:

    xyz_plot = xyz
    color_plot = z_norm


fig = plt.figure(figsize=(14, 11))

ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    xyz_plot[:, 0],
    xyz_plot[:, 1],
    xyz_plot[:, 2],
    c=color_plot,
    cmap="viridis",
    s=0.3,
    marker=".",
    alpha=0.7,
    linewidths=0,
    rasterized=True
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

ax.set_title(
    "Entire Forest — 10 cm Voxel Map — 3D View"
)

# Preserve physical proportions
ax.set_box_aspect((
    x.max() - x.min(),
    y.max() - y.min(),
    z.max() - z.min()
))

# Camera angle
ax.view_init(
    elev=25,
    azim=-60
)

plt.tight_layout()

plt.savefig(
    "forest_voxel_3d.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nDone.")
print("Created:")
print(" 56 forest_voxel_side.png")
print(" 56 forest_voxel_top.png")
print(" 56 forest_voxel_3d.png")