# Geometry Visualization

Browser-based, notebook compatible, interactive 3D/mesh plotting utils.

## Quick visualization example

```python
import numpy as np

from pointcloud_viewer import plot_pointcloud_viewer

# Build a noisy sphere point cloud.
n = 5000
phi = np.random.uniform(0.0, np.pi, n)
theta = np.random.uniform(0.0, 2.0 * np.pi, n)
r = 1.0 + 0.03 * np.random.randn(n)

x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)
points = np.stack([x, y, z], axis=1).astype(np.float32)

# Visualize multiple scalar fields and toggle them in the UI.
scalar_fields = {
    "height_z": points[:, 2],
    "radius": np.linalg.norm(points, axis=1),
}

fig = plot_pointcloud_viewer(
    points,
    scalar_fields=scalar_fields,
    cmap="viridis",
    show_colorbar=True,
    title="Noisy Sphere",
    label="Sphere cloud",
)

# Opens in your browser when run as a script.
fig.show(open_browser=True)
```

## Quick mesh example

```python
import numpy as np

from mesh_viewer import plot_mesh_viewer

# Cube vertices.
vertices = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float32,
)

# 12 triangles (2 per face).
faces = np.array(
    [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [1, 5, 6], [1, 6, 2],  # right
        [2, 6, 7], [2, 7, 3],  # back
        [3, 7, 4], [3, 4, 0],  # left
    ],
    dtype=np.int32,
)

# Per-vertex scalar fields selectable in the viewer.
scalar_fields = {
    "height_z": vertices[:, 2],
    "radius": np.linalg.norm(vertices, axis=1),
}

fig = plot_mesh_viewer(
    vertices,
    faces,
    scalar_fields=scalar_fields,
    cmap="plasma",
    show_colorbar=True,
    title="Cube Mesh",
    label="Cube",
)
fig.show(open_browser=True)
```
