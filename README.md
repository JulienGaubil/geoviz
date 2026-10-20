# Geometry Visualization

Browser-based, notebook compatible, interactive 3D/mesh plotting utils.

## Quick visualization example

```python
import numpy as np

from dev.geoviz import plot_pointcloud

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

fig = plot_pointcloud(
    points,
    scalar_fields=scalar_fields,
    opacity=0.7,
    cmap="viridis",
    title="Noisy Sphere",
    label="Sphere cloud",
)

# Opens in your browser when run as a script.
fig.show(in_browser=True)

# In a notebook, render inline without spawning a local server.
# You can also just evaluate `fig` in the last cell output.
fig.show(inline=True)
```

Point and mesh `scalar_fields` likewise accept either shape `(N,)` scalars or
shape `(N, 3)` explicit RGB colors. Use `cmap="uniform"` for direct RGB fields.

## Edge fields

Point-cloud graphs and meshes can carry one scalar or explicit RGB value per
edge. The selected
edge field colors each complete segment, and it is also available in the filter
panel as `Edge: <field name>`. Edge filters hide
individual edges without hiding the surface or endpoint vertices; point and
position filters still hide edges whose endpoints do not both pass.

```python
edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)
edge_taylor = np.array([0.08, 0.42, 0.91], dtype=np.float32)

fig = plot_pointcloud(
    points[:3],
    edges=edges,
    edge_scalar_fields={"normalized Taylor": edge_taylor},
    cmap="viridis",
)
fig.show(inline=True)
```

Mesh connectivity comes only from `faces`. Use `mesh_edges(faces)` to obtain the
canonical lexicographically ordered unique undirected edges on which mesh edge
fields are defined:

```python
from dev.geoviz import mesh_edges, plot_mesh

unique_edges = mesh_edges(faces)
edge_lengths = np.linalg.norm(
    vertices[unique_edges[:, 1]] - vertices[unique_edges[:, 0]], axis=1
)

fig = plot_mesh(
    vertices,
    faces,
    scalar_fields={"normal RGB": normal_rgb},
    edge_scalar_fields={"3D edge length": edge_lengths},
    cmap="viridis",
    show_edges=True,
)
```

For meshes, `edge_scalar_fields` arrays must have `len(mesh_edges(faces))`
rows in that exact order. Use shape `(E,)` for scalars or `(E, 3)` with
`cmap="uniform"` for direct edge colors. Edges removed because an endpoint is invalid are
removed from every edge field as well. The explicit `edges=` argument belongs
only to point-cloud graphs, whose connectivity is not otherwise defined. Such
graph data cannot be combined with `downsampling_factor > 1`; subsample and
remap the graph before plotting.

Activating a mesh edge field always displays its edges. While it is active, the
checked `Show edges` control is disabled. Deactivating it or selecting a vertex
or face field restores the previous `Show edges` choice.

## Face fields

Meshes can carry one scalar value per input face. Triangle fields are displayed
with a constant color over each triangle; quad fields use the same color for
both triangles generated for display.

```python
mesh_vertices = points[:4]
faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
triangle_quality = np.asarray([0.2, 0.8], dtype=np.float32)

fig = plot_mesh(
    mesh_vertices,
    faces,
    scalar_fields={"height": mesh_vertices[:, 2]},
    face_scalar_fields={"quality": triangle_quality},
    cmap="viridis",
)
```

`face_scalar_fields` accepts either one scalar or one RGB triplet for every row
of `faces`. RGB fields should use `cmap="uniform"` and provide direct categorical
or other non-colormapped triangle colors. If invalid
vertices cause faces to be removed, the corresponding values are removed from
every face field too. The layer controls distinguish Point/Vertex, Edge, and
Triangle/Face fields, but give each association the same field selector,
colormap, distribution, and clipping controls. Vertex and face fields share the
mesh surface, and field-association radio controls are mutually exclusive within
each layer. Independent layers can display fields simultaneously. Clicking an
active radio again leaves that layer's fields unselected and restores its
original mesh or point/edge colors.

All scalar associations are available to filters. Vertex filters remove
incident edges and faces. On a mesh, an edge filter removes the rejected mesh
edges and every incident face; face filters remove only faces.
New and reset rules span their complete range and are therefore treated as
non-restrictive until a slider is moved. Rule controls update immediately;
geometry filtering is queued for the next frame only when visibility can change.

## Large geometry

Geometry is kept as typed buffers and embedded in independently gzip-compressed
chunks so the viewer remains self-contained when a cluster browser bridge opens
it as a local `file://` document. Point positions remain `float32`, indices
remain `uint32`, and display colors are packed as `uint8`. No automatic
downsampling is applied, so clouds with 10 million points retain every input
point and color.

The viewer also loads hidden layers only when they are enabled, renders static
views on demand instead of continuously, and avoids building filter copies,
and mesh edge buffers until those features are requested. Hover inspection is
removed: cursor movement performs no geometry picking, BVH construction, or
redraw. Camera interaction, colors, and filtering remain available at full
resolution. Rotation uses a separate GPU depth pick only on primary-button down.

The default one-pixel mesh-edge overlay always follows the complete visible
face topology, independently of the selected field. It reuses
the mesh position buffer and splits indexed lines into bounded face chunks so
large meshes do not require one oversized GPU buffer. The unique mesh-edge
domain is used for coloring while an edge field is active, and for topology
while an edge filter is active. Meshes and point-cloud graphs both expose an
edge-thickness control from 1 to 12 in steps of 0.5. Wider screen-space mesh
edges require expanded endpoint attributes, which are built in bounded chunks;
the default one-pixel representation remains substantially cheaper.

For meshes above the picking threshold, camera drag and wheel interaction use a
10,000-vertex point proxy so viewpoint updates do not redraw millions of
triangles per input frame. The exact mesh is restored after interaction stops.

Browser and inline notebook display use the same self-contained document; no
geometry fetches or live buffer server connection are required after the HTML
has loaded.

The animated camera button beside the day/night toggle opens the operating
system's Save dialog for the current view. It renders at up to four times the
viewport dimensions, subject to the GPU's renderbuffer limit and a 64-megapixel
ceiling, and saves a compact WebP image at quality 0.92. It captures only the
geometry canvas, excluding the commands, filters, information display,
tooltips, and top action buttons. The native picker requests the Downloads
folder initially, although the browser or operating system may retain another
user-selected location. Browsers without the native file picker ask before
falling back to their normal download mechanism.

## Mixed Mesh + Point Cloud

You can combine both kinds of geometry in the same viewer by reusing `init_fig`:

```python
from dev.geoviz import plot_mesh, plot_pointcloud

fig = plot_mesh(vertices, faces, label="Surface")
fig = plot_pointcloud(points, init_fig=fig, label="Samples")
fig.show(in_browser=True)
```

## Quick mesh example

```python
import numpy as np

from dev.geoviz import plot_mesh, plot_trimesh

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

fig = plot_mesh(
    vertices,
    faces,
    scalar_fields=scalar_fields,
    opacity=0.75,
    cmap="plasma",
    title="Cube Mesh",
    label="Cube",
)
fig.show(in_browser=True)

# Equivalent convenience alias when your mesh is explicitly triangular.
fig = plot_trimesh(vertices, faces, label="Cube")

# In a notebook, render inline without spawning a local server.
# You can also just evaluate `fig` in the last cell output.
fig.show(inline=True)
```
