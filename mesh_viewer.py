from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Mapping

import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - optional dependency in some environments
    torch = None

try:
    from .viewer_common import (
        ViewerFigureBase,
        _prepare_color_data,
        _remap_filtered_indices,
        _to_numpy,
        _to_uint8_rgb,
    )
except ImportError:  # pragma: no cover - allow direct module imports from the repo root
    from viewer_common import (
        ViewerFigureBase,
        _prepare_color_data,
        _remap_filtered_indices,
        _to_numpy,
        _to_uint8_rgb,
    )


_AUTO_CMAP = object()


def _encode_array_payload(arr: np.ndarray, *, dtype: np.dtype | str | None = None) -> dict[str, Any]:
    encoded = np.ascontiguousarray(arr, dtype=dtype)
    return {
        "encoding": "base64",
        "dtype": str(encoded.dtype),
        "shape": list(encoded.shape),
        "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def _encode_texture_image(image: Any) -> dict[str, Any]:
    buffer = BytesIO()
    rgba = image.convert("RGBA")
    rgba.save(buffer, format="PNG")
    return {
        "image_data_uri": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        "width": int(rgba.size[0]),
        "height": int(rgba.size[1]),
    }


def _extract_trimesh_texture(mesh: Any) -> dict[str, Any] | None:
    visual = getattr(mesh, "visual", None)
    if visual is None or getattr(visual, "kind", None) != "texture":
        return None

    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "image", None) if material is not None else None
    if uv is None or image is None:
        return None

    uv_arr = np.asarray(uv, dtype=np.float32)
    if uv_arr.ndim != 2 or uv_arr.shape[1] != 2 or uv_arr.shape[0] != len(mesh.vertices):
        return None

    texture = _encode_texture_image(image)
    texture["uv"] = uv_arr
    return texture


def _encode_texture_payload(texture: dict[str, Any] | None, valid_mask: np.ndarray) -> dict[str, Any] | None:
    if not texture:
        return None
    uv = np.asarray(texture.get("uv"), dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] != valid_mask.shape[0]:
        return None
    return {
        "uv": _encode_array_payload(uv[valid_mask] if np.any(valid_mask) else np.zeros((0, 2), dtype=np.float32), dtype=np.float32),
        "image_data_uri": str(texture["image_data_uri"]),
        "width": int(texture.get("width", 0)),
        "height": int(texture.get("height", 0)),
    }


def _resolve_active_cmap(cmap: Any, texture: dict[str, Any] | None) -> str:
    if cmap is _AUTO_CMAP:
        return "texture" if texture is not None else "uniform"
    if not isinstance(cmap, str):
        raise ValueError("cmap must be a string")
    if cmap == "texture" and texture is None:
        return "uniform"
    return cmap


def _uses_raw_colors(cmap: Any) -> bool:
    return cmap is _AUTO_CMAP or cmap in {"texture", "uniform"}


def _prepare_mesh_inputs(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray]:
    raw_vertices = _to_numpy(vertices)
    if raw_vertices is None:
        raise ValueError("vertices cannot be None")
    raw_vertex_shape = tuple(np.asarray(raw_vertices).shape)
    verts = np.asarray(raw_vertices, dtype=np.float32).reshape(-1, 3)

    raw_faces = _to_numpy(faces)
    if raw_faces is None:
        raise ValueError("faces cannot be None")
    tri = np.asarray(raw_faces, dtype=np.int32).reshape(-1, 3)
    valid_vertices = np.isfinite(verts).all(axis=1)
    return verts, raw_vertex_shape, tri, valid_vertices


def _prepare_vertex_colors(
    vertex_color: np.ndarray | torch.Tensor | str,
    *,
    raw_vertex_shape: tuple[int, ...],
    vertex_count: int,
    cmap: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any] | None]:
    if cmap is _AUTO_CMAP:
        cmap = "uniform"
    return _prepare_color_data(
        vertex_color,
        raw_geometry_shape=raw_vertex_shape,
        item_count=vertex_count,
        cmap=cmap,
        color_label="vertex_color",
    )


def _filter_mesh_geometry(
    verts: np.ndarray,
    tri: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if np.any(valid_mask):
        verts = verts[valid_mask]
    else:
        verts = np.zeros((0, 3), dtype=np.float32)

    tri = _remap_filtered_indices(tri, valid_mask)
    return verts, tri


def _encode_mesh_layer(
    *,
    label: str,
    verts: np.ndarray,
    tri: np.ndarray,
    colors: np.ndarray,
    raw_colors: np.ndarray,
    scalar_meta: dict[str, Any] | None,
    scalar_values: np.ndarray | None,
    visible: bool,
    opacity: float,
    show_edges: bool,
    show_vertices: bool,
    texture: dict[str, Any] | None = None,
    active_cmap: str = "uniform",
) -> dict[str, Any]:
    if scalar_values is not None and scalar_values.shape[0] != verts.shape[0]:
        raise ValueError("scalar_values must match filtered vertex count")

    colors_uint8 = _to_uint8_rgb(colors)
    raw_colors_uint8 = _to_uint8_rgb(raw_colors)

    return {
        "layer_kind": "mesh",
        "label": label,
        "vertices": _encode_array_payload(verts, dtype=np.float32),
        "faces": _encode_array_payload(tri, dtype=np.uint32),
        "colors": _encode_array_payload(colors_uint8, dtype=np.uint8),
        "raw_colors": _encode_array_payload(raw_colors_uint8, dtype=np.uint8),
        "vertex_count": int(verts.shape[0]),
        "face_count": int(tri.shape[0]),
        "scalar_meta": scalar_meta,
        "values": None
        if scalar_values is None
        else _encode_array_payload(scalar_values, dtype=np.float32),
        "visible": bool(visible),
        "opacity": float(np.clip(opacity, 0.0, 1.0)),
        "edge_thickness": 1.0,
        "show_edges": bool(show_edges),
        "show_vertices": bool(show_vertices),
        "texture": texture,
        "active_cmap": active_cmap,
    }


def _prepare_mesh_layer(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    vertex_color: np.ndarray | torch.Tensor | str,
    cmap: Any,
    label: str,
    visible: bool,
    opacity: float,
    show_edges: bool,
    show_vertices: bool,
    texture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verts, raw_vertex_shape, tri, valid_vertices = _prepare_mesh_inputs(vertices, faces)
    colors, raw_colors, scalar_values, scalar_meta = _prepare_vertex_colors(
        vertex_color,
        raw_vertex_shape=raw_vertex_shape,
        vertex_count=verts.shape[0],
        cmap=cmap,
    )

    valid_mask = valid_vertices & np.isfinite(colors).all(axis=1)
    texture_payload = _encode_texture_payload(texture, valid_mask)
    verts, tri = _filter_mesh_geometry(verts, tri, valid_mask)

    if np.any(valid_mask):
        colors = colors[valid_mask]
        raw_colors = raw_colors[valid_mask]
        if scalar_values is not None:
            scalar_values = scalar_values[valid_mask]
    else:
        colors = np.zeros((0, 3), dtype=np.float32)
        raw_colors = np.zeros((0, 3), dtype=np.float32)
        scalar_values = np.zeros((0,), dtype=np.float32) if scalar_values is not None else None

    return _encode_mesh_layer(
        label=label,
        verts=verts,
        tri=tri,
        colors=colors,
        raw_colors=raw_colors,
        scalar_meta=scalar_meta,
        scalar_values=scalar_values,
        visible=visible,
        opacity=opacity,
        show_edges=show_edges,
        show_vertices=show_vertices,
        texture=texture_payload,
        active_cmap=_resolve_active_cmap(cmap, texture_payload),
    )


class MeshViewerFigure(ViewerFigureBase):
    default_title = "Mesh Viewer"
    html_meta_names = ("pointcloud-viewer", "mesh-viewer")

    def __init__(
        self,
        *,
        title: str | None = None,
        opacity: float = 1.0,
        share_colormap_scale: bool = False,
    ) -> None:
        super().__init__(title=title, point_size=10.0, opacity=float(np.clip(opacity, 0.0, 1.0)), share_colormap_scale=share_colormap_scale)

    def add_pointcloud(
        self,
        points: np.ndarray | torch.Tensor,
        edges: np.ndarray | torch.Tensor | None = None,
        point_color: np.ndarray | torch.Tensor | str = "blue",
        downsampling_factor: int = 1,
        cmap: str = "uniform",
        title: str | None = None,
        label: str = "Points",
        opacity: float | None = None,
    ) -> "MeshViewerFigure":
        try:
            from .pointcloud_viewer import _prepare_layer as _prepare_pointcloud_layer
        except ImportError:  # pragma: no cover - allow direct module imports from the repo root
            from pointcloud_viewer import _prepare_layer as _prepare_pointcloud_layer

        layer = _prepare_pointcloud_layer(
            points=points,
            edges=edges,
            point_color=point_color,
            downsampling_factor=downsampling_factor,
            cmap=cmap,
            label=label,
            opacity=float(np.clip(self.opacity if opacity is None else opacity, 0.0, 1.0)),
        )
        layer["layer_kind"] = "pointcloud"
        layer["point_size"] = float(np.clip(self.point_size, 1.0, 12.0))
        self.layers.append(layer)
        if title is not None:
            self.title = title
        return self

    def add_mesh(
        self,
        vertices: np.ndarray | torch.Tensor,
        faces: np.ndarray | torch.Tensor,
        vertex_color: np.ndarray | torch.Tensor | str = "#f97316",
        cmap: Any = _AUTO_CMAP,
        title: str | None = None,
        label: str = "Mesh",
        visible: bool = True,
        opacity: float | None = None,
        show_edges: bool = False,
        show_vertices: bool = False,
        texture: dict[str, Any] | None = None,
    ) -> "MeshViewerFigure":
        self.layers.append(
            _prepare_mesh_layer(
                vertices=vertices,
                faces=faces,
                vertex_color=vertex_color,
                cmap=cmap,
                label=label,
                visible=visible,
                opacity=float(np.clip(self.opacity if opacity is None else opacity, 0.0, 1.0)),
                show_edges=show_edges,
                show_vertices=show_vertices,
                texture=texture,
            )
        )
        if title is not None:
            self.title = title
        return self


def plot_mesh(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    init_fig: ViewerFigureBase | None = None,
    vertex_color: np.ndarray | torch.Tensor | str = "#f97316",
    scalar_fields: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    opacity: float | None = None,
    share_colormap_scale: bool | None = None,
    cmap: Any = _AUTO_CMAP,
    title: str | None = None,
    label: str = "Mesh",
    visible: bool = True,
    show_edges: bool = False,
    show_vertices: bool = False,
    texture: dict[str, Any] | None = None,
) -> MeshViewerFigure:
    if init_fig is None:
        fig = MeshViewerFigure()
    elif isinstance(init_fig, MeshViewerFigure):
        fig = init_fig
    else:
        fig = MeshViewerFigure(
            title=init_fig.title,
            opacity=float(np.clip(init_fig.opacity, 0.0, 1.0)),
            share_colormap_scale=bool(init_fig.share_colormap_scale),
        )
        fig.point_size = float(np.clip(getattr(init_fig, "point_size", fig.point_size), 1.0, 12.0))
        fig.layers.extend(init_fig.layers)
    if share_colormap_scale is not None:
        fig.share_colormap_scale = bool(share_colormap_scale)
    resolved_opacity = float(np.clip(fig.opacity if opacity is None else opacity, 0.0, 1.0))

    if scalar_fields is not None:
        if len(scalar_fields) == 0:
            raise ValueError("scalar_fields cannot be empty")

        verts, raw_vertex_shape, tri, valid_vertices = _prepare_mesh_inputs(vertices, faces)
        prepared_fields: list[tuple[str, np.ndarray, np.ndarray | None, dict[str, Any] | None]] = []
        base_field_name: str | None = None
        base_valid_mask: np.ndarray | None = None

        for field_name, field_values in scalar_fields.items():
            colors, raw_colors, scalar_values, scalar_meta = _prepare_vertex_colors(
                field_values,
                raw_vertex_shape=raw_vertex_shape,
                vertex_count=verts.shape[0],
                cmap=cmap,
            )
            valid_mask = valid_vertices & np.isfinite(colors).all(axis=1)

            if base_valid_mask is None:
                base_field_name = field_name
                base_valid_mask = valid_mask.copy()
            elif not np.array_equal(valid_mask, base_valid_mask):
                raise ValueError(
                    f"Field '{field_name}' has a different valid-vertex set than '{base_field_name}'. "
                    "Ensure all scalar fields are finite on the same vertices."
                )

            prepared_fields.append((field_name, colors, raw_colors, scalar_values, scalar_meta))

        if base_valid_mask is None or base_field_name is None:
            raise ValueError("scalar_fields cannot be empty")

        has_valid_vertices = bool(np.any(base_valid_mask))
        filtered_verts, filtered_faces = _filter_mesh_geometry(verts, tri, base_valid_mask)
        texture_payload = _encode_texture_payload(texture, base_valid_mask)
        base_colors, base_raw_colors, _, _ = _prepare_vertex_colors(
            vertex_color,
            raw_vertex_shape=raw_vertex_shape,
            vertex_count=verts.shape[0],
            cmap="uniform",
        )
        filtered_base_colors = base_colors[base_valid_mask] if has_valid_vertices else np.zeros((0, 3), dtype=np.float32)
        filtered_base_raw_colors = base_raw_colors[base_valid_mask] if has_valid_vertices else np.zeros((0, 3), dtype=np.float32)
        encoded_fields: list[tuple[str, dict[str, Any]]] = []

        for field_name, colors, raw_colors, scalar_values, scalar_meta in prepared_fields:
            filtered_colors = colors[base_valid_mask] if has_valid_vertices else np.zeros((0, 3), dtype=np.float32)
            filtered_raw_colors = raw_colors[base_valid_mask] if has_valid_vertices else np.zeros((0, 3), dtype=np.float32)
            if scalar_values is None:
                filtered_values = None
            elif has_valid_vertices:
                filtered_values = scalar_values[base_valid_mask]
            else:
                filtered_values = np.zeros((0,), dtype=np.float32)

            encoded_fields.append(
                (
                    field_name,
                    _encode_mesh_layer(
                        label=label,
                        verts=filtered_verts,
                        tri=filtered_faces,
                        colors=filtered_colors,
                        raw_colors=filtered_base_raw_colors,
                        scalar_meta=scalar_meta,
                        scalar_values=filtered_values,
                        visible=visible,
                        opacity=resolved_opacity,
                        show_edges=show_edges,
                        show_vertices=show_vertices,
                        active_cmap=_resolve_active_cmap(cmap, None),
                    ),
                )
            )

        _, base_layer = encoded_fields[0]

        combined_layer = {
            "layer_kind": "mesh",
            "label": label,
            "vertices": base_layer["vertices"],
            "faces": base_layer["faces"],
            "colors": base_layer["colors"],
            "raw_colors": _encode_array_payload(_to_uint8_rgb(filtered_base_raw_colors), dtype=np.uint8),
            "vertex_count": base_layer["vertex_count"],
            "face_count": base_layer["face_count"],
            "values": base_layer["values"],
            "field_names": [name for name, _ in encoded_fields],
            "field_colors": [layer_data["raw_colors"] for _, layer_data in encoded_fields],
            "field_values": [layer_data["values"] for _, layer_data in encoded_fields],
            "field_scalar_meta": [layer_data["scalar_meta"] for _, layer_data in encoded_fields],
            "active_field_idx": 0,
            "scalar_meta": base_layer["scalar_meta"],
            "visible": bool(visible),
            "opacity": resolved_opacity,
            "edge_thickness": 1.0,
            "show_edges": bool(show_edges),
            "show_vertices": bool(show_vertices),
            "texture": texture_payload,
            "active_cmap": _resolve_active_cmap(cmap, texture_payload),
        }
        if _uses_raw_colors(cmap):
            combined_layer["colors"] = _encode_array_payload(_to_uint8_rgb(filtered_base_colors), dtype=np.uint8)
        fig.layers.append(combined_layer)
        if title is not None:
            fig.title = title
        return fig

    return fig.add_mesh(
        vertices=vertices,
        faces=faces,
        vertex_color=vertex_color,
        cmap=cmap,
        title=title,
        label=label,
        visible=visible,
        opacity=resolved_opacity,
        show_edges=show_edges,
        show_vertices=show_vertices,
        texture=texture,
    )


def plot_trimesh(
    trimesh,
    **kwargs,
) -> MeshViewerFigure:
    if isinstance(trimesh, (list, tuple)):
        if len(trimesh) == 0:
            raise ValueError("trimesh list cannot be empty")

        init_fig = kwargs.pop("init_fig", None)
        label = kwargs.pop("label", None)
        fig = init_fig

        for idx, mesh in enumerate(trimesh):
            mesh_kwargs = dict(kwargs)
            mesh_kwargs["init_fig"] = fig
            if isinstance(label, (list, tuple)):
                mesh_kwargs["label"] = label[idx]
            elif label is not None:
                mesh_kwargs["label"] = label
            else:
                mesh_kwargs["label"] = f"Mesh {idx + 1}"
            fig = plot_trimesh(mesh, **mesh_kwargs)

        return fig

    if "texture" not in kwargs:
        texture = _extract_trimesh_texture(trimesh)
        if texture is not None:
            kwargs["texture"] = texture
    return plot_mesh(
        vertices=np.asarray(trimesh.vertices),
        faces=np.asarray(trimesh.faces),
        **kwargs
    )
