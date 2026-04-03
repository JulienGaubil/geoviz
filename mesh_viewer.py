from __future__ import annotations

import base64
from typing import Any, Mapping

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - optional dependency in some environments
    torch = None

try:
    from .viewer_common import (
        ViewerFigureBase,
        _raw_scalar_gradient,
        _sample_cmap,
        _to_numpy,
    )
except ImportError:  # pragma: no cover - allow direct module imports from the repo root
    from viewer_common import (
        ViewerFigureBase,
        _raw_scalar_gradient,
        _sample_cmap,
        _to_numpy,
    )


def _encode_array_payload(arr: np.ndarray, *, dtype: np.dtype | str | None = None) -> dict[str, Any]:
    encoded = np.ascontiguousarray(arr, dtype=dtype)
    return {
        "encoding": "base64",
        "dtype": str(encoded.dtype),
        "shape": list(encoded.shape),
        "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


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
    cmap: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any] | None]:
    scalar_meta = None
    scalar_values: np.ndarray | None = None

    if isinstance(vertex_color, str):
        rgb = np.asarray(mcolors.to_rgb(vertex_color), dtype=np.float32)
        colors = np.repeat(rgb[None, :], vertex_count, axis=0)
        raw_colors = colors
    else:
        color_arr = _to_numpy(vertex_color)
        if color_arr is None:
            raise ValueError("vertex_color cannot be None")
        color_arr = np.asarray(color_arr)
        base_shape = raw_vertex_shape[:-1]

        if list(color_arr.shape) != list(raw_vertex_shape):
            if color_arr.shape == tuple(base_shape):
                scalar_values = np.asarray(color_arr, dtype=np.float32).reshape(-1)
                color_arr = np.repeat(color_arr[..., None], 3, axis=-1)
            elif color_arr.shape == tuple(base_shape) + (1,):
                scalar_values = np.asarray(color_arr, dtype=np.float32).reshape(-1)
                color_arr = np.repeat(color_arr, 3, axis=-1)
            else:
                raise ValueError(f"Unrecognized vertex_color shape {color_arr.shape}")

        color_arr = np.asarray(color_arr, dtype=np.float32).reshape(-1, 3)
        if scalar_values is None:
            scalar_values = color_arr[:, 0].astype(np.float32, copy=False)
        raw_colors = color_arr

        scalar_vals = scalar_values.astype(np.float32, copy=False)
        finite_scalar = np.isfinite(scalar_vals)
        if np.any(finite_scalar):
            finite_vals = scalar_vals[finite_scalar]
            min_val = float(np.min(finite_vals))
            max_val = float(np.max(finite_vals))
        else:
            min_val, max_val = 0.0, 1.0
        scalar_meta = {
            "min": min_val,
            "max": max_val,
            "cmap": cmap,
            "gradient": _sample_cmap(cmap) if cmap is not None else _raw_scalar_gradient(min_val, max_val),
        }

        if cmap is not None:
            norm = (scalar_vals - min_val) / max(max_val - min_val, 1e-8)
            norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
            colors = plt.get_cmap(cmap)(norm)[:, :3].astype(np.float32)
        else:
            colors = color_arr

    return colors, raw_colors, scalar_values, scalar_meta


def _filter_mesh_geometry(
    verts: np.ndarray,
    tri: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if np.any(valid_mask):
        verts = verts[valid_mask]
    else:
        verts = np.zeros((0, 3), dtype=np.float32)

    n_raw = int(valid_mask.shape[0])
    in_range = (
        (tri[:, 0] >= 0)
        & (tri[:, 1] >= 0)
        & (tri[:, 2] >= 0)
        & (tri[:, 0] < n_raw)
        & (tri[:, 1] < n_raw)
        & (tri[:, 2] < n_raw)
    )
    tri = tri[in_range]

    if np.any(valid_mask) and tri.shape[0] > 0:
        remap = np.full((n_raw,), -1, dtype=np.int32)
        remap[np.where(valid_mask)[0]] = np.arange(int(np.sum(valid_mask)), dtype=np.int32)
        keep = valid_mask[tri[:, 0]] & valid_mask[tri[:, 1]] & valid_mask[tri[:, 2]]
        tri = tri[keep]
        if tri.shape[0] > 0:
            tri = np.stack([remap[tri[:, 0]], remap[tri[:, 1]], remap[tri[:, 2]]], axis=1)
    else:
        tri = np.zeros((0, 3), dtype=np.int32)
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
) -> dict[str, Any]:
    if scalar_values is not None and scalar_values.shape[0] != verts.shape[0]:
        raise ValueError("scalar_values must match filtered vertex count")

    def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return np.zeros((0, 3), dtype=np.uint8)
        max_color = float(np.nanmax(arr))
        if max_color <= 1.0 + 1e-6:
            return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)

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
        "opacity": float(np.clip(opacity, 0.05, 1.0)),
        "edge_thickness": 1.0,
        "show_edges": bool(show_edges),
    }


def _prepare_mesh_layer(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    vertex_color: np.ndarray | torch.Tensor | str,
    cmap: str | None,
    label: str,
    visible: bool,
    opacity: float,
    show_edges: bool,
) -> dict[str, Any]:
    verts, raw_vertex_shape, tri, valid_vertices = _prepare_mesh_inputs(vertices, faces)
    colors, raw_colors, scalar_values, scalar_meta = _prepare_vertex_colors(
        vertex_color,
        raw_vertex_shape=raw_vertex_shape,
        vertex_count=verts.shape[0],
        cmap=cmap,
    )

    valid_mask = valid_vertices & np.isfinite(colors).all(axis=1)
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
        super().__init__(title=title, point_size=5.0, opacity=float(np.clip(opacity, 0.05, 1.0)), share_colormap_scale=share_colormap_scale)

    def add_pointcloud(
        self,
        points: np.ndarray | torch.Tensor,
        edges: np.ndarray | torch.Tensor | None = None,
        point_color: np.ndarray | torch.Tensor | str = "blue",
        downsampling_factor: int = 1,
        cmap: str | None = None,
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
            opacity=float(np.clip(self.opacity if opacity is None else opacity, 0.05, 1.0)),
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
        vertex_color: np.ndarray | torch.Tensor | str = "#94a3b8",
        cmap: str | None = None,
        title: str | None = None,
        label: str = "Mesh",
        visible: bool = True,
        opacity: float | None = None,
        show_edges: bool = False,
    ) -> "MeshViewerFigure":
        self.layers.append(
            _prepare_mesh_layer(
                vertices=vertices,
                faces=faces,
                vertex_color=vertex_color,
                cmap=cmap,
                label=label,
                visible=visible,
                opacity=float(np.clip(self.opacity if opacity is None else opacity, 0.05, 1.0)),
                show_edges=show_edges,
            )
        )
        if title is not None:
            self.title = title
        return self


def plot_mesh(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    init_fig: ViewerFigureBase | None = None,
    vertex_color: np.ndarray | torch.Tensor | str = "#94a3b8",
    scalar_fields: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    opacity: float | None = None,
    share_colormap_scale: bool | None = None,
    cmap: str | None = None,
    title: str | None = None,
    label: str = "Mesh",
    visible: bool = True,
    show_edges: bool = False,
) -> MeshViewerFigure:
    if init_fig is None:
        fig = MeshViewerFigure()
    elif isinstance(init_fig, MeshViewerFigure):
        fig = init_fig
    else:
        fig = MeshViewerFigure(
            title=init_fig.title,
            opacity=float(np.clip(init_fig.opacity, 0.05, 1.0)),
            share_colormap_scale=bool(init_fig.share_colormap_scale),
        )
        fig.point_size = float(np.clip(getattr(init_fig, "point_size", fig.point_size), 1.0, 12.0))
        fig.layers.extend(init_fig.layers)
    if share_colormap_scale is not None:
        fig.share_colormap_scale = bool(share_colormap_scale)
    resolved_opacity = float(np.clip(fig.opacity if opacity is None else opacity, 0.05, 1.0))

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

        filtered_verts, filtered_faces = _filter_mesh_geometry(verts, tri, base_valid_mask)
        encoded_fields: list[tuple[str, dict[str, Any]]] = []
        has_valid_vertices = bool(np.any(base_valid_mask))

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
                        raw_colors=filtered_raw_colors,
                        scalar_meta=scalar_meta,
                        scalar_values=filtered_values,
                        visible=visible,
                        opacity=resolved_opacity,
                        show_edges=show_edges,
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
            "raw_colors": base_layer["raw_colors"],
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
        }
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
    )
