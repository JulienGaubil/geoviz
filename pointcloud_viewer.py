from __future__ import annotations

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


def _prepare_layer(
    points: np.ndarray | torch.Tensor,
    edges: np.ndarray | torch.Tensor | None,
    point_color: np.ndarray | torch.Tensor | str,
    downsampling_factor: int,
    cmap: str | None,
    label: str,
    opacity: float,
) -> dict[str, Any]:
    raw_points = _to_numpy(points)
    pts = raw_points
    if pts is None:
        raise ValueError("points cannot be None")
    raw_point_shape = tuple(np.asarray(pts).shape)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)

    edg = _to_numpy(edges)
    if edg is not None:
        edg = np.asarray(edg, dtype=np.int32).reshape(-1, 2)

    if downsampling_factor > 1 and edg is not None:
        raise ValueError("downsampling_factor > 1 is not supported when edges are provided")

    if downsampling_factor > 1:
        pts = pts[::downsampling_factor]

    colors, raw_colors, scalar_values, scalar_meta = _prepare_color_data(
        point_color,
        raw_geometry_shape=raw_point_shape,
        item_count=pts.shape[0],
        cmap=cmap,
        color_label="point_color",
        downsampling_factor=downsampling_factor,
    )

    # Keep only finite points/colors; invalid values can make Three.js compute NaN bounds
    # and cull the whole point cloud.
    finite_pts = np.isfinite(pts).all(axis=1)
    finite_cols = np.isfinite(colors).all(axis=1)
    valid_mask = finite_pts & finite_cols

    if np.any(valid_mask):
        pts = pts[valid_mask]
        colors = colors[valid_mask]
        if raw_colors is not None:
            raw_colors = raw_colors[valid_mask]
        if scalar_values is not None:
            scalar_values = scalar_values[valid_mask]
    else:
        pts = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.float32)
        raw_colors = np.zeros((0, 3), dtype=np.float32)
        scalar_values = np.zeros((0,), dtype=np.float32) if scalar_values is not None else None

    if edg is not None:
        edg = _remap_filtered_indices(edg, valid_mask)

    colors_uint8 = _to_uint8_rgb(colors)
    raw_colors_uint8 = _to_uint8_rgb(raw_colors)

    layer = {
        "layer_kind": "pointcloud",
        "label": label,
        "points": np.ascontiguousarray(pts, dtype=np.float32).tolist(),
        "colors": colors_uint8.tolist(),
        "raw_colors": raw_colors_uint8.tolist(),
        "edges": None if edg is None else edg.tolist(),
        "edge_thickness": 1.0,
        "scalar_meta": scalar_meta,
        "values": None if scalar_values is None else np.ascontiguousarray(scalar_values, dtype=np.float32).tolist(),
        "visible": True,
        "opacity": float(np.clip(opacity, 0.05, 1.0)),
    }
    return layer


class PointCloudViewerFigure(ViewerFigureBase):
    default_title = "Point Cloud Viewer"
    html_meta_names = ("pointcloud-viewer",)

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
    ) -> "PointCloudViewerFigure":
        layer = _prepare_layer(
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


def plot_pointcloud(
    points: np.ndarray | torch.Tensor,
    edges: np.ndarray | torch.Tensor | None = None,
    init_fig: ViewerFigureBase | None = None,
    point_color: np.ndarray | torch.Tensor | str = "blue",
    scalar_fields: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    downsampling_factor: int = 1,
    point_size: float | None = None,
    opacity: float | None = None,
    share_colormap_scale: bool | None = None,
    cmap: str | None = None,
    title: str | None = None,
    label: str = "Points",
) -> ViewerFigureBase:
    fig = init_fig if init_fig is not None else PointCloudViewerFigure()
    if not hasattr(fig, "add_pointcloud"):
        raise TypeError("init_fig must provide an add_pointcloud(...) method")
    if point_size is not None:
        fig.point_size = float(point_size)
    if share_colormap_scale is not None:
        fig.share_colormap_scale = bool(share_colormap_scale)
    resolved_opacity = float(np.clip(fig.opacity if opacity is None else opacity, 0.05, 1.0))
    if scalar_fields is not None:
        if len(scalar_fields) == 0:
            raise ValueError("scalar_fields cannot be empty")

        prepared_fields: list[tuple[str, dict[str, Any]]] = []
        for field_name, field_values in scalar_fields.items():
            prepared_fields.append(
                (
                    field_name,
                    _prepare_layer(
                        points=points,
                        edges=edges,
                        point_color=field_values,
                        downsampling_factor=downsampling_factor,
                        cmap=cmap,
                        label=label,
                        opacity=resolved_opacity,
                    ),
                )
            )

        base_name, base_layer = prepared_fields[0]
        base_points = base_layer["points"]
        base_edges = base_layer["edges"]

        for field_name, layer_data in prepared_fields[1:]:
            if layer_data["points"] != base_points:
                raise ValueError(
                    f"Field '{field_name}' has a different valid-point set than '{base_name}'. "
                    "Ensure all scalar fields are finite on the same points."
                )
            if layer_data["edges"] != base_edges:
                raise ValueError(
                    f"Field '{field_name}' has incompatible edges after filtering with '{base_name}'."
                )

        combined_layer = {
            "layer_kind": "pointcloud",
            "label": label,
            "points": base_layer["points"],
            "edges": base_layer["edges"],
            "edge_thickness": 1.0,
            "colors": base_layer["colors"],
            "raw_colors": base_layer["raw_colors"],
            "values": base_layer["values"],
            "field_names": [name for name, _ in prepared_fields],
            "field_colors": [layer_data["raw_colors"] for _, layer_data in prepared_fields],
            "field_values": [layer_data["values"] for _, layer_data in prepared_fields],
            "field_scalar_meta": [layer_data["scalar_meta"] for _, layer_data in prepared_fields],
            "active_field_idx": 0,
            "scalar_meta": base_layer["scalar_meta"],
            "point_size": float(np.clip(fig.point_size, 1.0, 12.0)),
            "visible": True,
            "opacity": resolved_opacity,
        }
        fig.layers.append(combined_layer)
        if title is not None:
            fig.title = title
        return fig

    return fig.add_pointcloud(
        points=points,
        edges=edges,
        point_color=point_color,
        downsampling_factor=downsampling_factor,
        cmap=cmap,
        title=title,
        label=label,
        opacity=resolved_opacity,
    )
