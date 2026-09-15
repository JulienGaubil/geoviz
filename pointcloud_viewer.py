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
        _encode_array_payload,
        _prepare_color_data,
        _remap_filtered_indices,
        _to_numpy,
        _to_uint8_rgb,
    )
except ImportError:  # pragma: no cover - allow direct module imports from the repo root
    from viewer_common import (
        ViewerFigureBase,
        _encode_array_payload,
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
    cmap: str,
    label: str,
    opacity: float,
    visible: bool = True,
    edge_color: str = "#000000",
    edge_scalar_fields: Mapping[str, np.ndarray | torch.Tensor] | None = None,
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
    original_edges = edg

    if downsampling_factor > 1 and edg is not None:
        raise ValueError(
            "downsampling_factor > 1 is not supported when edges are provided"
        )

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

    # An explicit RGB color with the uniform colormap is already the final
    # display value. Do not manufacture/transmit its first channel as an
    # unused scalar field. Scalar inputs retain their values and metadata.
    explicit_rgb = isinstance(point_color, str)
    if not explicit_rgb:
        raw_point_color = _to_numpy(point_color)
        explicit_rgb = (
            raw_point_color is not None
            and tuple(np.asarray(raw_point_color).shape) == raw_point_shape
        )
    if explicit_rgb and cmap == "uniform":
        scalar_values = None
        scalar_meta = None

    # Keep only finite points/colors; invalid values can make Three.js compute NaN bounds
    # and cull the whole point cloud.
    finite_pts = np.isfinite(pts).all(axis=1)
    finite_cols = np.isfinite(colors).all(axis=1)
    valid_mask = finite_pts & finite_cols

    if np.all(valid_mask):
        pass
    elif np.any(valid_mask):
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
        scalar_values = (
            np.zeros((0,), dtype=np.float32) if scalar_values is not None else None
        )

    edge_keep_mask: np.ndarray | None = None
    if edg is not None:
        edge_keep_mask = (
            (edg >= 0).all(axis=1)
            & (edg < valid_mask.shape[0]).all(axis=1)
        )
        if np.any(edge_keep_mask):
            in_range_edges = edg[edge_keep_mask]
            edge_keep_mask[edge_keep_mask] &= valid_mask[in_range_edges].all(axis=1)
        edg = _remap_filtered_indices(edg, valid_mask)

    encoded_edge_fields: list[
        tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    if edge_scalar_fields is not None:
        if original_edges is None:
            raise ValueError("edge_scalar_fields requires edges")
        if len(edge_scalar_fields) == 0:
            raise ValueError("edge_scalar_fields cannot be empty")
        edge_count = original_edges.shape[0]
        for field_name, field_values in edge_scalar_fields.items():
            values = _to_numpy(field_values)
            if values is None:
                raise ValueError(f"Edge field '{field_name}' cannot be None")
            values = np.asarray(values)
            if values.shape in {(edge_count,), (edge_count, 1)}:
                values = values.reshape(edge_count)
                raw_geometry_shape = (edge_count, 1)
            elif values.shape == (edge_count, 3):
                raw_geometry_shape = values.shape
            else:
                raise ValueError(
                    f"Edge field '{field_name}' has shape {values.shape}; "
                    f"expected one scalar or RGB value for each of "
                    f"{edge_count} edges"
                )
            field_colors, _, edge_scalar_values, edge_scalar_meta = _prepare_color_data(
                values,
                raw_geometry_shape=raw_geometry_shape,
                item_count=edge_count,
                cmap=cmap,
                color_label=f"edge_scalar_fields['{field_name}']",
                downsampling_factor=1,
            )
            if edge_scalar_values is None or edge_scalar_meta is None:
                raise ValueError(
                    f"Edge field '{field_name}' must be scalar or RGB"
                )
            if edge_keep_mask is not None:
                field_colors = field_colors[edge_keep_mask]
                edge_scalar_values = edge_scalar_values[edge_keep_mask]
            encoded_edge_fields.append(
                (
                    str(field_name),
                    _encode_array_payload(_to_uint8_rgb(field_colors), dtype=np.uint8),
                    _encode_array_payload(edge_scalar_values, dtype=np.float32),
                    edge_scalar_meta,
                )
            )

    colors_uint8 = _to_uint8_rgb(colors)
    raw_colors_uint8 = (
        colors_uint8
        if raw_colors is colors or np.shares_memory(raw_colors, colors)
        else _to_uint8_rgb(raw_colors)
    )
    raw_colors_payload = (
        None
        if explicit_rgb and np.array_equal(colors_uint8, raw_colors_uint8)
        else _encode_array_payload(raw_colors_uint8, dtype=np.uint8)
    )

    layer = {
        "layer_kind": "pointcloud",
        "label": label,
        "points": _encode_array_payload(pts, dtype=np.float32),
        "colors": _encode_array_payload(colors_uint8, dtype=np.uint8),
        "raw_colors": raw_colors_payload,
        "edges": (
            None if edg is None else _encode_array_payload(edg, dtype=np.uint32)
        ),
        "edge_thickness": 1.0,
        "edge_color": str(edge_color),
        "edge_field_names": [field[0] for field in encoded_edge_fields],
        "edge_field_colors": [field[1] for field in encoded_edge_fields],
        "edge_field_values": [field[2] for field in encoded_edge_fields],
        "edge_field_scalar_meta": [field[3] for field in encoded_edge_fields],
        "active_edge_field_idx": 0,
        "scalar_meta": scalar_meta,
        "values": (
            None
            if scalar_values is None
            else _encode_array_payload(scalar_values, dtype=np.float32)
        ),
        "visible": bool(visible),
        "opacity": float(np.clip(opacity, 0.0, 1.0)),
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
        cmap: str = "uniform",
        title: str | None = None,
        label: str = "Points",
        visible: bool = True,
        opacity: float | None = None,
        edge_color: str = "#000000",
        edge_scalar_fields: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    ) -> "PointCloudViewerFigure":
        layer = _prepare_layer(
            points=points,
            edges=edges,
            point_color=point_color,
            downsampling_factor=downsampling_factor,
            cmap=cmap,
            label=label,
            opacity=float(
                np.clip(self.opacity if opacity is None else opacity, 0.0, 1.0)
            ),
            visible=visible,
            edge_color=edge_color,
            edge_scalar_fields=edge_scalar_fields,
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
    edge_color: str = "#000000",
    share_colormap_scale: bool | None = None,
    cmap: str = "uniform",
    title: str | None = None,
    label: str = "Points",
    visible: bool = True,
    edge_scalar_fields: Mapping[str, np.ndarray | torch.Tensor] | None = None,
) -> ViewerFigureBase:
    fig = init_fig if init_fig is not None else PointCloudViewerFigure()
    if not hasattr(fig, "add_pointcloud"):
        raise TypeError("init_fig must provide an add_pointcloud(...) method")
    if point_size is not None:
        fig.point_size = float(point_size)
    if share_colormap_scale is not None:
        fig.share_colormap_scale = bool(share_colormap_scale)
    resolved_opacity = float(
        np.clip(fig.opacity if opacity is None else opacity, 0.0, 1.0)
    )
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
                        visible=visible,
                        edge_color=edge_color,
                        edge_scalar_fields=edge_scalar_fields,
                    ),
                )
            )

        base_name, base_layer = prepared_fields[0]
        base_points = base_layer["points"]
        base_edges = base_layer["edges"]

        for field_name, layer_data in prepared_fields[1:]:
            if not np.array_equal(
                layer_data["points"]["array"], base_points["array"]
            ):
                raise ValueError(
                    f"Field '{field_name}' has a different valid-point set than '{base_name}'. "
                    "Ensure all scalar fields are finite on the same points."
                )
            layer_edges = layer_data["edges"]
            edges_match = (layer_edges is None and base_edges is None) or (
                layer_edges is not None
                and base_edges is not None
                and np.array_equal(layer_edges["array"], base_edges["array"])
            )
            if not edges_match:
                raise ValueError(
                    f"Field '{field_name}' has incompatible edges after filtering with '{base_name}'."
                )

        combined_layer = {
            "layer_kind": "pointcloud",
            "label": label,
            "points": base_layer["points"],
            "edges": base_layer["edges"],
            "edge_thickness": 1.0,
            "edge_color": str(edge_color),
            "edge_field_names": base_layer["edge_field_names"],
            "edge_field_colors": base_layer["edge_field_colors"],
            "edge_field_values": base_layer["edge_field_values"],
            "edge_field_scalar_meta": base_layer["edge_field_scalar_meta"],
            "active_edge_field_idx": 0,
            "colors": base_layer["colors"],
            "raw_colors": base_layer["raw_colors"],
            "values": base_layer["values"],
            "field_names": [name for name, _ in prepared_fields],
            "field_colors": [
                (
                    layer_data["colors"]
                    if layer_data["raw_colors"] is None
                    else layer_data["raw_colors"]
                )
                for _, layer_data in prepared_fields
            ],
            "field_values": [layer_data["values"] for _, layer_data in prepared_fields],
            "field_scalar_meta": [
                layer_data["scalar_meta"] for _, layer_data in prepared_fields
            ],
            "active_field_idx": 0,
            "scalar_meta": base_layer["scalar_meta"],
            "point_size": float(np.clip(fig.point_size, 1.0, 12.0)),
            "visible": bool(visible),
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
        visible=visible,
        edge_color=edge_color,
        edge_scalar_fields=edge_scalar_fields,
        opacity=resolved_opacity,
    )
