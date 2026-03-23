from __future__ import annotations

import base64
import json
from html import escape
from typing import Any, Mapping

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - optional dependency in some environments
    torch = None

from pointcloud_viewer import PointCloudViewerFigure


def _to_numpy(x: np.ndarray | torch.Tensor | None) -> np.ndarray | None:
    if x is None:
        return None
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _sample_cmap(cmap: str, n: int = 7) -> list[str]:
    colormap = plt.get_cmap(cmap)
    values = np.linspace(0.0, 1.0, n)
    return [mcolors.to_hex(colormap(v), keep_alpha=False) for v in values]


def _sample_cmap_rgb(cmap: str, n: int = 64) -> list[list[int]]:
    colormap = plt.get_cmap(cmap)
    values = np.linspace(0.0, 1.0, n)
    rgb = np.asarray(colormap(values)[:, :3], dtype=np.float32)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8).tolist()


def _available_cmaps(preferred: str | None = None) -> list[str]:
    options: list[str] = []
    if preferred is not None:
        options.append(preferred)
    options.extend(
        [
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "jet",
            "coolwarm",
            "Spectral",
            "RdYlBu",
        ]
    )

    valid: list[str] = []
    seen: set[str] = set()
    for cmap_name in options:
        if cmap_name in seen:
            continue
        try:
            plt.get_cmap(cmap_name)
        except ValueError:
            continue
        valid.append(cmap_name)
        seen.add(cmap_name)
    return valid


def _collect_layer_cmaps(layers: list[dict[str, Any]]) -> list[str]:
    preferred: list[str] = []
    for layer in layers:
        scalar_meta = layer.get("scalar_meta")
        if isinstance(scalar_meta, Mapping):
            cmap_name = scalar_meta.get("cmap")
            if isinstance(cmap_name, str):
                preferred.append(cmap_name)
        field_metas = layer.get("field_scalar_meta")
        if isinstance(field_metas, list):
            for meta in field_metas:
                if not isinstance(meta, Mapping):
                    continue
                cmap_name = meta.get("cmap")
                if isinstance(cmap_name, str):
                    preferred.append(cmap_name)

    ordered: list[str] = []
    seen: set[str] = set()
    for cmap_name in preferred:
        if cmap_name in seen:
            continue
        ordered.append(cmap_name)
        seen.add(cmap_name)
    return ordered


def _colormap_catalog(cmap_names: list[str]) -> dict[str, dict[str, Any]]:
    return {
        cmap_name: {
            "gradient": _sample_cmap(cmap_name),
            "samples": _sample_cmap_rgb(cmap_name),
        }
        for cmap_name in cmap_names
    }


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
    show_colorbar: bool,
    cmap: str | None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any] | None]:
    scalar_meta = None
    scalar_values: np.ndarray | None = None

    if isinstance(vertex_color, str):
        rgb = np.asarray(mcolors.to_rgb(vertex_color), dtype=np.float32)
        colors = np.repeat(rgb[None, :], vertex_count, axis=0)
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

        if cmap is not None:
            scalar_vals = color_arr[:, 0]
            scalar_values = scalar_vals.astype(np.float32, copy=False)
            finite_scalar = np.isfinite(scalar_vals)
            if np.any(finite_scalar):
                finite_vals = scalar_vals[finite_scalar]
                min_val = float(np.min(finite_vals))
                max_val = float(np.max(finite_vals))
            else:
                min_val, max_val = 0.0, 1.0

            norm = (scalar_vals - min_val) / max(max_val - min_val, 1e-8)
            norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
            colors = plt.get_cmap(cmap)(norm)[:, :3].astype(np.float32)
            if show_colorbar:
                scalar_meta = {
                    "min": min_val,
                    "max": max_val,
                    "cmap": cmap,
                    "gradient": _sample_cmap(cmap),
                }
        else:
            colors = color_arr

    return colors, scalar_values, scalar_meta


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
    scalar_meta: dict[str, Any] | None,
    scalar_values: np.ndarray | None,
) -> dict[str, Any]:
    if scalar_values is not None and scalar_values.shape[0] != verts.shape[0]:
        raise ValueError("scalar_values must match filtered vertex count")

    if colors.size == 0:
        colors_uint8 = np.zeros((0, 3), dtype=np.uint8)
    else:
        max_color = float(np.nanmax(colors))
        if max_color <= 1.0 + 1e-6:
            colors_uint8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        else:
            colors_uint8 = np.clip(colors, 0, 255).astype(np.uint8)

    return {
        "label": label,
        "vertices": _encode_array_payload(verts, dtype=np.float32),
        "faces": _encode_array_payload(tri, dtype=np.uint32),
        "colors": _encode_array_payload(colors_uint8, dtype=np.uint8),
        "vertex_count": int(verts.shape[0]),
        "face_count": int(tri.shape[0]),
        "scalar_meta": scalar_meta,
        "values": None
        if scalar_values is None
        else _encode_array_payload(scalar_values, dtype=np.float32),
        "visible": True,
    }


def _prepare_mesh_layer(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    vertex_color: np.ndarray | torch.Tensor | str,
    show_colorbar: bool,
    cmap: str | None,
    label: str,
) -> dict[str, Any]:
    verts, raw_vertex_shape, tri, valid_vertices = _prepare_mesh_inputs(vertices, faces)
    colors, scalar_values, scalar_meta = _prepare_vertex_colors(
        vertex_color,
        raw_vertex_shape=raw_vertex_shape,
        vertex_count=verts.shape[0],
        show_colorbar=show_colorbar,
        cmap=cmap,
    )

    valid_mask = valid_vertices & np.isfinite(colors).all(axis=1)
    verts, tri = _filter_mesh_geometry(verts, tri, valid_mask)

    if np.any(valid_mask):
        colors = colors[valid_mask]
        if scalar_values is not None:
            scalar_values = scalar_values[valid_mask]
    else:
        colors = np.zeros((0, 3), dtype=np.float32)
        scalar_values = np.zeros((0,), dtype=np.float32) if scalar_values is not None else None

    return _encode_mesh_layer(
        label=label,
        verts=verts,
        tri=tri,
        colors=colors,
        scalar_meta=scalar_meta,
        scalar_values=scalar_values,
    )


class MeshViewerFigure(PointCloudViewerFigure):
    def __init__(
        self,
        *,
        title: str | None = None,
        mesh_opacity: float = 1.0,
    ) -> None:
        super().__init__(title=title, point_size=1.0)
        self.mesh_opacity = float(np.clip(mesh_opacity, 0.05, 1.0))

    def add_mesh(
        self,
        vertices: np.ndarray | torch.Tensor,
        faces: np.ndarray | torch.Tensor,
        vertex_color: np.ndarray | torch.Tensor | str = "#94a3b8",
        show_colorbar: bool = False,
        cmap: str | None = None,
        title: str | None = None,
        label: str = "Mesh",
    ) -> "MeshViewerFigure":
        self.layers.append(
            _prepare_mesh_layer(
                vertices=vertices,
                faces=faces,
                vertex_color=vertex_color,
                show_colorbar=show_colorbar,
                cmap=cmap,
                label=label,
            )
        )
        if title is not None:
            self.title = title
        return self

    def _payload(self) -> dict[str, Any]:
        preferred_cmaps = _collect_layer_cmaps(self.layers)
        cmap_names = _available_cmaps(preferred_cmaps[0] if preferred_cmaps else None)
        for cmap_name in preferred_cmaps[1:]:
            if cmap_name not in cmap_names:
                cmap_names.insert(0, cmap_name)
        return {
            "title": self.title or "Mesh Viewer",
            "mesh_opacity": self.mesh_opacity,
            "layers": self.layers,
            "available_cmaps": cmap_names,
            "colormap_presets": _colormap_catalog(cmap_names),
        }

    def to_html(self) -> str:
        payload = json.dumps(self._payload())
        page_title = escape(self.title or "Mesh Viewer")

        html = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
  <meta name=\"pointcloud-viewer\" content=\"1\">
  <meta name=\"mesh-viewer\" content=\"1\">
  <title>__TITLE__</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600&display=swap');
    :root {
      --bg-dark: #0b1120;
      --bg-panel: rgba(18, 18, 28, 0.92);
      --bg-canvas: #111827;
      --bg-canvas-solid: #111827;
      --border-color: rgba(100, 120, 255, 0.15);
      --accent-primary: #6366f1;
      --accent-secondary: #22d3ee;
      --text-primary: #e4e4e7;
      --text-secondary: #a1a1aa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      overflow: hidden;
      font-family: 'JetBrains Mono', monospace;
      background: var(--bg-dark);
      color: var(--text-primary);
    }
    body.day-mode {
      --bg-dark: #ffffff;
      --bg-panel: rgba(255, 255, 255, 0.94);
      --bg-canvas: #ffffff;
      --bg-canvas-solid: #ffffff;
      --border-color: rgba(55, 65, 81, 0.2);
      --accent-primary: #1d4ed8;
      --accent-secondary: #0891b2;
      --text-primary: #111827;
      --text-secondary: #374151;
    }
    #canvas-container {
      position: fixed;
      inset: 0;
      background: var(--bg-canvas);
    }
    #top-actions {
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 30;
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .top-action-btn {
      width: 44px;
      height: 44px;
      border-radius: 10px;
      border: 1px solid var(--border-color);
      background: var(--bg-panel);
      color: var(--text-primary);
      font-size: 20px;
      line-height: 1;
      cursor: pointer;
      backdrop-filter: blur(18px);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .top-action-btn.text-btn {
      width: auto;
      min-width: 132px;
      padding: 10px 14px;
      font-size: 11px;
      letter-spacing: 0.2px;
      font-family: 'JetBrains Mono', monospace;
    }
    .top-action-btn.active {
      background: rgba(99, 102, 241, 0.25);
      border-color: var(--accent-primary);
      color: var(--accent-primary);
    }
    .top-action-btn.danger {
      width: auto;
      min-width: 112px;
      padding: 0 12px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
      border-color: rgba(239, 68, 68, 0.4);
      color: #ef4444;
    }
    #controls {
      position: fixed;
      top: 20px;
      left: 20px;
      z-index: 10;
      min-width: 280px;
      max-width: min(360px, calc(100vw - 40px));
      max-height: calc(100vh - 40px);
      overflow: auto;
      padding: 18px;
      border-radius: 16px;
      background: var(--bg-panel);
      border: 1px solid var(--border-color);
      backdrop-filter: blur(18px);
      box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
      transition: transform 0.25s ease, opacity 0.25s ease;
    }
    #controls.collapsed {
      transform: translateX(calc(-100% - 40px));
      opacity: 0;
      pointer-events: none;
    }
    h1 {
      margin: 0 0 8px 0;
      font-family: 'Space Grotesk', sans-serif;
      font-size: 18px;
      color: var(--accent-primary);
    }
    .section {
      margin-top: 18px;
    }
    .section-title {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-secondary);
      margin-bottom: 10px;
    }
    .layer-row {
      display: grid;
      grid-template-columns: 20px 1fr;
      gap: 8px;
      align-items: start;
      margin-bottom: 10px;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.05);
      background: rgba(255,255,255,0.02);
    }
    .layer-label {
      font-size: 12px;
      line-height: 1.35;
    }
    .layer-select-wrapper {
      position: relative;
      margin-top: 6px;
    }
    .custom-dropdown {
      position: relative;
      width: 100%;
    }
    .layer-select-button {
      width: 100%;
      padding: 8px 10px;
      font-family: 'Space Grotesk', sans-serif;
      font-size: 12px;
      font-weight: 600;
      color: var(--accent-primary);
      background: rgba(99, 102, 241, 0.1);
      border: 1px solid rgba(99, 102, 241, 0.2);
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
      appearance: none;
      -webkit-appearance: none;
      -moz-appearance: none;
      text-align: left;
      display: flex;
      align-items: center;
      gap: 8px;
      background-image: url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2322d3ee' d='M6 9L1 4h10z'/%3E%3C/svg%3E\");
      background-repeat: no-repeat;
      background-position: right 8px center;
      padding-right: 28px;
    }
    .layer-select-button .layer-id {
      flex: 1;
      text-align: center;
    }
    .layer-select-button:hover {
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--accent-primary);
    }
    .layer-select-button:focus {
      outline: none;
      background: rgba(99, 102, 241, 0.2);
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1);
    }
    .custom-dropdown-menu {
      position: absolute;
      top: 100%;
      left: 0;
      right: 0;
      margin-top: 4px;
      max-height: 320px;
      overflow-y: auto;
      background: var(--bg-panel);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(99, 102, 241, 0.2);
      border-radius: 8px;
      box-shadow:
        0 4px 20px rgba(0, 0, 0, 0.35),
        0 0 24px rgba(99, 102, 241, 0.1);
      z-index: 1000;
      display: none;
    }
    .custom-dropdown-menu.open {
      display: block;
    }
    .custom-dropdown-menu.floating {
      position: fixed;
      top: 0;
      left: 0;
      right: auto;
      width: 240px;
      margin-top: 0;
      max-height: min(420px, calc(100vh - 24px));
      z-index: 1200;
    }
    .custom-dropdown-item {
      padding: 8px 10px;
      font-family: 'Space Grotesk', sans-serif;
      font-size: 12px;
      font-weight: 600;
      color: var(--accent-primary);
      background: rgba(99, 102, 241, 0.05);
      border-bottom: 1px solid rgba(99, 102, 241, 0.1);
      cursor: pointer;
      transition: all 0.15s;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .custom-dropdown-item .layer-id {
      flex: 1;
      text-align: center;
    }
    .custom-dropdown-item:last-child {
      border-bottom: none;
    }
    .custom-dropdown-item:hover {
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--accent-primary);
    }
    .custom-dropdown-item.selected {
      background: rgba(99, 102, 241, 0.25);
      color: var(--accent-primary);
    }
    .custom-dropdown-menu::-webkit-scrollbar {
      width: 8px;
    }
    .custom-dropdown-menu::-webkit-scrollbar-track {
      background: rgba(255, 255, 255, 0.03);
      border-radius: 4px;
    }
    .custom-dropdown-menu::-webkit-scrollbar-thumb {
      background: rgba(99, 102, 241, 0.3);
      border-radius: 4px;
    }
    .custom-dropdown-menu::-webkit-scrollbar-thumb:hover {
      background: rgba(99, 102, 241, 0.5);
    }
    .control {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 12px;
    }
    .control label {
      font-size: 12px;
      color: var(--text-secondary);
    }
    .control button {
      padding: 8px 10px;
      font-family: 'Space Grotesk', sans-serif;
      font-size: 12px;
      font-weight: 600;
      color: var(--accent-primary);
      background: rgba(99, 102, 241, 0.1);
      border: 1px solid rgba(99, 102, 241, 0.2);
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
      text-align: left;
    }
    .control button:hover {
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--accent-primary);
    }
    input[type=\"range\"] {
      -webkit-appearance: none;
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
      outline: none;
      opacity: 0.8;
      transition: opacity 0.2s;
    }
    input[type=\"range\"]:hover {
      opacity: 1;
    }
    input[type=\"range\"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--text-primary);
      cursor: pointer;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
      transition: transform 0.2s;
    }
    input[type=\"range\"]::-webkit-slider-thumb:hover {
      transform: scale(1.15);
    }
    input[type=\"range\"]::-moz-range-thumb {
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--text-primary);
      cursor: pointer;
      border: none;
    }
    .colorbar {
      margin-top: 12px;
      padding: 10px;
      border-radius: 10px;
      background: rgba(255,255,255,0.02);
      border: 1px solid rgba(255,255,255,0.05);
    }
    .colorbar-title {
      font-size: 11px;
      margin-bottom: 6px;
      color: var(--text-secondary);
    }
    .colorbar .layer-select-wrapper {
      margin-left: 28px;
      width: calc(100% - 28px);
      margin-bottom: 12px;
    }
    .dropdown-label {
      color: var(--text-secondary);
      font-size: 11px;
      letter-spacing: 0.2px;
      flex: 0 0 72px;
    }
    .cmap-swatch {
      width: 30px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.18);
      box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.14);
      flex: 0 0 auto;
    }
    .cmap-dropdown-prefix {
      flex: 0 0 72px;
      display: flex;
      align-items: center;
      justify-content: flex-start;
    }
    .cmap-dropdown-item .layer-id {
      text-align: left;
    }
    .cmap-dropdown-item {
      justify-content: flex-start;
    }
    .colorbar-visual {
      position: relative;
      height: 44px;
      margin-bottom: 6px;
    }
    .colorbar-distribution {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
    }
    .colorbar-distribution-fill {
      fill: var(--accent-secondary);
      opacity: 0.18;
    }
    .colorbar-distribution-line {
      fill: none;
      stroke: var(--accent-secondary);
      stroke-width: 1.4;
      stroke-linecap: butt;
      stroke-linejoin: round;
      vector-effect: non-scaling-stroke;
    }
    .colorbar-gradient {
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      height: 12px;
      border-radius: 999px;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
    }
    .colorbar-range {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
    }
    #info-panel {
      position: fixed;
      bottom: 20px;
      left: 20px;
      z-index: 100;
      background: var(--bg-panel);
      backdrop-filter: blur(20px);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 16px 20px;
      font-size: 11px;
      color: var(--text-secondary);
    }
    #info-panel .info-row {
      display: flex;
      gap: 20px;
    }
    #info-panel .info-item {
      display: flex;
      gap: 6px;
    }
    #info-panel .info-label {
      color: var(--text-secondary);
    }
    #info-panel .info-value {
      color: var(--accent-secondary);
      font-weight: 500;
    }
    #hover-tooltip {
      position: fixed;
      z-index: 40;
      pointer-events: none;
      display: none;
      min-width: 190px;
      max-width: 320px;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--border-color);
      background: var(--bg-panel);
      color: var(--text-primary);
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      line-height: 1.45;
      white-space: pre-line;
      backdrop-filter: blur(12px);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.28);
    }
    .hover-tooltip-title {
      margin-bottom: 6px;
      font-weight: 600;
    }
    .hover-tooltip-row {
      display: flex;
      align-items: center;
      gap: 8px;
      white-space: normal;
    }
    .hover-tooltip-label {
      min-width: 72px;
      color: var(--text-secondary);
    }
    .hover-tooltip-value {
      min-width: 0;
      flex: 1 1 auto;
      word-break: break-word;
    }
    .hover-tooltip-swatch {
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.18);
      box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.14);
      flex: 0 0 auto;
    }
  </style>
</head>
<body>
  <div id=\"canvas-container\"></div>
  <div id=\"top-actions\">
    <button id=\"btn-quit-top\" class=\"top-action-btn danger\" title=\"Quit viewer\" aria-label=\"Quit viewer\">Quit Viewer</button>
    <button id=\"btn-toggle-commands\" class=\"top-action-btn text-btn\" title=\"Hide commands\" aria-label=\"Hide commands\">Hide Commands</button>
    <button id=\"btn-theme-mode\" class=\"top-action-btn\" title=\"Switch to day mode\" aria-label=\"Switch to day mode\">☾</button>
  </div>
  <div id=\"controls\">
    <h1>__TITLE__</h1>
    <div class=\"section\">
      <div class=\"section-title\">Display</div>
      <div class=\"control\">
        <label for=\"mesh-opacity\">Mesh opacity</label>
        <input id=\"mesh-opacity\" type=\"range\" min=\"0.05\" max=\"1\" step=\"0.05\" value=\"__OPACITY__\">
      </div>
      <div class=\"control\">
        <label><input id=\"edge-overlay-mode\" type=\"checkbox\"> Show mesh edges</label>
      </div>
      <div class=\"control\">
        <button id=\"btn-reset-camera\" type=\"button\">Reset camera</button>
      </div>
    </div>
    <div class=\"section\">
      <div class=\"section-title\">Layers</div>
      <div id=\"layer-list\"></div>
    </div>
    <div class=\"section\">
      <div class=\"section-title\">Fields</div>
      <div class=\"control\">
        <label><input id=\"share-field-scale\" type=\"checkbox\"> Share field colormap scale</label>
      </div>
      <div id=\"colorbars\"></div>
    </div>
  </div>
  <div id=\"info-panel\">
    <div class=\"info-row\">
      <div class=\"info-item\">
        <span class=\"info-label\">Vertices:</span>
        <span class=\"info-value\" id=\"info-vertices\">--</span>
      </div>
      <div class=\"info-item\">
        <span class=\"info-label\">Faces:</span>
        <span class=\"info-value\" id=\"info-faces\">--</span>
      </div>
    </div>
  </div>
  <div id=\"hover-tooltip\"></div>
  <div id=\"dropdown-portal\"></div>

  <script type=\"importmap\">
    {
      \"imports\": {
        \"three\": \"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js\",
        \"three/addons/\": \"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/\",
        \"three-mesh-bvh\": \"https://cdn.jsdelivr.net/npm/three-mesh-bvh@0.9.9/build/index.module.js\"
      }
    }
  </script>
  <script type=\"module\">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import {
      acceleratedRaycast,
      computeBoundsTree,
      disposeBoundsTree,
    } from 'three-mesh-bvh';

    const payload = __PAYLOAD__;
    const colormapPresets = payload.colormap_presets || {};
    const availableCmaps = payload.available_cmaps || Object.keys(colormapPresets);
    THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
    THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
    THREE.Mesh.prototype.raycast = acceleratedRaycast;
    const arrayCtors = {
      float32: Float32Array,
      uint8: Uint8Array,
      uint32: Uint32Array,
      int32: Int32Array,
    };

    function decodeArraySpec(spec, fallbackCtor = Float32Array) {
      if (spec === null || spec === undefined) return null;
      if (ArrayBuffer.isView(spec)) return spec;
      if (spec && spec.encoding === 'base64' && typeof spec.data === 'string') {
        const ctor = arrayCtors[spec.dtype] || fallbackCtor;
        const binary = atob(spec.data);
        const bytes = new Uint8Array(binary.length);
        for (let idx = 0; idx < binary.length; idx += 1) {
          bytes[idx] = binary.charCodeAt(idx);
        }
        return new ctor(bytes.buffer);
      }
      if (!Array.isArray(spec)) {
        return null;
      }
      if (spec.length === 0) {
        return new fallbackCtor(0);
      }
      return new fallbackCtor(Array.isArray(spec[0]) ? spec.flat() : spec);
    }

    function normalizeLayerData(layer) {
      layer.vertices = decodeArraySpec(layer.vertices, Float32Array) || new Float32Array(0);
      layer.faces = decodeArraySpec(layer.faces, Uint32Array) || new Uint32Array(0);
      layer.colors = decodeArraySpec(layer.colors, Uint8Array) || new Uint8Array(0);
      layer.values = decodeArraySpec(layer.values, Float32Array);
      layer.field_colors = Array.isArray(layer.field_colors)
        ? layer.field_colors.map((colors) => decodeArraySpec(colors, Uint8Array) || new Uint8Array(0))
        : null;
      layer.field_values = Array.isArray(layer.field_values)
        ? layer.field_values.map((values) => decodeArraySpec(values, Float32Array))
        : null;
      layer.vertex_count = Number.isFinite(layer.vertex_count)
        ? Number(layer.vertex_count)
        : Math.floor(layer.vertices.length / 3);
      layer.face_count = Number.isFinite(layer.face_count)
        ? Number(layer.face_count)
        : Math.floor(layer.faces.length / 3);
      return layer;
    }

    function createStoredColorAttribute(colors) {
      const attr = new THREE.BufferAttribute(colors, 3, true);
      attr.setUsage(THREE.DynamicDrawUsage);
      return attr;
    }

    function setStoredColorAttribute(geometry, colors) {
      const colorArray = decodeArraySpec(colors, Uint8Array) || new Uint8Array(0);
      const existing = geometry.getAttribute('color');
      if (
        existing
        && existing.array instanceof Uint8Array
        && existing.array.length === colorArray.length
        && existing.normalized
      ) {
        existing.array.set(colorArray);
        existing.needsUpdate = true;
        return existing;
      }
      const attr = createStoredColorAttribute(colorArray);
      geometry.setAttribute('color', attr);
      return attr;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.001, 1e7);
    camera.position.set(0, 0, 4);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x111827, 1.0);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    controls.screenSpacePanning = true;
    controls.zoomSpeed = 1.0;
    controls.rotateSpeed = 0.9;
    controls.panSpeed = 0.8;

    const layers = [];
    let isDayMode = false;
    let commandsVisible = true;
    let shareFieldScale = false;
    const raycaster = new THREE.Raycaster();
    raycaster.firstHitOnly = true;
    const pointerNdc = new THREE.Vector2();
    const hoverTooltip = document.getElementById('hover-tooltip');
    const dropdownPortal = document.getElementById('dropdown-portal');
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
    const hemisphereLight = new THREE.HemisphereLight(0xe0f2fe, 0x111827, 0.7);
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.15);
    keyLight.position.set(2.5, 3.5, 4.0);
    const rimLight = new THREE.DirectionalLight(0x93c5fd, 0.4);
    rimLight.position.set(-3.0, 1.5, -2.5);
    scene.add(ambientLight, hemisphereLight, keyLight, rimLight);
    const pointerState = {
      inside: false,
      clientX: 0,
      clientY: 0,
    };
    let renderRequested = false;
    let hoverRefreshQueued = false;

    function renderScene() {
      renderRequested = false;
      renderer.render(scene, camera);
    }

    function requestRender() {
      if (renderRequested) return;
      renderRequested = true;
      requestAnimationFrame(renderScene);
    }

    function scheduleHoverRefresh() {
      if (!pointerState.inside || hoverRefreshQueued) return;
      hoverRefreshQueued = true;
      requestAnimationFrame(() => {
        hoverRefreshQueued = false;
        updateHoverFromPointer();
      });
    }

    function requestRenderAndHover() {
      requestRender();
      scheduleHoverRefresh();
    }

    function collectScalarMetas(layer) {
      const metas = [];
      if (layer && layer.scalar_meta) metas.push(layer.scalar_meta);
      if (layer && Array.isArray(layer.field_scalar_meta)) {
        layer.field_scalar_meta.forEach((meta) => {
          if (meta) metas.push(meta);
        });
      }
      return metas;
    }

    const sharedFieldScaleRange = (() => {
      let minVal = Infinity;
      let maxVal = -Infinity;
      payload.layers.forEach((layer) => {
        collectScalarMetas(layer).forEach((meta) => {
          const metaMin = Number(meta && meta.min);
          const metaMax = Number(meta && meta.max);
          if (!Number.isFinite(metaMin) || !Number.isFinite(metaMax)) return;
          minVal = Math.min(minVal, metaMin);
          maxVal = Math.max(maxVal, metaMax);
        });
      });
      if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null;
      return { min: minVal, max: maxVal };
    })();

    function edgeColorHex() {
      return isDayMode ? 0x0f172a : 0xffffff;
    }

    function applyLightingMode(dayMode) {
      ambientLight.intensity = dayMode ? 0.7 : 0.5;
      hemisphereLight.intensity = dayMode ? 0.85 : 0.65;
      hemisphereLight.color.set(dayMode ? 0xf8fafc : 0xe0f2fe);
      hemisphereLight.groundColor.set(dayMode ? 0xcbd5e1 : 0x111827);
      keyLight.intensity = dayMode ? 1.05 : 1.2;
      keyLight.color.set(dayMode ? 0xffffff : 0xf8fafc);
      rimLight.intensity = dayMode ? 0.28 : 0.42;
    }

    function applyThemeMode(dayMode) {
      isDayMode = Boolean(dayMode);
      document.body.classList.toggle('day-mode', isDayMode);
      const css = getComputedStyle(document.body);
      const bg = css.getPropertyValue('--bg-canvas-solid').trim() || '#111827';
      renderer.setClearColor(bg, 1.0);
      applyLightingMode(isDayMode);
      const btn = document.getElementById('btn-theme-mode');
      if (btn) {
        btn.textContent = isDayMode ? '☀' : '☾';
        btn.title = isDayMode ? 'Switch to night mode' : 'Switch to day mode';
        btn.setAttribute('aria-label', btn.title);
      }
      refreshEdgeOverlay();
      requestRender();
    }

    function buildMesh(layer) {
      const geometry = new THREE.BufferGeometry();
      const positionAttr = new THREE.BufferAttribute(layer.vertices, 3);
      positionAttr.setUsage(THREE.StaticDrawUsage);
      geometry.setAttribute('position', positionAttr);
      geometry.setAttribute('color', createStoredColorAttribute(layer.colors));
      const indexAttr = new THREE.BufferAttribute(layer.faces, 1);
      indexAttr.setUsage(THREE.StaticDrawUsage);
      geometry.setIndex(indexAttr);
      geometry.computeVertexNormals();
      geometry.computeBoundingBox();
      geometry.computeBoundingSphere();
      if (layer.face_count > 0) {
        geometry.computeBoundsTree();
      }

      const initialOpacity = Number(payload.mesh_opacity) || 1.0;
      const material = new THREE.MeshLambertMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        transparent: initialOpacity < 0.999,
        opacity: initialOpacity,
        depthWrite: initialOpacity >= 0.999,
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.frustumCulled = true;
      scene.add(mesh);

      const edgeGeometry = new THREE.WireframeGeometry(geometry);
      const edgeMaterial = new THREE.LineBasicMaterial({
        color: edgeColorHex(),
        transparent: true,
        opacity: isDayMode ? 0.92 : 1.0,
        depthTest: false,
        depthWrite: false,
      });
      const edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial);
      edgeLines.visible = false;
      edgeLines.frustumCulled = true;
      edgeLines.renderOrder = 3;
      scene.add(edgeLines);
      return { mesh, edgeLines };
    }

    function refreshEdgeOverlay() {
      const edgeInput = document.getElementById('edge-overlay-mode');
      const showEdges = Boolean(edgeInput && edgeInput.checked);
      const color = edgeColorHex();
      for (const entry of layers) {
        if (!entry.edgeLines) continue;
        entry.edgeLines.visible = entry.visible && showEdges;
        entry.edgeLines.material.color.setHex(color);
        entry.edgeLines.material.opacity = isDayMode ? 0.82 : 0.92;
      }
      requestRender();
    }

    function getFallbackCmap() {
      return availableCmaps[0] || 'viridis';
    }

    function getColormapPreset(cmapName) {
      if (cmapName && colormapPresets[cmapName]) return colormapPresets[cmapName];
      const fallbackName = getFallbackCmap();
      return colormapPresets[fallbackName] || null;
    }

    function gradientCss(gradient) {
      if (!Array.isArray(gradient) || gradient.length === 0) {
        return 'linear-gradient(90deg, #64748b, #e2e8f0)';
      }
      return `linear-gradient(90deg, ${gradient.join(', ')})`;
    }

    function getEntryDisplayedValues(entry, fieldIdx = entry.activeFieldIdx ?? 0) {
      const fieldValues = entry.layer.field_values || null;
      return fieldValues ? fieldValues[fieldIdx] : entry.layer.values || null;
    }

    function buildDistributionSvg(values, meta) {
      if (!values || typeof values.length !== 'number' || values.length === 0 || !meta) return null;

      const minVal = Number(meta.min);
      const maxVal = Number(meta.max);
      if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null;

      const bins = 48;
      const counts = new Float32Array(bins);
      const span = maxVal - minVal;
      const hasSpan = Number.isFinite(span) && Math.abs(span) > 1e-12;
      let finiteCount = 0;

      for (let idx = 0; idx < values.length; idx += 1) {
        const raw = Number(values[idx]);
        if (!Number.isFinite(raw)) continue;
        finiteCount += 1;

        let binIdx = 0;
        if (hasSpan) {
          const normalized = Math.min(1, Math.max(0, (raw - minVal) / span));
          binIdx = Math.min(bins - 1, Math.floor(normalized * bins));
        } else {
          binIdx = Math.floor((bins - 1) * 0.5);
        }
        counts[binIdx] += 1;
      }

      if (finiteCount === 0) return null;

      let peak = 0;
      for (let idx = 0; idx < bins; idx += 1) {
        peak = Math.max(peak, counts[idx]);
      }
      if (!(peak > 0)) return null;

      const width = 100;
      const height = 44;
      const baselineY = 29;
      const curveTopY = 4;
      const curveHeight = baselineY - curveTopY;
      const strokeInset = 0.7;
      const xStep = (width - 2 * strokeInset) / (bins - 1);

      let linePath = '';
      for (let idx = 0; idx < bins; idx += 1) {
        const x = strokeInset + idx * xStep;
        const y = baselineY - (counts[idx] / peak) * curveHeight;
        linePath += `${idx === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)} `;
      }

      const areaPath = `${linePath}L ${(width - strokeInset).toFixed(2)} ${baselineY} L ${strokeInset.toFixed(2)} ${baselineY} Z`;
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      svg.setAttribute('preserveAspectRatio', 'none');
      svg.classList.add('colorbar-distribution');

      const fillPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      fillPath.setAttribute('d', areaPath.trim());
      fillPath.setAttribute('class', 'colorbar-distribution-fill');

      const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      line.setAttribute('d', linePath.trim());
      line.setAttribute('class', 'colorbar-distribution-line');

      svg.appendChild(fillPath);
      svg.appendChild(line);
      return svg;
    }

    function createCmapSwatch(cmapName) {
      const swatch = document.createElement('span');
      swatch.className = 'cmap-swatch';
      const preset = getColormapPreset(cmapName);
      swatch.style.background = gradientCss(preset && preset.gradient ? preset.gradient : null);
      return swatch;
    }

    function setCmapButtonContent(button, cmapName) {
      button.replaceChildren();
      const prefix = document.createElement('span');
      prefix.className = 'dropdown-label';
      prefix.textContent = 'Colormap:';
      const value = document.createElement('span');
      value.className = 'layer-id';
      value.textContent = cmapName;
      button.appendChild(prefix);
      button.appendChild(value);
    }

    function setFieldButtonContent(button, fieldName) {
      button.replaceChildren();
      const prefix = document.createElement('span');
      prefix.className = 'dropdown-label';
      prefix.textContent = 'Field:';
      const value = document.createElement('span');
      value.className = 'layer-id';
      value.textContent = fieldName;
      button.appendChild(prefix);
      button.appendChild(value);
    }

    function closeDropdownMenu(menu) {
      if (dropdownPortal && menu.parentElement === dropdownPortal) {
        const host = menu.__hostElement || null;
        if (host) {
          host.appendChild(menu);
        }
      }
      menu.classList.remove('floating');
      menu.classList.remove('open');
      menu.style.top = '';
      menu.style.left = '';
      menu.style.width = '';
      menu.style.maxHeight = '';
    }

    function closeAllDropdownMenus(exceptMenu = null) {
      document.querySelectorAll('.custom-dropdown-menu.open').forEach((node) => {
        if (node !== exceptMenu) closeDropdownMenu(node);
      });
    }

    function positionDropdownMenu(button, menu) {
      const rect = button.getBoundingClientRect();
      const margin = 12;
      const gap = 4;
      const menuWidth = rect.width;
      const left = Math.min(
        window.innerWidth - menuWidth - margin,
        Math.max(margin, rect.left),
      );
      const spaceBelow = window.innerHeight - rect.bottom - margin - gap;
      const spaceAbove = rect.top - margin - gap;
      const openUpward = spaceBelow < 220 && spaceAbove > spaceBelow;
      const availableHeight = openUpward ? spaceAbove : spaceBelow;
      const maxHeight = Math.max(140, Math.min(420, availableHeight));

      menu.style.left = `${left}px`;
      menu.style.width = `${menuWidth}px`;
      menu.style.maxHeight = `${maxHeight}px`;

      if (openUpward) {
        const menuHeight = Math.min(menu.scrollHeight || maxHeight, maxHeight);
        menu.style.top = `${Math.max(margin, rect.top - menuHeight - gap)}px`;
      } else {
        menu.style.top = `${Math.min(
          window.innerHeight - maxHeight - margin,
          Math.max(margin, rect.bottom + gap),
        )}px`;
      }
    }

    function toggleDropdownMenu(button, menu) {
      const willOpen = !menu.classList.contains('open');
      closeAllDropdownMenus(willOpen ? menu : null);
      if (!willOpen) {
        closeDropdownMenu(menu);
        return;
      }
      if (dropdownPortal && menu.parentElement !== dropdownPortal) {
        menu.__hostElement = menu.parentElement;
        dropdownPortal.appendChild(menu);
      }
      menu.classList.add('floating');
      menu.classList.add('open');
      positionDropdownMenu(button, menu);
    }

    function getEntryCmap(entry, fieldIdx = entry.activeFieldIdx ?? 0) {
      return entry.activeCmap || getFallbackCmap();
    }

    function getScalarMeta(entry, fieldIdx = entry.activeFieldIdx ?? 0) {
      const fieldMetas = entry.layer.field_scalar_meta || null;
      const baseMeta = fieldMetas ? fieldMetas[fieldIdx] : entry.layer.scalar_meta;
      if (!baseMeta) return null;
      const cmapName = getEntryCmap(entry, fieldIdx);
      const preset = getColormapPreset(cmapName);
      const resolved = {
        ...baseMeta,
        cmap: cmapName,
        gradient: (preset && preset.gradient) ? preset.gradient : baseMeta.gradient,
      };
      if (shareFieldScale && sharedFieldScaleRange) {
        resolved.min = sharedFieldScaleRange.min;
        resolved.max = sharedFieldScaleRange.max;
      }
      return resolved;
    }

    function applyStoredColors(entry, colors) {
      setStoredColorAttribute(entry.mesh.geometry, colors);
    }

    function applyColorsFromValues(entry, values, meta) {
      const preset = getColormapPreset(meta && meta.cmap);
      const samples = preset && Array.isArray(preset.samples) ? preset.samples : null;
      if (!samples || samples.length === 0 || !values || typeof values.length !== 'number') return false;

      const colorCount = values.length * 3;
      const existing = entry.mesh.geometry.getAttribute('color');
      const colors = (
        existing
        && existing.array instanceof Uint8Array
        && existing.array.length === colorCount
        && existing.normalized
      )
        ? existing.array
        : new Uint8Array(colorCount);
      const minVal = Number.isFinite(meta.min) ? meta.min : 0.0;
      const maxVal = Number.isFinite(meta.max) ? meta.max : 1.0;
      const denom = Math.max(maxVal - minVal, 1e-8);
      const sampleMaxIdx = samples.length - 1;

      for (let idx = 0; idx < values.length; idx += 1) {
        const raw = Number(values[idx]);
        const normalized = Number.isFinite(raw) ? Math.min(1, Math.max(0, (raw - minVal) / denom)) : 0.0;
        const scaled = normalized * sampleMaxIdx;
        const leftIdx = Math.floor(scaled);
        const rightIdx = Math.min(sampleMaxIdx, leftIdx + 1);
        const mix = scaled - leftIdx;
        const left = samples[leftIdx];
        const right = samples[rightIdx];
        colors[(idx * 3)] = Math.round(left[0] + (right[0] - left[0]) * mix);
        colors[(idx * 3) + 1] = Math.round(left[1] + (right[1] - left[1]) * mix);
        colors[(idx * 3) + 2] = Math.round(left[2] + (right[2] - left[2]) * mix);
      }

      if (
        existing
        && existing.array instanceof Uint8Array
        && existing.array.length === colorCount
        && existing.normalized
      ) {
        existing.needsUpdate = true;
        return true;
      }

      entry.mesh.geometry.setAttribute('color', createStoredColorAttribute(colors));
      return true;
    }

    function renderEntryColors(entry, fieldIdx = entry.activeFieldIdx ?? 0) {
      const fieldValues = entry.layer.field_values || null;
      const fieldColors = entry.layer.field_colors || null;
      const values = fieldValues ? fieldValues[fieldIdx] : entry.layer.values || null;
      const meta = getScalarMeta(entry, fieldIdx);
      const updatedFromValues = values && meta ? applyColorsFromValues(entry, values, meta) : false;

      if (updatedFromValues) {
        entry.mesh.geometry.attributes.color.needsUpdate = true;
      } else if (fieldColors && fieldIdx >= 0 && fieldIdx < fieldColors.length) {
        applyStoredColors(entry, fieldColors[fieldIdx]);
      } else {
        applyStoredColors(entry, entry.layer.colors);
      }

      entry.activeFieldIdx = fieldIdx;
      entry.activeValues = values;
      if (!fieldValues) {
        entry.activeCmap = meta ? meta.cmap : entry.activeCmap;
      }
      requestRenderAndHover();
    }

    function setLayerCmap(entry, cmapName) {
      const fieldIdx = entry.activeFieldIdx ?? 0;
      entry.activeCmap = cmapName;
      renderEntryColors(entry, fieldIdx);
      updateColorbars();
    }

    function setShareFieldScale(enabled) {
      shareFieldScale = Boolean(enabled);
      for (const entry of layers) {
        const fieldIdx = Math.max(
          0,
          Math.min(
            entry.activeFieldIdx ?? 0,
            entry.fieldCmaps && entry.fieldCmaps.length > 0 ? entry.fieldCmaps.length - 1 : 0,
          ),
        );
        renderEntryColors(entry, fieldIdx);
      }
      updateColorbars();
      hideHoverTooltip();
    }

    function fitCamera() {
      const box = new THREE.Box3();
      let hasBox = false;
      for (const entry of layers) {
        if (!entry.visible) continue;
        box.expandByObject(entry.mesh);
        hasBox = true;
      }
      if (!hasBox || box.isEmpty()) return;

      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z) * 0.5 || 1.0;
      const fov = THREE.MathUtils.degToRad(camera.fov);
      const distance = (radius / Math.sin(fov / 2)) * 1.35;
      camera.position.set(center.x + distance, center.y + distance, center.z + distance);

      const near = Math.max(distance / 5000, 1e-6);
      const far = Math.max(distance * 50, 10.0);
      camera.near = near;
      camera.far = far;
      camera.updateProjectionMatrix();

      controls.target.copy(center);
      controls.update();
      requestRenderAndHover();
    }

    function updateColorbars() {
      const container = document.getElementById('colorbars');
      container.innerHTML = '';
      for (const entry of layers) {
        if (!entry.visible) continue;
        const fieldIdx = entry.activeFieldIdx ?? 0;
        const fieldNames = entry.layer.field_names || null;
        const meta = getScalarMeta(entry, fieldIdx);
        if (!meta) continue;

        const colorbarLabel = fieldNames
          ? `${entry.layer.label} - ${fieldNames[fieldIdx]}`
          : entry.layer.label;

        const wrapper = document.createElement('div');
        wrapper.className = 'colorbar';
        const title = document.createElement('div');
        title.className = 'colorbar-title';
        title.textContent = colorbarLabel;
        const dropdownWrap = document.createElement('div');
        dropdownWrap.className = 'layer-select-wrapper';
        const dropdown = document.createElement('div');
        dropdown.className = 'custom-dropdown';
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'layer-select-button cmap-dropdown-button';
        const currentCmap = meta.cmap || getFallbackCmap();
        setCmapButtonContent(button, currentCmap);
        const menu = document.createElement('div');
        menu.className = 'custom-dropdown-menu';
        availableCmaps.forEach((cmapName) => {
          const item = document.createElement('div');
          item.className = 'custom-dropdown-item cmap-dropdown-item';
          if (cmapName === currentCmap) item.classList.add('selected');
          const prefix = document.createElement('span');
          prefix.className = 'cmap-dropdown-prefix';
          prefix.appendChild(createCmapSwatch(cmapName));
          item.appendChild(prefix);
          const value = document.createElement('span');
          value.className = 'layer-id';
          value.textContent = cmapName;
          item.appendChild(value);
          item.addEventListener('click', () => {
            setLayerCmap(entry, cmapName);
            setCmapButtonContent(button, cmapName);
            menu.querySelectorAll('.custom-dropdown-item').forEach((el) => {
              el.classList.toggle('selected', el === item);
            });
            closeDropdownMenu(menu);
          });
          menu.appendChild(item);
        });
        button.addEventListener('click', (ev) => {
          ev.stopPropagation();
          toggleDropdownMenu(button, menu);
        });
        const visual = document.createElement('div');
        visual.className = 'colorbar-visual';
        const distribution = buildDistributionSvg(getEntryDisplayedValues(entry, fieldIdx), meta);
        const gradient = document.createElement('div');
        gradient.className = 'colorbar-gradient';
        gradient.style.background = gradientCss(meta.gradient);
        const range = document.createElement('div');
        range.className = 'colorbar-range';
        range.innerHTML = `<span>${meta.min.toFixed(4)}</span><span>${meta.max.toFixed(4)}</span>`;
        dropdown.appendChild(button);
        dropdown.appendChild(menu);
        dropdownWrap.appendChild(dropdown);
        wrapper.appendChild(title);
        wrapper.appendChild(dropdownWrap);
        if (distribution) {
          visual.appendChild(distribution);
        }
        visual.appendChild(gradient);
        wrapper.appendChild(visual);
        wrapper.appendChild(range);
        container.appendChild(wrapper);
      }
    }

    function updateInfoPanel() {
      let totalVertices = 0;
      let totalFaces = 0;
      for (const entry of layers) {
        if (!entry.visible) continue;
        totalVertices += Number(entry.layer.vertex_count) || 0;
        totalFaces += Number(entry.layer.face_count) || 0;
      }
      const verticesNode = document.getElementById('info-vertices');
      const facesNode = document.getElementById('info-faces');
      if (verticesNode) verticesNode.textContent = totalVertices.toLocaleString();
      if (facesNode) facesNode.textContent = totalFaces.toLocaleString();
    }

    function setLayerField(entry, fieldIdx) {
      const fieldNames = entry.layer.field_names || null;
      if (!fieldNames || fieldIdx < 0 || fieldIdx >= fieldNames.length) return;
      renderEntryColors(entry, fieldIdx);
      updateColorbars();
    }

    function hideHoverTooltip() {
      hoverTooltip.style.display = 'none';
    }

    function toHexChannel(value) {
      return Math.round(Math.min(255, Math.max(0, value))).toString(16).padStart(2, '0');
    }

    function formatHoverColor(colorAttr, idx) {
      if (!colorAttr || idx < 0 || idx >= colorAttr.count) return null;
      let r = colorAttr.getX(idx);
      let g = colorAttr.getY(idx);
      let b = colorAttr.getZ(idx);
      if (![r, g, b].every(Number.isFinite)) return null;
      const red = Math.round(Math.min(1, Math.max(0, r)) * 255);
      const green = Math.round(Math.min(1, Math.max(0, g)) * 255);
      const blue = Math.round(Math.min(1, Math.max(0, b)) * 255);
      return {
        rgb: `rgb(${red}, ${green}, ${blue})`,
        hex: `#${toHexChannel(red)}${toHexChannel(green)}${toHexChannel(blue)}`,
      };
    }

    function buildHoverTooltipContent(details) {
      hoverTooltip.replaceChildren();

      const title = document.createElement('div');
      title.className = 'hover-tooltip-title';
      title.textContent = details.layerLabel;
      hoverTooltip.appendChild(title);

      const addRow = (label, value = null, extraNode = null) => {
        const row = document.createElement('div');
        row.className = 'hover-tooltip-row';

        const labelNode = document.createElement('div');
        labelNode.className = 'hover-tooltip-label';
        labelNode.textContent = label;
        row.appendChild(labelNode);

        if (extraNode) {
          row.appendChild(extraNode);
        }

        if (value !== null && value !== undefined && value !== '') {
          const valueNode = document.createElement('div');
          valueNode.className = 'hover-tooltip-value';
          valueNode.textContent = value;
          row.appendChild(valueNode);
        }

        hoverTooltip.appendChild(row);
      };

      addRow('xyz', details.xyz);
      addRow('color', null, (() => {
        const swatch = document.createElement('div');
        swatch.className = 'hover-tooltip-swatch';
        swatch.style.background = details.color.rgb;
        return swatch;
      })());
      addRow('value', details.valueText);
    }

    function showHoverTooltip(clientX, clientY, details) {
      buildHoverTooltipContent(details);
      hoverTooltip.style.display = 'block';
      const margin = 12;
      const maxX = window.innerWidth - hoverTooltip.offsetWidth - margin;
      const maxY = window.innerHeight - hoverTooltip.offsetHeight - margin;
      hoverTooltip.style.left = `${Math.min(clientX + 14, Math.max(margin, maxX))}px`;
      hoverTooltip.style.top = `${Math.min(clientY + 14, Math.max(margin, maxY))}px`;
    }

    function pickClosestFaceVertexIndex(hit, mesh, posAttr) {
      if (!hit || !hit.face || !mesh || !posAttr) return -1;
      const localHit = mesh.worldToLocal(hit.point.clone());
      const candidates = [hit.face.a, hit.face.b, hit.face.c];
      let bestIdx = -1;
      let bestDist2 = Infinity;
      for (const idx of candidates) {
        if (idx < 0 || idx >= posAttr.count) continue;
        const dx = posAttr.getX(idx) - localHit.x;
        const dy = posAttr.getY(idx) - localHit.y;
        const dz = posAttr.getZ(idx) - localHit.z;
        const dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 < bestDist2) {
          bestDist2 = dist2;
          bestIdx = idx;
        }
      }
      return bestIdx;
    }

    function updateHoverFromPointer() {
      if (!pointerState.inside) {
        hideHoverTooltip();
        return;
      }
      const rect = renderer.domElement.getBoundingClientRect();
      const x = (pointerState.clientX - rect.left) / rect.width;
      const y = (pointerState.clientY - rect.top) / rect.height;
      pointerNdc.x = x * 2 - 1;
      pointerNdc.y = -(y * 2 - 1);
      raycaster.setFromCamera(pointerNdc, camera);

      let best = null;
      for (const entry of layers) {
        if (!entry.visible) continue;
        const hits = raycaster.intersectObject(entry.mesh, false);
        if (!hits || hits.length === 0) continue;
        const hit = hits[0];
        if (!best || hit.distance < best.hit.distance) {
          best = { entry, hit };
        }
      }

      if (!best) {
        hideHoverTooltip();
        return;
      }

      const posAttr = best.entry.mesh.geometry.getAttribute('position');
      const colorAttr = best.entry.mesh.geometry.getAttribute('color');
      const idx = pickClosestFaceVertexIndex(best.hit, best.entry.mesh, posAttr);
      if (idx < 0) {
        hideHoverTooltip();
        return;
      }

      const px = posAttr.getX(idx);
      const py = posAttr.getY(idx);
      const pz = posAttr.getZ(idx);
      const hoverColor = formatHoverColor(colorAttr, idx);
      if (!hoverColor) {
        hideHoverTooltip();
        return;
      }

      const fieldNames = best.entry.layer.field_names || null;
      const fieldLabel = fieldNames ? fieldNames[best.entry.activeFieldIdx ?? 0] : null;
      const activeValues = best.entry.activeValues || best.entry.layer.values || null;
      const value = activeValues && idx < activeValues.length ? activeValues[idx] : null;
      const valueText = (value === null || value === undefined || !Number.isFinite(value))
        ? 'n/a'
        : Number(value).toFixed(4);

      showHoverTooltip(pointerState.clientX, pointerState.clientY, {
        layerLabel: `${best.entry.layer.label}${fieldLabel ? ` (${fieldLabel})` : ''}`,
        xyz: `(${px.toFixed(4)}, ${py.toFixed(4)}, ${pz.toFixed(4)})`,
        color: hoverColor,
        valueText: valueText,
      });
    }

    function rebuildLayerList() {
      const container = document.getElementById('layer-list');
      container.innerHTML = '';
      layers.forEach((entry, idx) => {
        const row = document.createElement('div');
        row.className = 'layer-row';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = entry.visible;
        checkbox.addEventListener('change', () => {
          entry.visible = checkbox.checked;
          entry.mesh.visible = entry.visible;
          if (entry.edgeLines) {
            const edgeInput = document.getElementById('edge-overlay-mode');
            entry.edgeLines.visible = entry.visible && Boolean(edgeInput && edgeInput.checked);
          }
          updateColorbars();
          updateInfoPanel();
          hideHoverTooltip();
          requestRender();
        });

        const info = document.createElement('div');
        const label = document.createElement('div');
        label.className = 'layer-label';
        label.textContent = `${idx + 1}. ${entry.layer.label}`;
        info.appendChild(label);

        const fieldNames = entry.layer.field_names || [];
        if (fieldNames.length > 1) {
          const wrap = document.createElement('div');
          wrap.className = 'layer-select-wrapper';
          const dropdown = document.createElement('div');
          dropdown.className = 'custom-dropdown';
          const button = document.createElement('button');
          button.type = 'button';
          button.className = 'layer-select-button';
          setFieldButtonContent(button, fieldNames[entry.activeFieldIdx ?? 0]);
          const menu = document.createElement('div');
          menu.className = 'custom-dropdown-menu';

          fieldNames.forEach((name, fieldIdx) => {
            const item = document.createElement('div');
            item.className = 'custom-dropdown-item';
            if (fieldIdx === (entry.activeFieldIdx ?? 0)) item.classList.add('selected');
            const value = document.createElement('span');
            value.className = 'layer-id';
            value.textContent = name;
            item.appendChild(value);
            item.addEventListener('click', () => {
              setLayerField(entry, fieldIdx);
              setFieldButtonContent(button, name);
              menu.querySelectorAll('.custom-dropdown-item').forEach((el, idx2) => {
                el.classList.toggle('selected', idx2 === fieldIdx);
              });
              closeDropdownMenu(menu);
            });
            menu.appendChild(item);
          });

          button.addEventListener('click', (ev) => {
            ev.stopPropagation();
            toggleDropdownMenu(button, menu);
          });

          dropdown.appendChild(button);
          dropdown.appendChild(menu);
          wrap.appendChild(dropdown);
          info.appendChild(wrap);
        }

        row.appendChild(checkbox);
        row.appendChild(info);
        container.appendChild(row);
      });

      updateColorbars();
      updateInfoPanel();
    }

    document.addEventListener('click', (ev) => {
      if (!ev.target.closest('.custom-dropdown') && !ev.target.closest('.custom-dropdown-menu')) {
        closeAllDropdownMenus();
      }
    });

    payload.layers.forEach((rawLayer) => {
      const layer = normalizeLayerData(rawLayer);
      const built = buildMesh(layer);
      const fieldMetas = layer.field_scalar_meta || null;
      const fieldValues = layer.field_values || null;
      const fieldCmaps = fieldMetas
        ? fieldMetas.map((meta) => (meta && meta.cmap) ? meta.cmap : getFallbackCmap())
        : null;
      const entry = {
        layer,
        mesh: built.mesh,
        edgeLines: built.edgeLines,
        visible: layer.visible !== false,
        activeFieldIdx: Number.isInteger(layer.active_field_idx) ? layer.active_field_idx : 0,
        activeValues: layer.values || null,
        activeCmap: layer.scalar_meta && layer.scalar_meta.cmap ? layer.scalar_meta.cmap : getFallbackCmap(),
        fieldCmaps,
      };
      entry.mesh.visible = entry.visible;
      if (entry.edgeLines) entry.edgeLines.visible = false;

      const fieldNames = layer.field_names || null;
      if (fieldNames && fieldNames.length > 0) {
        setLayerField(entry, Math.max(0, Math.min(entry.activeFieldIdx, fieldNames.length - 1)));
      } else if (fieldValues || layer.values) {
        renderEntryColors(entry, entry.activeFieldIdx);
      }

      layers.push(entry);
    });

    const opacityInput = document.getElementById('mesh-opacity');
    const edgeOverlayInput = document.getElementById('edge-overlay-mode');
    const shareFieldScaleInput = document.getElementById('share-field-scale');
    opacityInput.addEventListener('input', () => {
      const opacity = Number(opacityInput.value);
      layers.forEach((entry) => {
        entry.mesh.material.opacity = opacity;
        entry.mesh.material.transparent = opacity < 0.999;
        entry.mesh.material.depthWrite = opacity >= 0.999;
        entry.mesh.material.needsUpdate = true;
      });
      requestRender();
    });

    edgeOverlayInput.addEventListener('change', refreshEdgeOverlay);
    if (shareFieldScaleInput) {
      shareFieldScaleInput.checked = shareFieldScale;
      shareFieldScaleInput.disabled = !sharedFieldScaleRange;
      shareFieldScaleInput.addEventListener('change', () => {
        setShareFieldScale(shareFieldScaleInput.checked);
      });
    }

    rebuildLayerList();
    refreshEdgeOverlay();

    document.getElementById('btn-theme-mode').addEventListener('click', () => {
      applyThemeMode(!isDayMode);
    });
    const commandsToggleBtn = document.getElementById('btn-toggle-commands');
    const controlsPanel = document.getElementById('controls');
    if (commandsToggleBtn && controlsPanel) {
      const updateCommandsToggleButton = () => {
        commandsToggleBtn.textContent = commandsVisible ? 'Hide Commands' : 'Show Commands';
        commandsToggleBtn.classList.toggle('active', !commandsVisible);
      };
      commandsToggleBtn.addEventListener('click', () => {
        commandsVisible = !commandsVisible;
        controlsPanel.classList.toggle('collapsed', !commandsVisible);
        updateCommandsToggleButton();
      });
      updateCommandsToggleButton();
      controlsPanel.addEventListener('scroll', () => {
        closeAllDropdownMenus();
      }, { passive: true });
    }
    document.getElementById('btn-reset-camera').addEventListener('click', () => {
      fitCamera();
    });
    controls.addEventListener('change', () => {
      requestRender();
      scheduleHoverRefresh();
    });
    renderer.domElement.addEventListener('pointermove', (event) => {
      pointerState.inside = true;
      pointerState.clientX = event.clientX;
      pointerState.clientY = event.clientY;
      scheduleHoverRefresh();
    });
    renderer.domElement.addEventListener('pointerleave', () => {
      pointerState.inside = false;
      hideHoverTooltip();
    });

    document.getElementById('btn-quit-top').addEventListener('click', async () => {
      try {
        await fetch('/_shutdown', { method: 'POST' });
      } catch (_err) {
      }
      window.close();
      window.location.href = 'about:blank';
    });

    window.addEventListener('resize', () => {
      closeAllDropdownMenus();
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      requestRenderAndHover();
    });

    applyThemeMode(true);
    if (layers.length > 0) {
      requestAnimationFrame(() => {
        fitCamera();
        requestRender();
      });
    }
    requestRender();
  </script>
</body>
</html>
"""
        return (
            html.replace("__PAYLOAD__", payload)
            .replace("__TITLE__", page_title)
            .replace("__OPACITY__", f"{self.mesh_opacity:.2f}")
        )


def plot_mesh_viewer(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    init_fig: MeshViewerFigure | None = None,
    vertex_color: np.ndarray | torch.Tensor | str = "#94a3b8",
    scalar_fields: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    show_colorbar: bool = False,
    cmap: str | None = None,
    title: str | None = None,
    label: str = "Mesh",
) -> MeshViewerFigure:
    fig = init_fig if init_fig is not None else MeshViewerFigure()

    if scalar_fields is not None:
        if len(scalar_fields) == 0:
            raise ValueError("scalar_fields cannot be empty")

        verts, raw_vertex_shape, tri, valid_vertices = _prepare_mesh_inputs(vertices, faces)
        prepared_fields: list[tuple[str, np.ndarray, np.ndarray | None, dict[str, Any] | None]] = []
        base_field_name: str | None = None
        base_valid_mask: np.ndarray | None = None

        for field_name, field_values in scalar_fields.items():
            colors, scalar_values, scalar_meta = _prepare_vertex_colors(
                field_values,
                raw_vertex_shape=raw_vertex_shape,
                vertex_count=verts.shape[0],
                show_colorbar=show_colorbar,
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

            prepared_fields.append((field_name, colors, scalar_values, scalar_meta))

        if base_valid_mask is None or base_field_name is None:
            raise ValueError("scalar_fields cannot be empty")

        filtered_verts, filtered_faces = _filter_mesh_geometry(verts, tri, base_valid_mask)
        encoded_fields: list[tuple[str, dict[str, Any]]] = []
        has_valid_vertices = bool(np.any(base_valid_mask))

        for field_name, colors, scalar_values, scalar_meta in prepared_fields:
            filtered_colors = colors[base_valid_mask] if has_valid_vertices else np.zeros((0, 3), dtype=np.float32)
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
                        scalar_meta=scalar_meta,
                        scalar_values=filtered_values,
                    ),
                )
            )

        _, base_layer = encoded_fields[0]

        combined_layer = {
            "label": label,
            "vertices": base_layer["vertices"],
            "faces": base_layer["faces"],
            "colors": base_layer["colors"],
            "vertex_count": base_layer["vertex_count"],
            "face_count": base_layer["face_count"],
            "values": base_layer["values"],
            "field_names": [name for name, _ in encoded_fields],
            "field_colors": [
                None
                if layer_data["values"] is not None and layer_data["scalar_meta"] is not None
                else layer_data["colors"]
                for _, layer_data in encoded_fields
            ],
            "field_values": [layer_data["values"] for _, layer_data in encoded_fields],
            "field_scalar_meta": [layer_data["scalar_meta"] for _, layer_data in encoded_fields],
            "active_field_idx": 0,
            "scalar_meta": base_layer["scalar_meta"],
            "visible": True,
        }
        fig.layers.append(combined_layer)
        if title is not None:
            fig.title = title
        return fig

    return fig.add_mesh(
        vertices=vertices,
        faces=faces,
        vertex_color=vertex_color,
        show_colorbar=show_colorbar,
        cmap=cmap,
        title=title,
        label=label,
    )
