from __future__ import annotations

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

    scalar_meta = None
    scalar_values: np.ndarray | None = None
    raw_colors: np.ndarray | None = None
    if isinstance(point_color, str):
        rgb = np.asarray(mcolors.to_rgb(point_color), dtype=np.float32)
        colors = np.repeat(rgb[None, :], pts.shape[0], axis=0)
        raw_colors = colors
    else:
        color_arr = _to_numpy(point_color)
        if color_arr is None:
            raise ValueError("point_color cannot be None")
        color_arr = np.asarray(color_arr)
        base_shape = raw_point_shape[:-1]
        if list(color_arr.shape) != list(raw_point_shape):
            if color_arr.shape == tuple(base_shape):
                scalar_values = np.asarray(color_arr, dtype=np.float32).reshape(-1)
                color_arr = np.repeat(color_arr[..., None], 3, axis=-1)
            elif color_arr.shape == tuple(base_shape) + (1,):
                scalar_values = np.asarray(color_arr, dtype=np.float32).reshape(-1)
                color_arr = np.repeat(color_arr, 3, axis=-1)
            else:
                raise ValueError(f"Unrecognized point_color shape {color_arr.shape}")
        color_arr = np.asarray(color_arr, dtype=np.float32).reshape(-1, 3)
        if downsampling_factor > 1:
            color_arr = color_arr[::downsampling_factor]
            if scalar_values is not None:
                scalar_values = scalar_values[::downsampling_factor]
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
            finite_scalar = np.isfinite(scalar_vals)
            if np.any(finite_scalar):
                finite_vals = scalar_vals[finite_scalar]
                min_val = float(np.min(finite_vals))
                max_val = float(np.max(finite_vals))
            norm = (scalar_vals - min_val) / max(max_val - min_val, 1e-8)
            norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
            colors = plt.get_cmap(cmap)(norm)[:, :3].astype(np.float32)
        else:
            colors = color_arr

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
        # Drop malformed edges and remap indices after filtering.
        n = int(valid_mask.shape[0])
        in_range = (
            (edg[:, 0] >= 0)
            & (edg[:, 1] >= 0)
            & (edg[:, 0] < n)
            & (edg[:, 1] < n)
        )
        edg = edg[in_range]
        if np.any(valid_mask) and edg.shape[0] > 0:
            remap = np.full((n,), -1, dtype=np.int32)
            remap[np.where(valid_mask)[0]] = np.arange(int(np.sum(valid_mask)), dtype=np.int32)
            keep = valid_mask[edg[:, 0]] & valid_mask[edg[:, 1]]
            edg = edg[keep]
            if edg.shape[0] > 0:
                edg = np.stack([remap[edg[:, 0]], remap[edg[:, 1]]], axis=1)
        else:
            edg = np.zeros((0, 2), dtype=np.int32)

    def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return np.zeros((0, 3), dtype=np.uint8)
        max_color = float(np.nanmax(arr))
        if max_color <= 1.0 + 1e-6:
            return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)

    colors_uint8 = _to_uint8_rgb(colors)
    raw_colors_uint8 = _to_uint8_rgb(raw_colors if raw_colors is not None else colors)

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

    def to_html(self) -> str:
        payload = json.dumps(self._payload())
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="pointcloud-viewer" content="1">
  <title>{escape(self.title or "Point Cloud Viewer")}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600&display=swap');
    :root {{
      --bg-dark: #0b1120;
      --bg-panel: rgba(18, 18, 28, 0.92);
      --bg-canvas: #111827;
      --bg-canvas-solid: #111827;
      --border-color: rgba(100, 120, 255, 0.15);
      --accent-primary: #6366f1;
      --accent-secondary: #22d3ee;
      --text-primary: #e4e4e7;
      --text-secondary: #a1a1aa;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      overflow: hidden;
      font-family: 'JetBrains Mono', monospace;
      background: var(--bg-dark);
      color: var(--text-primary);
    }}
    body.day-mode {{
      --bg-dark: #ffffff;
      --bg-panel: rgba(255, 255, 255, 0.94);
      --bg-canvas: #ffffff;
      --bg-canvas-solid: #ffffff;
      --border-color: rgba(55, 65, 81, 0.2);
      --accent-primary: #1d4ed8;
      --accent-secondary: #0891b2;
      --text-primary: #111827;
      --text-secondary: #374151;
    }}
    #canvas-container {{
      position: fixed;
      inset: 0;
      background: var(--bg-canvas);
    }}
    #top-actions {{
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 30;
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    .top-action-btn {{
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
    }}
    .top-action-btn.text-btn {{
      width: auto;
      min-width: 132px;
      padding: 10px 14px;
      font-size: 11px;
      letter-spacing: 0.2px;
      font-family: 'JetBrains Mono', monospace;
    }}
    .top-action-btn.active {{
      background: rgba(99, 102, 241, 0.25);
      border-color: var(--accent-primary);
      color: var(--accent-primary);
    }}
    .top-action-btn.danger {{
      width: auto;
      min-width: 112px;
      padding: 0 12px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
      border-color: rgba(239, 68, 68, 0.4);
      color: #ef4444;
    }}
    #controls {{
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
    }}
    #controls.collapsed {{
      transform: translateX(calc(-100% - 40px));
      opacity: 0;
      pointer-events: none;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-family: 'Space Grotesk', sans-serif;
      font-size: 18px;
      color: var(--accent-primary);
    }}
    .section {{
      margin-top: 18px;
    }}
    .section-title {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-secondary);
      margin-bottom: 10px;
    }}
    .layer-row {{
      display: grid;
      grid-template-columns: 20px minmax(0, 1fr) auto;
      gap: 8px;
      align-items: start;
      margin-bottom: 10px;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.05);
      background: rgba(255,255,255,0.02);
    }}
    .layer-main {{
      min-width: 0;
    }}
    .layer-label {{
      font-size: 12px;
      line-height: 1.35;
    }}
    .layer-options-toggle {{
      width: 32px;
      min-width: 32px;
      height: 32px;
      padding: 0;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--text-secondary);
      cursor: pointer;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      line-height: 1;
    }}
    .layer-options-toggle:hover {{
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--accent-primary);
      color: var(--accent-primary);
    }}
    .layer-options-toggle[aria-expanded="true"] {{
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--accent-primary);
      color: var(--accent-primary);
    }}
    .layer-controls-body {{
      grid-column: 1 / -1;
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .layer-controls-body.hidden {{
      display: none;
    }}
    .layer-controls-body.disabled {{
      opacity: 0.45;
    }}
    .layer-controls-body.disabled .layer-option-toggle,
    .layer-controls-body.disabled .layer-option-control label,
    .layer-controls-body.disabled .layer-select-wrapper,
    .layer-controls-body.disabled .colorbar,
    .layer-controls-body.disabled input:disabled,
    .layer-controls-body.disabled button:disabled {{
      cursor: not-allowed;
    }}
    .layer-option-grid {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .layer-option-control {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .layer-option-control label {{
      font-size: 12px;
      color: var(--text-secondary);
    }}
    .layer-option-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--text-secondary);
    }}
    .layer-option-toggle input[type="checkbox"] {{
      margin: 0;
    }}
    .layer-select-wrapper {{
      position: relative;
      margin-top: 6px;
    }}
    .layer-field-controls {{
      margin-top: 0;
    }}
    .layer-colorbar-host {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .layer-controls-body .colorbar {{
      margin-top: 0;
    }}
    .field-select-inline {{
      display: grid;
      grid-template-columns: 40px minmax(0, 1fr) 40px;
      gap: 6px;
      align-items: center;
      width: 100%;
    }}
    .custom-dropdown {{
      position: relative;
      width: 100%;
      min-width: 0;
    }}
    .field-nav-button {{
      width: 40px;
      min-width: 40px;
      height: 36px;
      padding: 0;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      font-weight: 500;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--text-primary);
      cursor: pointer;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      line-height: 1;
    }}
    .field-nav-button:hover {{
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--accent-primary);
    }}
    .field-nav-button:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}
    .layer-select-button {{
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
      min-width: 0;
      overflow: hidden;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2322d3ee' d='M6 9L1 4h10z'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 8px center;
      padding-right: 28px;
    }}
    .layer-select-button .layer-id {{
      flex: 1;
      text-align: center;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .layer-select-button:hover {{
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--accent-primary);
    }}
    .layer-select-button:focus {{
      outline: none;
      background: rgba(99, 102, 241, 0.2);
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1);
    }}
    .layer-select-button:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}
    .custom-dropdown-menu {{
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
    }}
    .custom-dropdown-menu.open {{
      display: block;
    }}
    .custom-dropdown-menu.floating {{
      position: fixed;
      top: 0;
      left: 0;
      right: auto;
      width: 240px;
      margin-top: 0;
      max-height: min(420px, calc(100vh - 24px));
      z-index: 1200;
    }}
    .custom-dropdown-item {{
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
    }}
    .custom-dropdown-item .layer-id {{
      flex: 1;
      text-align: center;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .custom-dropdown-item:last-child {{
      border-bottom: none;
    }}
    .custom-dropdown-item:hover {{
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--accent-primary);
    }}
    .custom-dropdown-item.selected {{
      background: rgba(99, 102, 241, 0.25);
      color: var(--accent-primary);
    }}
    .custom-dropdown-item.disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}
    .custom-dropdown-menu::-webkit-scrollbar {{
      width: 8px;
    }}
    .custom-dropdown-menu::-webkit-scrollbar-track {{
      background: rgba(255, 255, 255, 0.03);
      border-radius: 4px;
    }}
    .custom-dropdown-menu::-webkit-scrollbar-thumb {{
      background: rgba(99, 102, 241, 0.3);
      border-radius: 4px;
    }}
    .custom-dropdown-menu::-webkit-scrollbar-thumb:hover {{
      background: rgba(99, 102, 241, 0.5);
    }}
    .control {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 12px;
    }}
    .control label {{
      font-size: 12px;
      color: var(--text-secondary);
    }}
    .control button {{
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
    }}
    .control button:hover {{
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--accent-primary);
    }}
    input[type="range"] {{
      -webkit-appearance: none;
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
      outline: none;
      opacity: 0.8;
      transition: opacity 0.2s;
    }}
    input[type="range"]:hover {{
      opacity: 1;
    }}
    input[type="range"]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--text-primary);
      cursor: pointer;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
      transition: transform 0.2s;
    }}
    input[type="range"]::-webkit-slider-thumb:hover {{
      transform: scale(1.15);
    }}
    input[type="range"]::-moz-range-thumb {{
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--text-primary);
      cursor: pointer;
      border: none;
    }}
    .colorbar {{
      margin-top: 12px;
      padding: 10px;
      border-radius: 10px;
      background: rgba(255,255,255,0.02);
      border: 1px solid rgba(255,255,255,0.05);
    }}
    .colorbar-title {{
      font-size: 11px;
      margin-bottom: 6px;
      color: var(--text-secondary);
    }}
    .colorbar .layer-select-wrapper {{
      margin-left: 28px;
      width: calc(100% - 28px);
      margin-bottom: 12px;
    }}
    .dropdown-label {{
      color: var(--text-secondary);
      font-size: 11px;
      letter-spacing: 0.2px;
      flex: 0 0 72px;
    }}
    .field-dropdown-label {{
      flex: 0 0 auto;
    }}
    .cmap-swatch {{
      width: 30px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.18);
      box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.14);
      flex: 0 0 auto;
    }}
    .cmap-dropdown-prefix {{
      flex: 0 0 72px;
      display: flex;
      align-items: center;
      justify-content: flex-start;
    }}
    .cmap-dropdown-item .layer-id {{
      text-align: left;
    }}
    .cmap-dropdown-item {{
      justify-content: flex-start;
    }}
    .colorbar-visual {{
      display: flex;
      flex-direction: column;
      gap: 2px;
      margin-bottom: 6px;
    }}
    .colorbar-distribution-wrap {{
      position: relative;
      height: 32px;
    }}
    .colorbar-distribution {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
    }}
    .colorbar-distribution-fill {{
      fill: var(--accent-secondary);
      opacity: 0.18;
    }}
    .colorbar-distribution-line {{
      fill: none;
      stroke: var(--accent-secondary);
      stroke-width: 1.4;
      stroke-linecap: butt;
      stroke-linejoin: round;
      vector-effect: non-scaling-stroke;
    }}
    .colorbar-distribution-fill.faded {{
      opacity: 0.07;
    }}
    .colorbar-distribution-line.faded {{
      opacity: 0.28;
    }}
    .colorbar-gradient-wrap {{
      position: relative;
      height: 12px;
    }}
    .colorbar-gradient {{
      position: absolute;
      inset: 0;
      border-radius: 999px;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
    }}
    .colorbar-range {{
      display: flex;
      justify-content: space-between;
      font-size: 11px;
    }}
    .colorbar-clip {{
      display: flex;
      flex-direction: column;
      gap: 0;
    }}
    .colorbar-clip-slider {{
      position: relative;
      height: 14px;
    }}
    .colorbar-clip-track,
    .colorbar-clip-fill {{
      position: absolute;
      left: 0;
      right: 0;
      top: 50%;
      height: 6px;
      border-radius: 3px;
      transform: translateY(-50%);
    }}
    .colorbar-clip-track {{
      background: rgba(255, 255, 255, 0.08);
    }}
    .colorbar-clip-fill {{
      left: 0;
      right: auto;
      width: 100%;
      background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
      opacity: 0.8;
    }}
    .colorbar-clip-slider input[type="range"] {{
      position: absolute;
      inset: 0;
      width: 100%;
      margin: 0;
      background: none;
      opacity: 1;
      pointer-events: none;
      z-index: 1;
    }}
    .colorbar-clip-slider input[type="range"]::-webkit-slider-runnable-track {{
      height: 6px;
      background: transparent;
      border: none;
    }}
    .colorbar-clip-slider input[type="range"]::-moz-range-track {{
      height: 6px;
      background: transparent;
      border: none;
    }}
    .colorbar-clip-slider input[type="range"]::-moz-range-progress {{
      background: transparent;
    }}
    .colorbar-clip-slider input[type="range"]::-webkit-slider-thumb {{
      pointer-events: auto;
      width: 14px;
      height: 14px;
    }}
    .colorbar-clip-slider input[type="range"]::-moz-range-thumb {{
      pointer-events: auto;
      width: 14px;
      height: 14px;
    }}
    #info-panel {{
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
    }}
    #info-panel .info-row {{
      display: flex;
      gap: 20px;
    }}
    #info-panel .info-item {{
      display: flex;
      gap: 6px;
    }}
    #info-panel .info-label {{
      color: var(--text-secondary);
    }}
    #info-panel .info-value {{
      color: var(--accent-secondary);
      font-weight: 500;
    }}
    #hover-tooltip {{
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
    }}
    .hover-tooltip-title {{
      margin-bottom: 6px;
      font-weight: 600;
    }}
    .hover-tooltip-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      white-space: normal;
    }}
    .hover-tooltip-label {{
      min-width: 72px;
      color: var(--text-secondary);
    }}
    .hover-tooltip-value {{
      min-width: 0;
      flex: 1 1 auto;
      word-break: break-word;
    }}
    .hover-tooltip-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.18);
      box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.14);
      flex: 0 0 auto;
    }}
  </style>
</head>
<body>
  <div id="canvas-container"></div>
  <div id="top-actions">
    <button id="btn-quit-top" class="top-action-btn danger" title="Quit viewer" aria-label="Quit viewer">Quit Viewer</button>
    <button id="btn-toggle-commands" class="top-action-btn text-btn" title="Hide commands" aria-label="Hide commands">Hide Commands</button>
    <button id="btn-theme-mode" class="top-action-btn" title="Switch to day mode" aria-label="Switch to day mode">☾</button>
  </div>
  <div id="controls">
    <h1>{escape(self.title or "Point Cloud Viewer")}</h1>
    <div class="section">
      <div class="section-title">Display</div>
      <div class="control">
        <button id="btn-reset-camera" type="button">Reset camera</button>
      </div>
    </div>
    <div class="section">
      <div class="section-title">Layers</div>
      <div id="layer-list"></div>
    </div>
  </div>
  <div id="info-panel">
    <div class="info-row">
      <div class="info-item">
        <span class="info-label">Points:</span>
        <span class="info-value" id="info-points">--</span>
      </div>
      <div class="info-item">
        <span class="info-label">Vertices:</span>
        <span class="info-value" id="info-vertices">--</span>
      </div>
      <div class="info-item">
        <span class="info-label">Faces:</span>
        <span class="info-value" id="info-faces">--</span>
      </div>
    </div>
  </div>
  <div id="hover-tooltip"></div>
  <div id="dropdown-portal"></div>

  <script type="importmap">
    {{
      "imports": {{
        "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
      }}
    }}
  </script>
  <script type="module">
    import * as THREE from 'three';
    import {{ TrackballControls }} from 'three/addons/controls/TrackballControls.js';
    import {{ LineMaterial }} from 'three/addons/lines/LineMaterial.js';
    import {{ LineSegments2 }} from 'three/addons/lines/LineSegments2.js';
    import {{ LineSegmentsGeometry }} from 'three/addons/lines/LineSegmentsGeometry.js';

    const payload = {payload};
    const colormapPresets = payload.colormap_presets || {{}};
    const availableCmaps = payload.available_cmaps || Object.keys(colormapPresets);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.001, 1e7);
    camera.position.set(0, 0, 4);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x111827, 1.0);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    const controls = new TrackballControls(camera, renderer.domElement);
    controls.rotateSpeed = 2.0;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.8;
    controls.dynamicDampingFactor = 0.15;
    controls.staticMoving = false;

    const layers = [];
    const raycaster = new THREE.Raycaster();
    const pointerNdc = new THREE.Vector2();
    const hoverTooltip = document.getElementById('hover-tooltip');
    const dropdownPortal = document.getElementById('dropdown-portal');
    let isDayMode = false;
    let commandsVisible = true;
    const CLIP_SLIDER_MAX = 1000;
    const vertexShader = `
      attribute vec3 color;
      varying vec3 vColor;
      uniform float size;
      void main() {{
        vColor = color;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size;
        gl_Position = projectionMatrix * mvPosition;
      }}
    `;

    const fragmentShader = `
      varying vec3 vColor;
      uniform float opacity;
      void main() {{
        vec2 center = gl_PointCoord - vec2(0.5);
        if (length(center) > 0.5) discard;
        gl_FragColor = vec4(vColor, opacity);
      }}
    `;

    function createStoredColorAttribute(colors) {{
      return new THREE.BufferAttribute(colors, 3, true);
    }}

    function flattenColorArray(colors) {{
      if (!Array.isArray(colors) || colors.length === 0) return new Uint8Array(0);
      const flat = new Uint8Array(colors.length * 3);
      for (let idx = 0; idx < colors.length; idx += 1) {{
        const row = colors[idx];
        flat[idx * 3] = Number(row?.[0] ?? 0);
        flat[idx * 3 + 1] = Number(row?.[1] ?? 0);
        flat[idx * 3 + 2] = Number(row?.[2] ?? 0);
      }}
      return flat;
    }}

    function setStoredColorAttribute(geometry, colors) {{
      const colorArray = colors instanceof Uint8Array ? colors : flattenColorArray(colors);
      const existing = geometry.getAttribute('color');
      if (
        existing
        && existing.array instanceof Uint8Array
        && existing.array.length === colorArray.length
        && existing.normalized
      ) {{
        existing.array.set(colorArray);
        existing.needsUpdate = true;
        return existing;
      }}
      const attr = createStoredColorAttribute(colorArray);
      geometry.setAttribute('color', attr);
      return attr;
    }}

    function collectScalarMetas(layer) {{
      const metas = [];
      if (layer && layer.scalar_meta) metas.push(layer.scalar_meta);
      if (layer && Array.isArray(layer.field_scalar_meta)) {{
        layer.field_scalar_meta.forEach((meta) => {{
          if (meta) metas.push(meta);
        }});
      }}
      return metas;
    }}

    function clampUnit(value, fallback = 0.0) {{
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return fallback;
      return Math.min(1.0, Math.max(0.0, numeric));
    }}

    function createClipState(minFrac = 0.0, maxFrac = 1.0) {{
      const start = clampUnit(minFrac, 0.0);
      const end = clampUnit(maxFrac, 1.0);
      return {{
        minFrac: Math.min(start, end),
        maxFrac: Math.max(start, end),
      }};
    }}

    function getEntryFieldCount(entry) {{
      return Math.max(
        Array.isArray(entry.layer.field_names) ? entry.layer.field_names.length : 0,
        Array.isArray(entry.layer.field_values) ? entry.layer.field_values.length : 0,
        Array.isArray(entry.layer.field_scalar_meta) ? entry.layer.field_scalar_meta.length : 0,
        Array.isArray(entry.layer.field_colors) ? entry.layer.field_colors.length : 0,
        1,
      );
    }}

    function ensureFieldClipStates(entry) {{
      const fieldCount = getEntryFieldCount(entry);
      if (!Array.isArray(entry.fieldClipStates)) {{
        entry.fieldClipStates = [];
      }}
      while (entry.fieldClipStates.length < fieldCount) {{
        entry.fieldClipStates.push(createClipState());
      }}
      return entry.fieldClipStates;
    }}

    function getClipState(entry, fieldIdx = entry.activeFieldIdx ?? 0) {{
      if (entry.shareFieldScale) {{
        if (!entry.sharedFieldClip) {{
          entry.sharedFieldClip = createClipState();
        }}
        return entry.sharedFieldClip;
      }}
      const states = ensureFieldClipStates(entry);
      const clampedIdx = Math.max(0, Math.min(states.length - 1, fieldIdx));
      if (!states[clampedIdx]) {{
        states[clampedIdx] = createClipState();
      }}
      return states[clampedIdx];
    }}

    function sliderValueToClip(value, fallback = 0.0) {{
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return fallback;
      return clampUnit(numeric / CLIP_SLIDER_MAX, fallback);
    }}

    function clipToSliderValue(value, fallback = 0.0) {{
      return Math.round(clampUnit(value, fallback) * CLIP_SLIDER_MAX);
    }}

    function setClipBoundary(state, boundary, nextFrac) {{
      if (!state) return;
      const clamped = clampUnit(nextFrac, boundary === 'max' ? 1.0 : 0.0);
      if (boundary === 'max') {{
        state.maxFrac = Math.max(clamped, state.minFrac);
      }} else {{
        state.minFrac = Math.min(clamped, state.maxFrac);
      }}
    }}

    function applyClipToRange(range, clipState) {{
      if (!range) return null;
      const minVal = Number(range.min);
      const maxVal = Number(range.max);
      if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null;
      const span = maxVal - minVal;
      if (!(Math.abs(span) > 1e-12)) {{
        return {{ min: minVal, max: maxVal }};
      }}
      const startFrac = clampUnit(clipState && clipState.minFrac, 0.0);
      const endFrac = clampUnit(clipState && clipState.maxFrac, 1.0);
      return {{
        min: minVal + span * Math.min(startFrac, endFrac),
        max: minVal + span * Math.max(startFrac, endFrac),
      }};
    }}

    function formatScalarValue(value) {{
      return Number.isFinite(value) ? Number(value).toFixed(4) : 'n/a';
    }}

    function syncAllColorbarClipControls() {{
      document.querySelectorAll('.colorbar').forEach((node) => {{
        if (typeof node.__syncClipControl === 'function') {{
          node.__syncClipControl();
        }}
      }});
    }}

    const NO_CMAP = 'none';

    function getFallbackCmap() {{
      return availableCmaps[0] || 'viridis';
    }}

    function isNoCmap(cmapName) {{
      return !cmapName || cmapName === NO_CMAP;
    }}

    function getColormapPreset(cmapName) {{
      if (cmapName && colormapPresets[cmapName]) return colormapPresets[cmapName];
      const fallbackName = getFallbackCmap();
      return colormapPresets[fallbackName] || null;
    }}

    function gradientCss(gradient) {{
      if (!Array.isArray(gradient) || gradient.length === 0) {{
        return 'linear-gradient(90deg, #64748b, #e2e8f0)';
      }}
      return `linear-gradient(90deg, ${{gradient.join(', ')}})`;
    }}

    function getEntryDisplayedValues(entry, fieldIdx = entry.activeFieldIdx ?? 0) {{
      const fieldValues = entry.layer.field_values || null;
      return fieldValues ? fieldValues[fieldIdx] : entry.layer.values || null;
    }}

    function computeFiniteValueRange(values) {{
      if (!values || typeof values.length !== 'number' || values.length === 0) return null;
      let minVal = Infinity;
      let maxVal = -Infinity;
      for (let idx = 0; idx < values.length; idx += 1) {{
        const value = Number(values[idx]);
        if (!Number.isFinite(value)) continue;
        minVal = Math.min(minVal, value);
        maxVal = Math.max(maxVal, value);
      }}
      if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null;
      return {{ min: minVal, max: maxVal }};
    }}

    function sampleRawGradientFromColors(colors, stepCount = 7) {{
      const flat = colors instanceof Uint8Array ? colors : flattenColorArray(colors);
      const colorCount = Math.floor(flat.length / 3);
      if (!flat || colorCount === 0) return null;
      const gradient = [];
      for (let idx = 0; idx < stepCount; idx += 1) {{
        const sampleIdx = stepCount <= 1 || colorCount <= 1
          ? 0
          : Math.round((idx * (colorCount - 1)) / (stepCount - 1));
        gradient.push(`rgb(${{flat[sampleIdx * 3]}}, ${{flat[sampleIdx * 3 + 1]}}, ${{flat[sampleIdx * 3 + 2]}})`);
      }}
      return gradient;
    }}

    function buildRawScalarGradient(range, stepCount = 7) {{
      if (!range) return null;
      const useUnitScale = range.max <= 1.0 + 1e-6;
      const gradient = [];
      for (let idx = 0; idx < stepCount; idx += 1) {{
        const t = stepCount <= 1 ? 0.0 : idx / (stepCount - 1);
        const raw = range.min + (range.max - range.min) * t;
        const intensity = useUnitScale ? raw * 255.0 : raw;
        const clipped = Math.max(0, Math.min(255, Math.round(intensity)));
        gradient.push(`rgb(${{clipped}}, ${{clipped}}, ${{clipped}})`);
      }}
      return gradient;
    }}

    function getEntryRawColors(entry, fieldIdx = entry.activeFieldIdx ?? 0) {{
      const fieldColors = entry.layer.field_colors || null;
      if (fieldColors && fieldIdx >= 0 && fieldIdx < fieldColors.length && fieldColors[fieldIdx]) {{
        return fieldColors[fieldIdx];
      }}
      return entry.layer.raw_colors || entry.layer.colors || null;
    }}

    function getEntryColorModeOptions(entry, fieldIdx = entry.activeFieldIdx ?? 0) {{
      const options = [NO_CMAP];
      const values = getEntryDisplayedValues(entry, fieldIdx);
      if (values && typeof values.length === 'number') {{
        availableCmaps.forEach((cmapName) => options.push(cmapName));
      }}
      return Array.from(new Set(options));
    }}

    function getEntryRawGradient(entry, fieldIdx = entry.activeFieldIdx ?? 0) {{
      const values = getEntryDisplayedValues(entry, fieldIdx);
      const valueRange = computeFiniteValueRange(values);
      if (valueRange) return buildRawScalarGradient(valueRange);
      return sampleRawGradientFromColors(getEntryRawColors(entry, fieldIdx));
    }}

    function createCmapSwatch(entry, fieldIdx, cmapName) {{
      const swatch = document.createElement('span');
      swatch.className = 'cmap-swatch';
      const gradient = isNoCmap(cmapName)
        ? getEntryRawGradient(entry, fieldIdx)
        : (getColormapPreset(cmapName)?.gradient || null);
      swatch.style.background = gradientCss(gradient);
      return swatch;
    }}

    function setCmapButtonContent(button, cmapName) {{
      button.replaceChildren();
      const prefix = document.createElement('span');
      prefix.className = 'dropdown-label';
      prefix.textContent = 'Colormap:';
      const value = document.createElement('span');
      value.className = 'layer-id';
      value.textContent = isNoCmap(cmapName) ? NO_CMAP : cmapName;
      button.appendChild(prefix);
      button.appendChild(value);
    }}

    function setFieldButtonContent(button, fieldName) {{
      button.replaceChildren();
      const prefix = document.createElement('span');
      prefix.className = 'dropdown-label field-dropdown-label';
      prefix.textContent = 'Field:';
      const value = document.createElement('span');
      value.className = 'layer-id';
      value.textContent = fieldName;
      button.appendChild(prefix);
      button.appendChild(value);
    }}

    function closeDropdownMenu(menu) {{
      if (dropdownPortal && menu.parentElement === dropdownPortal) {{
        const host = menu.__hostElement || null;
        if (host) {{
          host.appendChild(menu);
        }}
      }}
      menu.classList.remove('floating');
      menu.classList.remove('open');
      menu.style.top = '';
      menu.style.left = '';
      menu.style.width = '';
      menu.style.maxHeight = '';
    }}

    function closeAllDropdownMenus(exceptMenu = null) {{
      document.querySelectorAll('.custom-dropdown-menu.open').forEach((node) => {{
        if (node !== exceptMenu) closeDropdownMenu(node);
      }});
    }}

    function positionDropdownMenu(button, menu) {{
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

      menu.style.left = `${{left}}px`;
      menu.style.width = `${{menuWidth}}px`;
      menu.style.maxHeight = `${{maxHeight}}px`;

      if (openUpward) {{
        const menuHeight = Math.min(menu.scrollHeight || maxHeight, maxHeight);
        menu.style.top = `${{Math.max(margin, rect.top - menuHeight - gap)}}px`;
      }} else {{
        menu.style.top = `${{Math.min(
          window.innerHeight - maxHeight - margin,
          Math.max(margin, rect.bottom + gap),
        )}}px`;
      }}
    }}

    function toggleDropdownMenu(button, menu) {{
      const willOpen = !menu.classList.contains('open');
      closeAllDropdownMenus(willOpen ? menu : null);
      if (!willOpen) {{
        closeDropdownMenu(menu);
        return;
      }}
      if (dropdownPortal && menu.parentElement !== dropdownPortal) {{
        menu.__hostElement = menu.parentElement;
        dropdownPortal.appendChild(menu);
      }}
      menu.classList.add('floating');
      menu.classList.add('open');
      positionDropdownMenu(button, menu);
    }}

    function getLayerOpacity(layer) {{
      const value = Number(layer?.opacity ?? payload.opacity);
      return Number.isFinite(value) ? Math.min(1.0, Math.max(0.05, value)) : 1.0;
    }}

    function getLayerPointSize(layer) {{
      const value = Number(layer?.point_size ?? payload.point_size);
      return Number.isFinite(value) ? Math.min(12.0, Math.max(1.0, value)) : payload.point_size;
    }}

    function getLayerEdgeThickness(layer) {{
      const value = Number(layer?.edge_thickness ?? 1.0);
      return Number.isFinite(value) ? Math.min(12.0, Math.max(1.0, value)) : 1.0;
    }}

    function updateEntryPointSize(entry, size) {{
      const numeric = Number(size);
      const clamped = Number.isFinite(numeric) ? Math.min(12.0, Math.max(1.0, numeric)) : getLayerPointSize(entry.layer);
      entry.pointSize = clamped;
      entry.objects.points.material.uniforms.size.value = clamped;
    }}

    function updateEntryOpacity(entry, opacity) {{
      const nextOpacity = Math.min(1.0, Math.max(0.05, Number(opacity) || 1.0));
      entry.opacity = nextOpacity;
      entry.objects.points.material.uniforms.opacity.value = nextOpacity;
      if (entry.objects.lines) {{
        entry.objects.lines.material.opacity = (isDayMode ? 0.28 : 0.35) * nextOpacity;
      }}
    }}

    function updateEntryEdgeThickness(entry, thickness) {{
      const nextThickness = Math.min(12.0, Math.max(1.0, Number(thickness) || 1.0));
      entry.edgeThickness = nextThickness;
      if (entry.objects.lines) {{
        entry.objects.lines.material.linewidth = nextThickness;
      }}
    }}

    function applyThemeMode(dayMode) {{
      isDayMode = Boolean(dayMode);
      document.body.classList.toggle('day-mode', isDayMode);
      const css = getComputedStyle(document.body);
      const bg = css.getPropertyValue('--bg-canvas-solid').trim() || '#111827';
      renderer.setClearColor(bg, 1.0);
      const btn = document.getElementById('btn-theme-mode');
      if (btn) {{
        btn.textContent = isDayMode ? '☀' : '☾';
        btn.title = isDayMode ? 'Switch to night mode' : 'Switch to day mode';
        btn.setAttribute('aria-label', btn.title);
      }}
      const lineColor = isDayMode ? 0x0f172a : 0xe4e4e7;
      for (const entry of layers) {{
        if (!entry.objects.lines) continue;
        entry.objects.lines.material.color.setHex(lineColor);
        entry.objects.lines.material.opacity = (isDayMode ? 0.28 : 0.35) * entry.opacity;
      }}
    }}

    function setLineMaterialResolution(material) {{
      if (!material || !material.resolution) return;
      material.resolution.set(Math.max(1, window.innerWidth), Math.max(1, window.innerHeight));
    }}

    function refreshLineMaterialResolutions() {{
      for (const entry of layers) {{
        const material = entry.objects?.lines?.material;
        setLineMaterialResolution(material);
      }}
    }}

    function buildPoints(layer) {{
      const baseOpacity = getLayerOpacity(layer);
      const baseEdgeThickness = getLayerEdgeThickness(layer);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(layer.points.flat(), 3));
      geometry.setAttribute('color', createStoredColorAttribute(flattenColorArray(layer.colors)));
      const material = new THREE.ShaderMaterial({{
        uniforms: {{
          size: {{ value: payload.point_size }},
          opacity: {{ value: baseOpacity }},
        }},
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        transparent: true,
      }});
      const points = new THREE.Points(geometry, material);
      points.frustumCulled = false;
      scene.add(points);

      let lines = null;
      if (layer.edges) {{
        const positions = [];
        for (const [i, j] of layer.edges) {{
          const pi = layer.points[i];
          const pj = layer.points[j];
          positions.push(...pi, ...pj);
        }}
        const lineGeometry = new LineSegmentsGeometry();
        lineGeometry.setPositions(positions);
        const lineMaterial = new LineMaterial({{
          color: isDayMode ? 0x0f172a : 0xe4e4e7,
          opacity: (isDayMode ? 0.28 : 0.35) * baseOpacity,
          transparent: true,
          linewidth: baseEdgeThickness,
        }});
        setLineMaterialResolution(lineMaterial);
        lines = new LineSegments2(lineGeometry, lineMaterial);
        lines.frustumCulled = false;
        scene.add(lines);
      }}
      return {{ points, lines }};
    }}

    function fitCamera() {{
      const box = new THREE.Box3();
      let hasBox = false;
      for (const entry of layers) {{
        if (!entry.visible) continue;
        box.expandByObject(entry.objects.points);
        hasBox = true;
      }}
      if (!hasBox || box.isEmpty()) return;
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z) * 0.5 || 1.0;

      // Fit a bounding sphere in view for arbitrary world scales.
      const fov = THREE.MathUtils.degToRad(camera.fov);
      const distance = (radius / Math.sin(fov / 2)) * 1.2;
      camera.position.set(center.x + distance, center.y + distance, center.z + distance);

      // Update clipping planes from scene scale to avoid clipping large clouds.
      const near = Math.max(distance / 5000, 1e-6);
      const far = Math.max(distance * 50, 10.0);
      camera.near = near;
      camera.far = far;
      camera.updateProjectionMatrix();

      controls.target.copy(center);
      controls.update();
    }}

    function getEntryCmap(entry, fieldIdx = entry.activeFieldIdx ?? 0) {{
      return entry.activeCmap || NO_CMAP;
    }}

    function getEntryScaleRange(entry, fieldIdx = entry.activeFieldIdx ?? 0) {{
      const values = getEntryDisplayedValues(entry, fieldIdx);
      const fieldMetas = entry.layer.field_scalar_meta || null;
      const baseMeta = fieldMetas ? fieldMetas[fieldIdx] : entry.layer.scalar_meta;
      const metaMin = Number(baseMeta && baseMeta.min);
      const metaMax = Number(baseMeta && baseMeta.max);
      if (Number.isFinite(metaMin) && Number.isFinite(metaMax)) {{
        return {{ min: metaMin, max: metaMax }};
      }}
      return computeFiniteValueRange(values);
    }}

    function getEntryAllScaleRanges(entry) {{
      const ranges = [];
      const fieldValues = entry.layer.field_values || null;
      const fieldMetas = entry.layer.field_scalar_meta || null;
      if (fieldValues && fieldValues.length > 0) {{
        for (let fieldIdx = 0; fieldIdx < fieldValues.length; fieldIdx += 1) {{
          const values = fieldValues[fieldIdx];
          const meta = fieldMetas ? fieldMetas[fieldIdx] : null;
          const metaMin = Number(meta && meta.min);
          const metaMax = Number(meta && meta.max);
          if (Number.isFinite(metaMin) && Number.isFinite(metaMax)) {{
            ranges.push({{ min: metaMin, max: metaMax }});
            continue;
          }}
          const range = computeFiniteValueRange(values);
          if (range) ranges.push(range);
        }}
        return ranges;
      }}
      const range = getEntryScaleRange(entry);
      return range ? [range] : [];
    }}

    function getSharedFieldScaleRange(entry) {{
      if (!entry || isNoCmap(getEntryCmap(entry))) return null;
      let minVal = Infinity;
      let maxVal = -Infinity;
      for (const range of getEntryAllScaleRanges(entry)) {{
        minVal = Math.min(minVal, range.min);
        maxVal = Math.max(maxVal, range.max);
      }}
      if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null;
      return {{ min: minVal, max: maxVal }};
    }}

    function getScalarMeta(entry, fieldIdx = entry.activeFieldIdx ?? 0) {{
      const values = getEntryDisplayedValues(entry, fieldIdx);
      const fieldMetas = entry.layer.field_scalar_meta || null;
      const baseMeta = fieldMetas ? fieldMetas[fieldIdx] : entry.layer.scalar_meta;
      const cmapName = getEntryCmap(entry, fieldIdx);
      if (!baseMeta) {{
        const range = computeFiniteValueRange(values);
        if (!range) return null;
        const resolved = {{
          min: range.min,
          max: range.max,
          cmap: cmapName,
          gradient: isNoCmap(cmapName)
            ? getEntryRawGradient(entry, fieldIdx)
            : (getColormapPreset(cmapName)?.gradient || null),
        }};
        const sharedFieldScaleRange = getSharedFieldScaleRange(entry);
        if (!isNoCmap(cmapName) && entry.shareFieldScale && sharedFieldScaleRange) {{
          resolved.min = sharedFieldScaleRange.min;
          resolved.max = sharedFieldScaleRange.max;
        }}
        resolved.fullMin = resolved.min;
        resolved.fullMax = resolved.max;
        if (!isNoCmap(cmapName)) {{
          const clippedRange = applyClipToRange(
            {{ min: resolved.fullMin, max: resolved.fullMax }},
            getClipState(entry, fieldIdx),
          );
          if (clippedRange) {{
            resolved.min = clippedRange.min;
            resolved.max = clippedRange.max;
          }}
        }}
        resolved.clipState = createClipState(
          getClipState(entry, fieldIdx).minFrac,
          getClipState(entry, fieldIdx).maxFrac,
        );
        resolved.clipEnabled = !isNoCmap(cmapName) && Math.abs(resolved.fullMax - resolved.fullMin) > 1e-12;
        return resolved;
      }}
      const resolved = {{
        ...baseMeta,
        cmap: cmapName,
        gradient: isNoCmap(cmapName)
          ? (baseMeta.gradient || getEntryRawGradient(entry, fieldIdx))
          : (getColormapPreset(cmapName)?.gradient || baseMeta.gradient),
      }};
      const sharedFieldScaleRange = getSharedFieldScaleRange(entry);
      if (!isNoCmap(cmapName) && entry.shareFieldScale && sharedFieldScaleRange) {{
        resolved.min = sharedFieldScaleRange.min;
        resolved.max = sharedFieldScaleRange.max;
      }}
      resolved.fullMin = resolved.min;
      resolved.fullMax = resolved.max;
      if (!isNoCmap(cmapName)) {{
        const clippedRange = applyClipToRange(
          {{ min: resolved.fullMin, max: resolved.fullMax }},
          getClipState(entry, fieldIdx),
        );
        if (clippedRange) {{
          resolved.min = clippedRange.min;
          resolved.max = clippedRange.max;
        }}
      }}
      resolved.clipState = createClipState(
        getClipState(entry, fieldIdx).minFrac,
        getClipState(entry, fieldIdx).maxFrac,
      );
      resolved.clipEnabled = !isNoCmap(cmapName) && Math.abs(resolved.fullMax - resolved.fullMin) > 1e-12;
      return resolved;
    }}

    function buildDistributionSvg(values, meta) {{
      if (!values || typeof values.length !== 'number' || values.length === 0 || !meta) return null;

      const minVal = Number(meta.min);
      const maxVal = Number(meta.max);
      if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null;

      const bins = 48;
      const counts = new Float32Array(bins);
      const span = maxVal - minVal;
      const hasSpan = Number.isFinite(span) && Math.abs(span) > 1e-12;
      let finiteCount = 0;

      for (let idx = 0; idx < values.length; idx += 1) {{
        const raw = Number(values[idx]);
        if (!Number.isFinite(raw)) continue;
        finiteCount += 1;

        let binIdx = 0;
        if (hasSpan) {{
          const normalized = Math.min(1, Math.max(0, (raw - minVal) / span));
          binIdx = Math.min(bins - 1, Math.floor(normalized * bins));
        }} else {{
          binIdx = Math.floor((bins - 1) * 0.5);
        }}
        counts[binIdx] += 1;
      }}

      if (finiteCount === 0) return null;

      let peak = 0;
      for (let idx = 0; idx < bins; idx += 1) {{
        peak = Math.max(peak, counts[idx]);
      }}
      if (!(peak > 0)) return null;

      const width = 100;
      const height = 32;
      const baselineY = 30;
      const curveTopY = 3;
      const curveHeight = baselineY - curveTopY;
      const strokeInset = 0.7;
      const xStep = (width - 2 * strokeInset) / (bins - 1);

      let linePath = '';
      for (let idx = 0; idx < bins; idx += 1) {{
        const x = strokeInset + idx * xStep;
        const y = baselineY - (counts[idx] / peak) * curveHeight;
        linePath += `${{idx === 0 ? 'M' : 'L'}} ${{x.toFixed(2)}} ${{y.toFixed(2)}} `;
      }}

      const areaPath = `${{linePath}}L ${{(width - strokeInset).toFixed(2)}} ${{baselineY}} L ${{strokeInset.toFixed(2)}} ${{baselineY}} Z`;
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
      svg.setAttribute('preserveAspectRatio', 'none');
      svg.classList.add('colorbar-distribution');

      const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
      const clipIdBase = `dist-clip-${{Math.random().toString(36).slice(2)}}`;

      const centerClip = document.createElementNS('http://www.w3.org/2000/svg', 'clipPath');
      centerClip.setAttribute('id', `${{clipIdBase}}-center`);
      const centerRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      centerRect.setAttribute('y', '0');
      centerRect.setAttribute('height', String(height));
      centerClip.appendChild(centerRect);

      const leftClip = document.createElementNS('http://www.w3.org/2000/svg', 'clipPath');
      leftClip.setAttribute('id', `${{clipIdBase}}-left`);
      const leftRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      leftRect.setAttribute('x', '0');
      leftRect.setAttribute('y', '0');
      leftRect.setAttribute('height', String(height));
      leftClip.appendChild(leftRect);

      const rightClip = document.createElementNS('http://www.w3.org/2000/svg', 'clipPath');
      rightClip.setAttribute('id', `${{clipIdBase}}-right`);
      const rightRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rightRect.setAttribute('y', '0');
      rightRect.setAttribute('height', String(height));
      rightClip.appendChild(rightRect);

      defs.appendChild(centerClip);
      defs.appendChild(leftClip);
      defs.appendChild(rightClip);
      svg.appendChild(defs);

      const buildGroup = (fillClass, lineClass, clipId) => {{
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('clip-path', `url(#${{clipId}})`);

        const fillPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        fillPath.setAttribute('d', areaPath.trim());
        fillPath.setAttribute('class', fillClass);

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        line.setAttribute('d', linePath.trim());
        line.setAttribute('class', lineClass);

        group.appendChild(fillPath);
        group.appendChild(line);
        return group;
      }};

      svg.appendChild(buildGroup('colorbar-distribution-fill faded', 'colorbar-distribution-line faded', `${{clipIdBase}}-left`));
      svg.appendChild(buildGroup('colorbar-distribution-fill', 'colorbar-distribution-line', `${{clipIdBase}}-center`));
      svg.appendChild(buildGroup('colorbar-distribution-fill faded', 'colorbar-distribution-line faded', `${{clipIdBase}}-right`));

      const setClipFractions = (startFrac = 0.0, endFrac = 1.0) => {{
        const start = Math.min(1.0, Math.max(0.0, Number(startFrac) || 0.0));
        const end = Math.min(1.0, Math.max(start, Number(endFrac) || 0.0));
        const leftWidth = width * start;
        const centerX = leftWidth;
        const centerWidth = width * Math.max(0.0, end - start);
        const rightX = width * end;
        const rightWidth = Math.max(0.0, width - rightX);

        leftRect.setAttribute('width', leftWidth.toFixed(2));
        centerRect.setAttribute('x', centerX.toFixed(2));
        centerRect.setAttribute('width', centerWidth.toFixed(2));
        rightRect.setAttribute('x', rightX.toFixed(2));
        rightRect.setAttribute('width', rightWidth.toFixed(2));
      }};

      setClipFractions(0.0, 1.0);
      return {{ svg, setClipFractions }};
    }}

    function applyStoredColors(entry, colors) {{
      setStoredColorAttribute(entry.objects.points.geometry, colors);
    }}

    function applyColorsFromValues(entry, values, meta) {{
      const preset = getColormapPreset(meta && meta.cmap);
      const samples = preset && Array.isArray(preset.samples) ? preset.samples : null;
      if (!samples || samples.length === 0 || !values || typeof values.length !== 'number') return false;

      const colorCount = values.length * 3;
      const existing = entry.objects.points.geometry.getAttribute('color');
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

      for (let idx = 0; idx < values.length; idx += 1) {{
        const raw = Number(values[idx]);
        const normalized = Number.isFinite(raw) ? Math.min(1, Math.max(0, (raw - minVal) / denom)) : 0.0;
        const scaled = normalized * sampleMaxIdx;
        const leftIdx = Math.floor(scaled);
        const rightIdx = Math.min(sampleMaxIdx, leftIdx + 1);
        const mix = scaled - leftIdx;
        const left = samples[leftIdx];
        const right = samples[rightIdx];
        colors[idx * 3] = Math.round(left[0] + (right[0] - left[0]) * mix);
        colors[idx * 3 + 1] = Math.round(left[1] + (right[1] - left[1]) * mix);
        colors[idx * 3 + 2] = Math.round(left[2] + (right[2] - left[2]) * mix);
      }}

      if (
        existing
        && existing.array instanceof Uint8Array
        && existing.array.length === colorCount
        && existing.normalized
      ) {{
        existing.needsUpdate = true;
        return true;
      }}

      entry.objects.points.geometry.setAttribute('color', createStoredColorAttribute(colors));
      return true;
    }}

    function createEntryFieldPanel(entry) {{
      const fieldIdx = entry.activeFieldIdx ?? 0;
      const fieldNames = entry.layer.field_names || null;
      const meta = getScalarMeta(entry, fieldIdx);
      if (!meta) return null;
      const fieldLabel = fieldNames ? fieldNames[fieldIdx] : null;
      const colorbarLabel = fieldLabel ? `Field: ${{fieldLabel}}` : 'Field colors';
      const controlsEnabled = entry.visible;
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
      button.disabled = !controlsEnabled;
      const currentCmap = getEntryCmap(entry, fieldIdx);
      setCmapButtonContent(button, currentCmap);
      const menu = document.createElement('div');
      menu.className = 'custom-dropdown-menu';
      getEntryColorModeOptions(entry, fieldIdx).forEach((cmapName) => {{
        const item = document.createElement('div');
        item.className = 'custom-dropdown-item cmap-dropdown-item';
        if (!controlsEnabled) item.classList.add('disabled');
        if (cmapName === currentCmap) item.classList.add('selected');
        const prefix = document.createElement('span');
        prefix.className = 'cmap-dropdown-prefix';
        prefix.appendChild(createCmapSwatch(entry, fieldIdx, cmapName));
        item.appendChild(prefix);
        const value = document.createElement('span');
        value.className = 'layer-id';
        value.textContent = isNoCmap(cmapName) ? NO_CMAP : cmapName;
        item.appendChild(value);
        item.addEventListener('click', () => {{
          if (!controlsEnabled) return;
          setLayerCmap(entry, cmapName);
          setCmapButtonContent(button, cmapName);
          menu.querySelectorAll('.custom-dropdown-item').forEach((el) => {{
            el.classList.toggle('selected', el === item);
          }});
          closeDropdownMenu(menu);
        }});
        menu.appendChild(item);
      }});
      button.addEventListener('click', (ev) => {{
        if (!controlsEnabled) return;
        ev.stopPropagation();
        toggleDropdownMenu(button, menu);
      }});
      const visual = document.createElement('div');
      visual.className = 'colorbar-visual';
      const distributionWrap = document.createElement('div');
      distributionWrap.className = 'colorbar-distribution-wrap';
      const distributionViz = buildDistributionSvg(
        getEntryDisplayedValues(entry, fieldIdx),
        {{ min: meta.fullMin, max: meta.fullMax }},
      );
      const distribution = distributionViz ? distributionViz.svg : null;
      const gradientWrap = document.createElement('div');
      gradientWrap.className = 'colorbar-gradient-wrap';
      const gradient = document.createElement('div');
      gradient.className = 'colorbar-gradient';
      gradient.style.background = gradientCss(meta.gradient);
      wrapper.appendChild(title);
      dropdown.appendChild(button);
      dropdown.appendChild(menu);
      dropdownWrap.appendChild(dropdown);
      wrapper.appendChild(dropdownWrap);
      if (distribution) {{
        distributionWrap.appendChild(distribution);
      }}
      visual.appendChild(distributionWrap);
      gradientWrap.appendChild(gradient);
      wrapper.appendChild(visual);
      if (Number.isFinite(meta.min) && Number.isFinite(meta.max)) {{
        const range = document.createElement('div');
        range.className = 'colorbar-range';
        const minLabel = document.createElement('span');
        const maxLabel = document.createElement('span');
        range.appendChild(minLabel);
        range.appendChild(maxLabel);
        wrapper.appendChild(range);

        const currentCmapIsClipped = !isNoCmap(currentCmap);
        if (currentCmapIsClipped) {{
          const clipWrap = document.createElement('div');
          clipWrap.className = 'colorbar-clip';
          const clipSlider = document.createElement('div');
          clipSlider.className = 'colorbar-clip-slider';
          const clipTrack = document.createElement('div');
          clipTrack.className = 'colorbar-clip-track';
          const clipFill = document.createElement('div');
          clipFill.className = 'colorbar-clip-fill';
          const minInput = document.createElement('input');
          minInput.type = 'range';
          minInput.min = '0';
          minInput.max = String(CLIP_SLIDER_MAX);
          minInput.step = '1';
          minInput.setAttribute('aria-label', `Minimum clip for ${{colorbarLabel}}`);
          const maxInput = document.createElement('input');
          maxInput.type = 'range';
          maxInput.min = '0';
          maxInput.max = String(CLIP_SLIDER_MAX);
          maxInput.step = '1';
          maxInput.setAttribute('aria-label', `Maximum clip for ${{colorbarLabel}}`);

          clipSlider.appendChild(clipTrack);
          clipSlider.appendChild(clipFill);
          clipSlider.appendChild(minInput);
          clipSlider.appendChild(maxInput);
          clipWrap.appendChild(clipSlider);
          visual.appendChild(clipWrap);

          const setActiveClipHandle = (handleName) => {{
            const activeHandle = handleName === 'min' ? 'min' : 'max';
            clipSlider.dataset.activeHandle = activeHandle;
            minInput.style.zIndex = activeHandle === 'min' ? '3' : '2';
            maxInput.style.zIndex = activeHandle === 'max' ? '3' : '2';
          }};

          const updateClipHandlePriority = (clientX = null) => {{
            const rect = clipSlider.getBoundingClientRect();
            if (!(rect.width > 0)) {{
              setActiveClipHandle(clipSlider.dataset.activeHandle || 'max');
              return;
            }}
            const currentMeta = getScalarMeta(entry, fieldIdx);
            const clipState = currentMeta?.clipState || createClipState();
            const minPos = rect.width * clampUnit(clipState.minFrac, 0.0);
            const maxPos = rect.width * clampUnit(clipState.maxFrac, 1.0);
            let activeHandle = clipSlider.dataset.activeHandle || 'max';

            if (Number.isFinite(clientX)) {{
              const x = Math.max(0.0, Math.min(rect.width, clientX - rect.left));
              if (Math.abs(maxPos - minPos) <= 14) {{
                activeHandle = x <= minPos ? 'min' : 'max';
              }} else {{
                activeHandle = Math.abs(x - minPos) <= Math.abs(x - maxPos) ? 'min' : 'max';
              }}
            }}

            setActiveClipHandle(activeHandle);
          }};

          clipSlider.addEventListener('pointerdown', (ev) => {{
            updateClipHandlePriority(ev.clientX);
          }});
          clipSlider.addEventListener('pointermove', (ev) => {{
            if ((ev.buttons || 0) === 0) {{
              updateClipHandlePriority(ev.clientX);
            }}
          }});
          minInput.addEventListener('pointerdown', () => {{
            setActiveClipHandle('min');
          }});
          maxInput.addEventListener('pointerdown', () => {{
            setActiveClipHandle('max');
          }});

          const syncClipControl = () => {{
            const currentMeta = getScalarMeta(entry, fieldIdx);
            const clipState = currentMeta?.clipState || createClipState();
            const startPct = clampUnit(clipState.minFrac, 0.0) * 100.0;
            const endPct = clampUnit(clipState.maxFrac, 1.0) * 100.0;
            clipFill.style.left = `${{startPct}}%`;
            clipFill.style.width = `${{Math.max(0.0, endPct - startPct)}}%`;
            distributionViz?.setClipFractions(clipState.minFrac, clipState.maxFrac);
            minInput.value = String(clipToSliderValue(clipState.minFrac, 0.0));
            maxInput.value = String(clipToSliderValue(clipState.maxFrac, 1.0));
            minInput.disabled = !controlsEnabled || !currentMeta?.clipEnabled;
            maxInput.disabled = !controlsEnabled || !currentMeta?.clipEnabled;
            minLabel.textContent = formatScalarValue(currentMeta?.min);
            maxLabel.textContent = formatScalarValue(currentMeta?.max);
            updateClipHandlePriority();
          }};

          minInput.addEventListener('input', () => {{
            setColorClipBoundary(entry, fieldIdx, 'min', minInput.value);
          }});
          maxInput.addEventListener('input', () => {{
            setColorClipBoundary(entry, fieldIdx, 'max', maxInput.value);
          }});

          wrapper.__syncClipControl = syncClipControl;
          syncClipControl();
        }} else {{
          minLabel.textContent = formatScalarValue(meta.min);
          maxLabel.textContent = formatScalarValue(meta.max);
          distributionViz?.setClipFractions(0.0, 1.0);
        }}
        visual.appendChild(gradientWrap);
      }}
      return wrapper;
    }}

    function updateColorbars() {{
      for (const entry of layers) {{
        if (!entry.fieldOptionsContainer) continue;
        entry.fieldOptionsContainer.innerHTML = '';
        const panel = createEntryFieldPanel(entry);
        if (panel) entry.fieldOptionsContainer.appendChild(panel);
      }}
    }}

    function updateInfoPanel() {{
      let totalPoints = 0;
      let totalVertices = 0;
      let totalFaces = 0;
      for (const entry of layers) {{
        if (!entry.visible) continue;
        const pointCount = Array.isArray(entry.layer.points) ? entry.layer.points.length : 0;
        totalPoints += pointCount;
        totalVertices += pointCount;
      }}
      const pointsNode = document.getElementById('info-points');
      const verticesNode = document.getElementById('info-vertices');
      const facesNode = document.getElementById('info-faces');
      if (pointsNode) pointsNode.textContent = totalPoints.toLocaleString();
      if (verticesNode) verticesNode.textContent = totalVertices.toLocaleString();
      if (facesNode) facesNode.textContent = totalFaces.toLocaleString();
    }}

    function rebuildLayerList() {{
      const container = document.getElementById('layer-list');
      container.innerHTML = '';
      layers.forEach((entry, idx) => {{
        const row = document.createElement('div');
        row.className = 'layer-row';
        const hasMultipleFields = getEntryFieldCount(entry) > 1;
        const hasScalarOptions = getEntryFieldCount(entry) > 0 || Boolean(entry.layer.scalar_meta) || Boolean(entry.layer.values);
        const hasLayerOptions = true;
        let refreshFieldControls = null;
        let controlsBody = null;
        let toggleButton = null;
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = entry.visible;
        checkbox.addEventListener('change', () => {{
          entry.visible = checkbox.checked;
          entry.objects.points.visible = entry.visible;
          if (entry.objects.lines) entry.objects.lines.visible = entry.visible;
          rebuildLayerList();
          hideHoverTooltip();
          requestRenderAndHover();
        }});
        const info = document.createElement('div');
        info.className = 'layer-main';
        const label = document.createElement('div');
        label.className = 'layer-label';
        label.textContent = `${{idx + 1}}. ${{entry.layer.label}}`;
        info.appendChild(label);

        if (hasLayerOptions) {{
          const toggle = document.createElement('button');
          toggle.type = 'button';
          toggle.className = 'layer-options-toggle';
          toggle.setAttribute('aria-label', `Toggle field options for ${{entry.layer.label}}`);
          toggle.setAttribute('aria-expanded', entry.optionsExpanded ? 'true' : 'false');
          toggle.textContent = entry.optionsExpanded ? '▾' : '▸';
          toggle.addEventListener('click', () => {{
            entry.optionsExpanded = !entry.optionsExpanded;
            toggle.setAttribute('aria-expanded', entry.optionsExpanded ? 'true' : 'false');
            toggle.textContent = entry.optionsExpanded ? '▾' : '▸';
            controlsBody?.classList.toggle('hidden', !entry.optionsExpanded);
            if (!entry.optionsExpanded) closeAllDropdownMenus();
          }});
          toggleButton = toggle;
        }}

        controlsBody = document.createElement('div');
        controlsBody.className = 'layer-controls-body';
        controlsBody.classList.toggle('hidden', !entry.optionsExpanded);
        controlsBody.classList.toggle('disabled', !entry.visible);

        const optionGrid = document.createElement('div');
        optionGrid.className = 'layer-option-grid';

        const pointSizeWrap = document.createElement('div');
        pointSizeWrap.className = 'layer-option-control';
        const pointSizeLabel = document.createElement('label');
        pointSizeLabel.textContent = 'Point size';
        const pointSizeInput = document.createElement('input');
        pointSizeInput.type = 'range';
        pointSizeInput.min = '1';
        pointSizeInput.max = '12';
        pointSizeInput.step = '0.5';
        pointSizeInput.value = String(entry.pointSize);
        pointSizeInput.disabled = !entry.visible;
        pointSizeInput.addEventListener('input', () => {{
          updateEntryPointSize(entry, pointSizeInput.value);
          requestRenderAndHover();
        }});
        pointSizeWrap.appendChild(pointSizeLabel);
        pointSizeWrap.appendChild(pointSizeInput);
        optionGrid.appendChild(pointSizeWrap);

        const opacityWrap = document.createElement('div');
        opacityWrap.className = 'layer-option-control';
        const opacityLabel = document.createElement('label');
        opacityLabel.textContent = 'Point opacity';
        const opacityInput = document.createElement('input');
        opacityInput.type = 'range';
        opacityInput.min = '0.05';
        opacityInput.max = '1';
        opacityInput.step = '0.05';
        opacityInput.value = String(entry.opacity);
        opacityInput.disabled = !entry.visible;
        opacityInput.addEventListener('input', () => {{
          updateEntryOpacity(entry, opacityInput.value);
          requestRenderAndHover();
        }});
        opacityWrap.appendChild(opacityLabel);
        opacityWrap.appendChild(opacityInput);
        optionGrid.appendChild(opacityWrap);

        if (entry.objects.lines && entry.visible) {{
          const edgeThicknessWrap = document.createElement('div');
          edgeThicknessWrap.className = 'layer-option-control';
          const edgeThicknessLabel = document.createElement('label');
          edgeThicknessLabel.textContent = 'Edge thickness';
          const edgeThicknessInput = document.createElement('input');
          edgeThicknessInput.type = 'range';
          edgeThicknessInput.min = '1';
          edgeThicknessInput.max = '12';
          edgeThicknessInput.step = '0.5';
          edgeThicknessInput.value = String(entry.edgeThickness);
          edgeThicknessInput.disabled = !entry.visible;
          edgeThicknessInput.addEventListener('input', () => {{
            updateEntryEdgeThickness(entry, edgeThicknessInput.value);
            requestRenderAndHover();
          }});
          edgeThicknessWrap.appendChild(edgeThicknessLabel);
          edgeThicknessWrap.appendChild(edgeThicknessInput);
          optionGrid.appendChild(edgeThicknessWrap);
        }}

        if (hasMultipleFields) {{
          const shareScaleLabel = document.createElement('label');
          shareScaleLabel.className = 'layer-option-toggle';
          const shareScaleInput = document.createElement('input');
          shareScaleInput.type = 'checkbox';
          shareScaleInput.checked = Boolean(entry.shareFieldScale);
          shareScaleInput.disabled = !entry.visible || !getSharedFieldScaleRange(entry);
          shareScaleInput.addEventListener('change', () => {{
            setLayerShareFieldScale(entry, shareScaleInput.checked);
          }});
          shareScaleLabel.appendChild(shareScaleInput);
          shareScaleLabel.appendChild(document.createTextNode('Share colormap scale across fields'));
          optionGrid.appendChild(shareScaleLabel);
        }}

        controlsBody.appendChild(optionGrid);

        const fieldNames = entry.layer.field_names || [];
        if (fieldNames.length > 1) {{
          const wrap = document.createElement('div');
          wrap.className = 'layer-select-wrapper layer-field-controls';
          const controls = document.createElement('div');
          controls.className = 'field-select-inline';
          const prevButton = document.createElement('button');
          prevButton.type = 'button';
          prevButton.className = 'field-nav-button';
          prevButton.textContent = '←';
          prevButton.setAttribute('aria-label', 'Previous field');
          const dropdown = document.createElement('div');
          dropdown.className = 'custom-dropdown';
          const button = document.createElement('button');
          button.type = 'button';
          button.className = 'layer-select-button';
          const menu = document.createElement('div');
          menu.className = 'custom-dropdown-menu';
          const nextButton = document.createElement('button');
          nextButton.type = 'button';
          nextButton.className = 'field-nav-button';
          nextButton.textContent = '→';
          nextButton.setAttribute('aria-label', 'Next field');

          const updateFieldControls = (fieldIdx) => {{
            setFieldButtonContent(button, fieldNames[fieldIdx]);
            menu.querySelectorAll('.custom-dropdown-item').forEach((el, idx2) => {{
              el.classList.toggle('selected', idx2 === fieldIdx);
            }});
            button.disabled = !entry.visible;
            if (!entry.visible) closeDropdownMenu(menu);
            prevButton.disabled = !entry.visible || fieldIdx <= 0;
            nextButton.disabled = !entry.visible || fieldIdx >= fieldNames.length - 1;
          }};
          refreshFieldControls = () => updateFieldControls(Math.max(0, Math.min(fieldNames.length - 1, entry.activeFieldIdx ?? 0)));

          const selectField = (fieldIdx) => {{
            const nextFieldIdx = Math.max(0, Math.min(fieldNames.length - 1, fieldIdx));
            setLayerField(entry, nextFieldIdx);
            updateFieldControls(nextFieldIdx);
          }};

          fieldNames.forEach((name, fieldIdx) => {{
            const item = document.createElement('div');
            item.className = 'custom-dropdown-item';
            if (fieldIdx === (entry.activeFieldIdx ?? 0)) item.classList.add('selected');
            const value = document.createElement('span');
            value.className = 'layer-id';
            value.textContent = name;
            item.appendChild(value);
            item.addEventListener('click', () => {{
              selectField(fieldIdx);
              closeDropdownMenu(menu);
            }});
            menu.appendChild(item);
          }});

          prevButton.addEventListener('click', (ev) => {{
            ev.stopPropagation();
            closeDropdownMenu(menu);
            selectField((entry.activeFieldIdx ?? 0) - 1);
          }});
          button.addEventListener('click', (ev) => {{
            ev.stopPropagation();
            toggleDropdownMenu(button, menu);
          }});
          nextButton.addEventListener('click', (ev) => {{
            ev.stopPropagation();
            closeDropdownMenu(menu);
            selectField((entry.activeFieldIdx ?? 0) + 1);
          }});

          updateFieldControls(Math.max(0, Math.min(fieldNames.length - 1, entry.activeFieldIdx ?? 0)));
          controls.appendChild(prevButton);
          dropdown.appendChild(button);
          dropdown.appendChild(menu);
          controls.appendChild(dropdown);
          controls.appendChild(nextButton);
          wrap.appendChild(controls);
          controlsBody.appendChild(wrap);
        }}

        if (hasScalarOptions) {{
          const colorbarHost = document.createElement('div');
          colorbarHost.className = 'layer-colorbar-host';
          controlsBody.appendChild(colorbarHost);
          entry.fieldOptionsContainer = colorbarHost;
        }} else {{
          entry.fieldOptionsContainer = null;
        }}

        row.appendChild(checkbox);
        row.appendChild(info);
        if (toggleButton) row.appendChild(toggleButton);
        if (hasLayerOptions) row.appendChild(controlsBody);
        container.appendChild(row);
      }});
      updateColorbars();
      updateInfoPanel();
    }}

    document.addEventListener('click', (ev) => {{
      if (!ev.target.closest('.custom-dropdown') && !ev.target.closest('.custom-dropdown-menu')) {{
        closeAllDropdownMenus();
      }}
    }});

    function setLayerField(entry, fieldIdx) {{
      const fieldNames = entry.layer.field_names || null;
      if (!fieldNames || fieldIdx < 0 || fieldIdx >= fieldNames.length) return;
      renderEntryColors(entry, fieldIdx);
      updateColorbars();
    }}

    function renderEntryColors(entry, fieldIdx = entry.activeFieldIdx ?? 0) {{
      const fieldValues = entry.layer.field_values || null;
      const values = fieldValues ? fieldValues[fieldIdx] : entry.layer.values || null;
      const rawColors = getEntryRawColors(entry, fieldIdx);
      const cmapName = getEntryCmap(entry, fieldIdx);
      const meta = getScalarMeta(entry, fieldIdx);
      const updatedFromValues = !isNoCmap(cmapName) && values && meta ? applyColorsFromValues(entry, values, meta) : false;

      if (updatedFromValues) {{
        entry.objects.points.geometry.attributes.color.needsUpdate = true;
      }} else if (rawColors) {{
        applyStoredColors(entry, rawColors);
      }} else {{
        applyStoredColors(entry, entry.layer.colors);
      }}

      entry.activeFieldIdx = fieldIdx;
      entry.activeValues = values;
      entry.activeCmap = cmapName;
    }}

    function setLayerCmap(entry, cmapName) {{
      const fieldIdx = entry.activeFieldIdx ?? 0;
      entry.activeCmap = cmapName;
      renderEntryColors(entry, fieldIdx);
      updateColorbars();
    }}

    function setLayerShareFieldScale(entry, enabled) {{
      entry.shareFieldScale = Boolean(enabled);
      const fieldIdx = Math.max(
        0,
        Math.min(
          entry.activeFieldIdx ?? 0,
          entry.fieldCmaps && entry.fieldCmaps.length > 0 ? entry.fieldCmaps.length - 1 : 0,
        ),
      );
      renderEntryColors(entry, fieldIdx);
      updateColorbars();
      hideHoverTooltip();
    }}

    function setColorClipBoundary(entry, fieldIdx, boundary, value) {{
      const state = getClipState(entry, fieldIdx);
      setClipBoundary(state, boundary, sliderValueToClip(value, boundary === 'max' ? 1.0 : 0.0));
      renderEntryColors(entry, fieldIdx);
      syncAllColorbarClipControls();
      hideHoverTooltip();
    }}

    function hideHoverTooltip() {{
      hoverTooltip.style.display = 'none';
    }}

    function toHexChannel(value) {{
      return Math.round(Math.min(255, Math.max(0, value))).toString(16).padStart(2, '0');
    }}

    function formatHoverColor(colorAttr, idx) {{
      if (!colorAttr || idx < 0 || idx >= colorAttr.count) return null;
      const r = colorAttr.getX(idx);
      const g = colorAttr.getY(idx);
      const b = colorAttr.getZ(idx);
      if (![r, g, b].every(Number.isFinite)) return null;
      const red = Math.round(Math.min(1, Math.max(0, r)) * 255);
      const green = Math.round(Math.min(1, Math.max(0, g)) * 255);
      const blue = Math.round(Math.min(1, Math.max(0, b)) * 255);
      return {{
        rgb: `rgb(${{red}}, ${{green}}, ${{blue}})`,
        hex: `#${{toHexChannel(red)}}${{toHexChannel(green)}}${{toHexChannel(blue)}}`,
      }};
    }}

    function buildHoverTooltipContent(details) {{
      hoverTooltip.replaceChildren();

      const title = document.createElement('div');
      title.className = 'hover-tooltip-title';
      title.textContent = details.layerLabel;
      hoverTooltip.appendChild(title);

      const addRow = (label, value = null, extraNode = null) => {{
        const row = document.createElement('div');
        row.className = 'hover-tooltip-row';

        const labelNode = document.createElement('div');
        labelNode.className = 'hover-tooltip-label';
        labelNode.textContent = label;
        row.appendChild(labelNode);

        if (extraNode) {{
          row.appendChild(extraNode);
        }}

        if (value !== null && value !== undefined && value !== '') {{
          const valueNode = document.createElement('div');
          valueNode.className = 'hover-tooltip-value';
          valueNode.textContent = value;
          row.appendChild(valueNode);
        }}

        hoverTooltip.appendChild(row);
      }};

      addRow('xyz', details.xyz);
      addRow('color', null, (() => {{
        const swatch = document.createElement('div');
        swatch.className = 'hover-tooltip-swatch';
        swatch.style.background = details.color.rgb;
        return swatch;
      }})());
      addRow('value', details.valueText);
    }}

    function showHoverTooltip(clientX, clientY, details) {{
      buildHoverTooltipContent(details);
      hoverTooltip.style.display = 'block';
      const margin = 12;
      const maxX = window.innerWidth - hoverTooltip.offsetWidth - margin;
      const maxY = window.innerHeight - hoverTooltip.offsetHeight - margin;
      hoverTooltip.style.left = `${{Math.min(clientX + 14, Math.max(margin, maxX))}}px`;
      hoverTooltip.style.top = `${{Math.min(clientY + 14, Math.max(margin, maxY))}}px`;
    }}

    function getHoverPointSize() {{
      let maxPointSize = 0.0;
      for (const entry of layers) {{
        if (!entry.visible) continue;
        maxPointSize = Math.max(maxPointSize, Number(entry.pointSize) || 0.0);
      }}
      if (maxPointSize > 0.0) return maxPointSize;
      return Math.max(1.0, Number(payload.point_size) || 1.0);
    }}

    function computeHoverThreshold(viewportHeightPx) {{
      const distance = camera.position.distanceTo(controls.target);
      const fov = THREE.MathUtils.degToRad(camera.fov);
      const worldPerPixel = (2 * Math.max(distance, 1e-6) * Math.tan(fov / 2)) / Math.max(viewportHeightPx, 1);
      const pointRadiusPx = Math.max(getHoverPointSize() * 0.5, 1.5);
      return Math.max(worldPerPixel * pointRadiusPx * 1.15, 1e-5);
    }}

    function pickBestPointHit(hits) {{
      if (!hits || hits.length === 0) return null;
      let bestHit = null;
      for (const hit of hits) {{
        if (hit.index === undefined || hit.index === null) continue;
        if (!bestHit) {{
          bestHit = hit;
          continue;
        }}
        const bestDistanceToRay = Number.isFinite(bestHit.distanceToRay) ? bestHit.distanceToRay : Infinity;
        const hitDistanceToRay = Number.isFinite(hit.distanceToRay) ? hit.distanceToRay : Infinity;
        if (hitDistanceToRay < bestDistanceToRay - 1e-9) {{
          bestHit = hit;
          continue;
        }}
        if (Math.abs(hitDistanceToRay - bestDistanceToRay) <= 1e-9 && hit.distance < bestHit.distance) {{
          bestHit = hit;
        }}
      }}
      return bestHit;
    }}

    function onPointerMove(event) {{
      const rect = renderer.domElement.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width;
      const y = (event.clientY - rect.top) / rect.height;
      pointerNdc.x = x * 2 - 1;
      pointerNdc.y = -(y * 2 - 1);
      raycaster.params.Points.threshold = computeHoverThreshold(rect.height);
      raycaster.setFromCamera(pointerNdc, camera);

      let best = null;
      for (const entry of layers) {{
        if (!entry.visible) continue;
        const hits = raycaster.intersectObject(entry.objects.points, false);
        if (!hits || hits.length === 0) continue;
        const hit = pickBestPointHit(hits);
        if (!hit) continue;
        const bestDistanceToRay = best && Number.isFinite(best.hit.distanceToRay) ? best.hit.distanceToRay : Infinity;
        const hitDistanceToRay = Number.isFinite(hit.distanceToRay) ? hit.distanceToRay : Infinity;
        if (
          !best
          || hitDistanceToRay < bestDistanceToRay - 1e-9
          || (Math.abs(hitDistanceToRay - bestDistanceToRay) <= 1e-9 && hit.distance < best.hit.distance)
        ) {{
          best = {{ entry, hit }};
        }}
      }}

      if (!best) {{
        hideHoverTooltip();
        return;
      }}

      const idx = best.hit.index ?? -1;
      if (idx < 0) {{
        hideHoverTooltip();
        return;
      }}

      const posAttr = best.entry.objects.points.geometry.getAttribute('position');
      const colorAttr = best.entry.objects.points.geometry.getAttribute('color');
      const px = posAttr.getX(idx);
      const py = posAttr.getY(idx);
      const pz = posAttr.getZ(idx);
      const hoverColor = formatHoverColor(colorAttr, idx);
      if (!hoverColor) {{
        hideHoverTooltip();
        return;
      }}

      const fieldNames = best.entry.layer.field_names || null;
      const fieldLabel = fieldNames ? fieldNames[best.entry.activeFieldIdx ?? 0] : null;
      const activeValues = best.entry.activeValues || best.entry.layer.values || null;
      const value = activeValues && idx < activeValues.length ? activeValues[idx] : null;
      const valueText = (value === null || value === undefined || !Number.isFinite(value))
        ? 'n/a'
        : Number(value).toFixed(4);

      showHoverTooltip(event.clientX, event.clientY, {{
        layerLabel: `${{best.entry.layer.label}}${{fieldLabel ? ` (${{fieldLabel}})` : ''}}`,
        xyz: `(${{px.toFixed(4)}}, ${{py.toFixed(4)}}, ${{pz.toFixed(4)}})`,
        color: hoverColor,
        valueText: valueText,
      }});
    }}

    for (const layer of payload.layers) {{
      const objects = buildPoints(layer);
      const fieldMetas = layer.field_scalar_meta || null;
      const fieldValues = layer.field_values || null;
      const fieldColorCount = Array.isArray(layer.field_colors) ? layer.field_colors.length : 0;
      const fieldCount = Math.max(
        Array.isArray(layer.field_names) ? layer.field_names.length : 0,
        Array.isArray(fieldValues) ? fieldValues.length : 0,
        Array.isArray(fieldMetas) ? fieldMetas.length : 0,
        fieldColorCount,
      );
      const fieldCmaps = fieldCount > 0
        ? Array.from({{ length: fieldCount }}, (_value, idx) => ((fieldMetas && fieldMetas[idx] && fieldMetas[idx].cmap) ? fieldMetas[idx].cmap : NO_CMAP))
        : null;
      const entry = {{
        layer,
        objects,
        visible: layer.visible !== false,
        opacity: getLayerOpacity(layer),
        edgeThickness: getLayerEdgeThickness(layer),
        pointSize: getLayerPointSize(layer),
        activeFieldIdx: Number(layer.active_field_idx || 0),
        activeValues: layer.values || null,
        activeCmap: layer.scalar_meta && layer.scalar_meta.cmap ? layer.scalar_meta.cmap : NO_CMAP,
        fieldCmaps,
        fieldClipStates: Array.from({{ length: Math.max(fieldCount, 1) }}, () => createClipState()),
        sharedFieldClip: createClipState(),
        shareFieldScale: Boolean(layer.share_field_scale ?? payload.initial_share_field_scale),
        optionsExpanded: layer.visible !== false,
        fieldOptionsContainer: null,
      }};
      layers.push(entry);
      entry.objects.points.visible = entry.visible;
      if (entry.objects.lines) entry.objects.lines.visible = entry.visible;
      updateEntryOpacity(entry, entry.opacity);
      updateEntryEdgeThickness(entry, entry.edgeThickness);
      updateEntryPointSize(entry, entry.pointSize);
      if (layer.field_names && layer.field_names.length > 0) {{
        setLayerField(entry, entry.activeFieldIdx);
      }} else if (fieldValues || layer.values) {{
        renderEntryColors(entry, entry.activeFieldIdx);
      }}
    }}

    rebuildLayerList();
    document.getElementById('btn-quit-top').addEventListener('click', async () => {{
      try {{
        await fetch('/__quit__', {{ method: 'POST' }});
      }} catch (_err) {{
        // Ignore network errors when server shuts down immediately.
      }}
      document.body.innerHTML = '<div style="display:grid;place-items:center;height:100vh;font-family:JetBrains Mono,monospace;">Viewer closed.</div>';
      window.close();
    }});
    document.getElementById('btn-theme-mode').addEventListener('click', () => applyThemeMode(!isDayMode));
    const commandsToggleBtn = document.getElementById('btn-toggle-commands');
    const controlsPanel = document.getElementById('controls');
    if (commandsToggleBtn && controlsPanel) {{
      const updateCommandsToggleButton = () => {{
        commandsToggleBtn.textContent = commandsVisible ? 'Hide Commands' : 'Show Commands';
        commandsToggleBtn.classList.toggle('active', !commandsVisible);
      }};
      commandsToggleBtn.addEventListener('click', () => {{
        commandsVisible = !commandsVisible;
        controlsPanel.classList.toggle('collapsed', !commandsVisible);
        updateCommandsToggleButton();
      }});
      updateCommandsToggleButton();
      controlsPanel.addEventListener('scroll', () => {{
        closeAllDropdownMenus();
      }}, {{ passive: true }});
    }}
    document.getElementById('btn-reset-camera').addEventListener('click', () => fitCamera());
    renderer.domElement.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('pointerleave', hideHoverTooltip);
    applyThemeMode(true);

    window.addEventListener('resize', () => {{
      closeAllDropdownMenus();
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      refreshLineMaterialResolutions();
      controls.handleResize?.();
    }});

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    if (layers.length > 0) {{
      requestAnimationFrame(() => {{
        fitCamera();
        renderer.render(scene, camera);
      }});
    }}
    animate();
  </script>
</body>
</html>"""

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
