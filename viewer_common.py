from __future__ import annotations

import json
import base64
import gzip
import os
import shutil
import tempfile
import threading
import webbrowser
from dataclasses import dataclass, field
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from pathlib import Path
from typing import Any, ClassVar, Mapping

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency in some environments
    torch = None


@lru_cache(maxsize=None)
def _read_asset_text_version(name: str, mtime_ns: int) -> str:
    del mtime_ns  # Included in the cache key so edited assets are reloaded.
    return (Path(__file__).with_name("assets") / name).read_text(encoding="utf-8")


def _read_asset_text(name: str) -> str:
    path = Path(__file__).with_name("assets") / name
    return _read_asset_text_version(name, path.stat().st_mtime_ns)


def _render_viewer_html(
    payload: Mapping[str, Any],
    *,
    title: str,
    meta_names: tuple[str, ...] = (),
) -> str:
    meta_tags = "\n".join(f'  <meta name="{escape(name)}" content="1">' for name in meta_names)
    template = _read_asset_text("viewer_template.html")
    inline_css = _read_asset_text("viewer.css")
    inline_js = _read_asset_text("viewer.js")
    payload_json = json.dumps(payload).replace("</", "<\\/")
    escaped_title = escape(title)
    return (
        template
        .replace("__META_TAGS__", meta_tags)
        .replace("__TITLE__", escaped_title)
        .replace("__INLINE_CSS__", inline_css)
        .replace("__INLINE_JS__", inline_js)
        .replace("__PAYLOAD_JSON__", payload_json)
    )


def _to_numpy(x: np.ndarray | torch.Tensor | None) -> np.ndarray | None:
    if x is None:
        return None
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _encode_array_payload(
    arr: np.ndarray, *, dtype: np.dtype | str | None = None
) -> dict[str, Any]:
    """Encode a NumPy array without expanding it into Python/JSON numbers.

    A nested ``tolist()`` representation is particularly costly for point
    clouds: every float becomes a Python object and then a decimal JSON token.
    The viewer consumes typed binary buffers, so keep geometry packed from
    Python through JavaScript.
    """
    encoded = np.ascontiguousarray(arr, dtype=dtype)
    return {
        "encoding": "numpy",
        "dtype": str(encoded.dtype),
        "shape": list(encoded.shape),
        "array": encoded,
    }


def _serialize_array_payloads(
    value: Any,
) -> Any:
    """Embed typed arrays as independently compressed base64 chunks.

    The RTX browser bridge copies viewer HTML to a local ``file://`` URL, so
    geometry cannot depend on any server request. Small independent chunks
    bound peak decoding memory while gzip keeps the transferred file compact.
    """
    if isinstance(value, dict) and value.get("encoding") == "numpy":
        array = np.ascontiguousarray(value["array"])
        spec = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        chunk_bytes = 4 * 1024 * 1024
        chunks = []
        raw = memoryview(array).cast("B")
        for start in range(0, array.nbytes, chunk_bytes):
            end = min(start + chunk_bytes, array.nbytes)
            compressed = gzip.compress(raw[start:end], compresslevel=1)
            chunks.append(base64.b64encode(compressed).decode("ascii"))
        spec.update(
            encoding="gzip-base64-chunks",
            chunks=chunks,
            nbytes=int(array.nbytes),
        )
        return spec
    if isinstance(value, dict):
        return {
            key: _serialize_array_payloads(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _serialize_array_payloads(item)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _serialize_array_payloads(item)
            for item in value
        ]
    return value


def _serialize_array_payloads_to_files(
    value: Any, directory: Path
) -> Any:
    """Store packed arrays beside server-mode HTML for lazy browser loading."""
    directory.mkdir(parents=True, exist_ok=True)
    next_id = 0

    def serialize(item: Any) -> Any:
        nonlocal next_id
        if isinstance(item, dict) and item.get("encoding") == "numpy":
            array = np.ascontiguousarray(item["array"])
            name = f"array_{next_id:05d}.bin"
            next_id += 1
            path = directory / name
            with path.open("wb") as handle:
                if array.nbytes:
                    handle.write(memoryview(array).cast("B"))
            return {
                "encoding": "raw-url",
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "url": f"{directory.name}/{name}",
                "nbytes": int(array.nbytes),
            }
        if isinstance(item, dict):
            return {key: serialize(child) for key, child in item.items()}
        if isinstance(item, list):
            return [serialize(child) for child in item]
        if isinstance(item, tuple):
            return [serialize(child) for child in item]
        return item

    return serialize(value)


def _sample_cmap(cmap: str, n: int = 7) -> list[str]:
    colormap = plt.get_cmap(cmap)
    values = np.linspace(0.0, 1.0, n)
    return [mcolors.to_hex(colormap(v), keep_alpha=False) for v in values]


def _sample_cmap_rgb(cmap: str, n: int = 64) -> list[list[int]]:
    colormap = plt.get_cmap(cmap)
    values = np.linspace(0.0, 1.0, n)
    rgb = np.asarray(colormap(values)[:, :3], dtype=np.float32)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8).tolist()


def _raw_scalar_gradient(min_val: float, max_val: float, n: int = 7) -> list[str]:
    use_unit_scale = max_val <= 1.0 + 1e-6
    gradient: list[str] = []
    for idx in range(n):
        t = 0.0 if n <= 1 else idx / (n - 1)
        raw = min_val + (max_val - min_val) * t
        intensity = raw * 255.0 if use_unit_scale else raw
        clipped = int(np.clip(np.round(intensity), 0, 255))
        gradient.append(f"rgb({clipped}, {clipped}, {clipped})")
    return gradient


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    max_color = float(np.nanmax(arr))
    if max_color <= 1.0 + 1e-6:
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _prepare_color_data(
    color_value: np.ndarray | torch.Tensor | str,
    *,
    raw_geometry_shape: tuple[int, ...],
    item_count: int,
    cmap: str,
    color_label: str,
    downsampling_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any] | None]:
    scalar_meta = None
    scalar_values: np.ndarray | None = None
    if not isinstance(cmap, str):
        raise ValueError("cmap must be a string")
    render_cmap = None if cmap in {"texture", "uniform"} else cmap
    if render_cmap is not None:
        plt.get_cmap(render_cmap)

    if isinstance(color_value, str):
        rgb = np.asarray(mcolors.to_rgb(color_value), dtype=np.float32)
        colors = np.repeat(rgb[None, :], item_count, axis=0)
        raw_colors = colors
        return colors, raw_colors, scalar_values, scalar_meta

    color_arr = _to_numpy(color_value)
    if color_arr is None:
        raise ValueError(f"{color_label} cannot be None")
    color_arr = np.asarray(color_arr)
    base_shape = raw_geometry_shape[:-1]
    if list(color_arr.shape) != list(raw_geometry_shape):
        if color_arr.shape == tuple(base_shape):
            scalar_values = np.asarray(color_arr, dtype=np.float32).reshape(-1)
            color_arr = np.repeat(color_arr[..., None], 3, axis=-1)
        elif color_arr.shape == tuple(base_shape) + (1,):
            scalar_values = np.asarray(color_arr, dtype=np.float32).reshape(-1)
            color_arr = np.repeat(color_arr, 3, axis=-1)
        else:
            raise ValueError(f"Unrecognized {color_label} shape {color_arr.shape}")

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
        "gradient": _sample_cmap(render_cmap) if render_cmap is not None else _raw_scalar_gradient(min_val, max_val),
    }

    if render_cmap is not None:
        norm = (scalar_vals - min_val) / max(max_val - min_val, 1e-8)
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
        colors = plt.get_cmap(render_cmap)(norm)[:, :3].astype(np.float32)
    else:
        colors = color_arr

    return colors, raw_colors, scalar_values, scalar_meta


def _remap_filtered_indices(indices: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int32)
    if idx.ndim != 2:
        raise ValueError("indices must be a 2D array")
    width = int(idx.shape[1])
    if idx.shape[0] == 0:
        return np.zeros((0, width), dtype=np.int32)

    n_raw = int(valid_mask.shape[0])
    in_range = np.ones((idx.shape[0],), dtype=bool)
    for col in range(width):
        in_range &= (idx[:, col] >= 0) & (idx[:, col] < n_raw)
    idx = idx[in_range]

    if not np.any(valid_mask) or idx.shape[0] == 0:
        return np.zeros((0, width), dtype=np.int32)

    keep = np.ones((idx.shape[0],), dtype=bool)
    for col in range(width):
        keep &= valid_mask[idx[:, col]]
    idx = idx[keep]
    if idx.shape[0] == 0:
        return np.zeros((0, width), dtype=np.int32)

    remap = np.full((n_raw,), -1, dtype=np.int32)
    remap[np.where(valid_mask)[0]] = np.arange(int(np.sum(valid_mask)), dtype=np.int32)
    return remap[idx]


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
            if isinstance(cmap_name, str) and cmap_name not in {"uniform", "texture"}:
                preferred.append(cmap_name)
        for meta_key in (
            "field_scalar_meta",
            "edge_field_scalar_meta",
            "face_field_scalar_meta",
        ):
            field_metas = layer.get(meta_key)
            if isinstance(field_metas, list):
                for meta in field_metas:
                    if not isinstance(meta, Mapping):
                        continue
                    cmap_name = meta.get("cmap")
                    if isinstance(cmap_name, str) and cmap_name not in {
                        "uniform",
                        "texture",
                    }:
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


@dataclass
class ViewerFigureBase:
    layers: list[dict[str, Any]] = field(default_factory=list)
    title: str | None = None
    point_size: float = 10.0
    opacity: float = 1.0
    share_colormap_scale: bool = False
    _server: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _server_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _served_html_path: Path | None = field(default=None, init=False, repr=False)
    _served_array_dir: Path | None = field(default=None, init=False, repr=False)
    _served_html_is_temp: bool = field(default=False, init=False, repr=False)
    _served_url: str | None = field(default=None, init=False, repr=False)

    default_title: ClassVar[str] = "Viewer"
    html_meta_names: ClassVar[tuple[str, ...]] = ()

    def _payload(self) -> dict[str, Any]:
        preferred_cmaps = _collect_layer_cmaps(self.layers)
        cmap_names = _available_cmaps(preferred_cmaps[0] if preferred_cmaps else None)
        for cmap_name in preferred_cmaps[1:]:
            if cmap_name not in cmap_names:
                cmap_names.insert(0, cmap_name)
        return {
            "title": self.title or self.default_title,
            "point_size": float(np.clip(self.point_size, 1.0, 12.0)),
            "opacity": float(np.clip(self.opacity, 0.0, 1.0)),
            "initial_share_field_scale": self.share_colormap_scale,
            "layers": self.layers,
            "available_cmaps": cmap_names,
            "colormap_presets": _colormap_catalog(cmap_names),
        }

    def to_html(self) -> str:
        payload = _serialize_array_payloads(self._payload())
        return _render_viewer_html(
            payload,
            title=self.title or self.default_title,
            meta_names=self.html_meta_names,
        )

    def _write_html(self, path: str | Path | None = None) -> Path:
        if path is None:
            fd, tmp_path = tempfile.mkstemp(prefix="pointcloud_viewer_", suffix=".html")
            os.close(fd)
            out = Path(tmp_path)
        else:
            out = Path(path)
        out.write_text(self.to_html(), encoding="utf-8")
        return out

    def _notebook_iframe_html(
        self,
        *,
        url: str | None = None,
        width: int | str = "100%",
        height: int = 760,
    ) -> str:
        width_css = f"{width}px" if isinstance(width, int) else str(width)
        viewer_url = url or self._serve()
        return (
            f'<div style="width:{width_css};max-width:100%;">'
            f'<iframe src="{escape(viewer_url)}"'
            f' style="display:block;width:100%;height:{int(height)}px;border:0;border-radius:8px;"'
            f' referrerpolicy="no-referrer"'
            f' sandbox="allow-scripts allow-same-origin">'
            f'</iframe>'
            f'</div>'
        )

    def _notebook_embedded_iframe_html(
        self,
        *,
        width: int | str = "100%",
        height: int = 760,
    ) -> str:
        """Return a kernel-independent notebook iframe with embedded arrays."""
        width_css = f"{width}px" if isinstance(width, int) else str(width)
        document = escape(self.to_html(), quote=True)
        return (
            f'<div style="width:{width_css};max-width:100%;">'
            f'<iframe srcdoc="{document}"'
            f' style="display:block;width:100%;height:{int(height)}px;border:0;border-radius:8px;"'
            f' sandbox="allow-scripts">'
            f'</iframe>'
            f'</div>'
        )

    def _repr_html_(self) -> str:
        return self._notebook_embedded_iframe_html()

    def _serve(self, host: str = "127.0.0.1", port: int = 0, path: str | Path | None = None) -> str:
        if self._server is not None:
            self.close_server()

        is_temp_html = path is None
        if path is None:
            fd, tmp_path = tempfile.mkstemp(prefix="pointcloud_viewer_", suffix=".html")
            os.close(fd)
            html_path = Path(tmp_path)
        else:
            html_path = Path(path)

        array_dir = Path(
            tempfile.mkdtemp(prefix=f"{html_path.stem}_arrays_", dir=html_path.parent)
        )
        payload = _serialize_array_payloads_to_files(self._payload(), array_dir)
        html_path.write_text(
            _render_viewer_html(
                payload,
                title=self.title or self.default_title,
                meta_names=self.html_meta_names,
            ),
            encoding="utf-8",
        )
        viewer = self
        server_ref: dict[str, ThreadingHTTPServer | None] = {"server": None}

        class _QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                if self.path in {"/__quit__", "/_shutdown"}:
                    body = b'{"status":"ok"}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                    def _shutdown() -> None:
                        try:
                            viewer.close_server()
                        except Exception:
                            srv = server_ref.get("server")
                            if srv is not None:
                                try:
                                    srv.shutdown()
                                except Exception:
                                    pass

                    threading.Thread(target=_shutdown, daemon=True, name="pointcloud-viewer-shutdown").start()
                    return
                self.send_error(404, "Not Found")

        handler = lambda *args, **kwargs: _QuietHandler(*args, directory=str(html_path.parent), **kwargs)
        server = ThreadingHTTPServer((host, port), handler)
        server_ref["server"] = server
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True, name="pointcloud-viewer-server")
        thread.start()

        self._server = server
        self._server_thread = thread
        self._served_html_path = html_path
        self._served_array_dir = array_dir
        self._served_html_is_temp = is_temp_html
        self._served_url = f"http://{host}:{server.server_port}/{html_path.name}"
        return self._served_url

    def close_server(self) -> None:
        served_html_path = self._served_html_path
        served_html_is_temp = self._served_html_is_temp
        served_array_dir = self._served_array_dir

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None

        if served_html_is_temp and served_html_path is not None:
            try:
                served_html_path.unlink(missing_ok=True)
            except Exception:
                pass
        if served_array_dir is not None:
            shutil.rmtree(served_array_dir, ignore_errors=True)

        self._served_html_path = None
        self._served_array_dir = None
        self._served_html_is_temp = False
        self._served_url = None

    def show(
        self,
        *,
        inline: bool | None = False,
        in_browser: bool | None = True,
        open_browser: bool | None = None,
        notebook_proxy: bool | str | None = None,
        width: int | str = "100%",
        height: int = 760,
        path: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        block: bool = False,
    ) -> Any:
        shell = None
        try:
            from IPython import get_ipython

            shell = get_ipython()
        except Exception:
            shell = None

        if open_browser is not None:
            if in_browser is not None and bool(in_browser) != bool(open_browser):
                raise ValueError("in_browser and open_browser must match when both are provided")
            in_browser = open_browser

        if inline is None and in_browser is None:
            inline = False
            in_browser = True
        elif inline is None:
            inline = not bool(in_browser)
        elif in_browser is None:
            in_browser = not bool(inline)

        inline = bool(inline)
        in_browser = bool(in_browser)

        if inline and shell is None:
            raise RuntimeError("inline=True requires an IPython notebook environment")
        if inline and in_browser:
            raise ValueError("inline=True and in_browser=True are mutually exclusive")

        if inline:
            from IPython.display import HTML, display

            _ = notebook_proxy
            display(
                HTML(
                    self._notebook_embedded_iframe_html(
                        width=width,
                        height=height,
                    )
                )
            )
            return None

        url = self._serve(host=host, port=port, path=path)

        if in_browser:
            webbrowser.open(url, new=2)

        if block and self._server_thread is not None:
            self._server_thread.join()

        return None

    def open_in_browser(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        path: str | Path | None = None,
        block: bool = False,
    ) -> str:
        url = self._serve(host=host, port=port, path=path)
        webbrowser.open(url, new=2)
        if block and self._server_thread is not None:
            self._server_thread.join()
        return url
