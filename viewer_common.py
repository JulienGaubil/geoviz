from __future__ import annotations

import tempfile
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar, Mapping

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency in some environments
    torch = None


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
    _served_url: str | None = field(default=None, init=False, repr=False)

    default_title: ClassVar[str] = "Viewer"

    def _payload(self) -> dict[str, Any]:
        preferred_cmaps = _collect_layer_cmaps(self.layers)
        cmap_names = _available_cmaps(preferred_cmaps[0] if preferred_cmaps else None)
        for cmap_name in preferred_cmaps[1:]:
            if cmap_name not in cmap_names:
                cmap_names.insert(0, cmap_name)
        return {
            "title": self.title or self.default_title,
            "point_size": float(np.clip(self.point_size, 1.0, 12.0)),
            "opacity": float(np.clip(self.opacity, 0.05, 1.0)),
            "initial_share_field_scale": self.share_colormap_scale,
            "layers": self.layers,
            "available_cmaps": cmap_names,
            "colormap_presets": _colormap_catalog(cmap_names),
        }

    def to_html(self) -> str:
        raise NotImplementedError("to_html() must be implemented by subclasses")

    def _write_html(self, path: str | Path | None = None) -> Path:
        if path is None:
            fd, tmp_path = tempfile.mkstemp(prefix="pointcloud_viewer_", suffix=".html")
            Path(tmp_path).write_text(self.to_html(), encoding="utf-8")
            Path(tmp_path).chmod(0o644)
            return Path(tmp_path)
        out = Path(path)
        out.write_text(self.to_html(), encoding="utf-8")
        return out

    def _notebook_iframe_html(
        self,
        *,
        width: int | str = "100%",
        height: int = 760,
    ) -> str:
        width_css = f"{width}px" if isinstance(width, int) else str(width)
        html_content = self.to_html()
        srcdoc = (
            html_content
            .replace("&", "&amp;")
            .replace('"', "&quot;")
        )
        return (
            f'<div style="width:{width_css};max-width:100%;">'
            f'<iframe srcdoc="{srcdoc}"'
            f' style="display:block;width:100%;height:{int(height)}px;border:0;border-radius:8px;"'
            f' referrerpolicy="no-referrer"'
            f' sandbox="allow-scripts allow-same-origin">'
            f'</iframe>'
            f'</div>'
        )

    def _repr_html_(self) -> str:
        return self._notebook_iframe_html()

    def _serve(self, host: str = "127.0.0.1", port: int = 0, path: str | Path | None = None) -> str:
        if self._server is not None:
            self.close_server()

        html_path = self._write_html(path=path)
        viewer = self
        server_ref: dict[str, ThreadingHTTPServer | None] = {"server": None}

        class _QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                if self.path == "/__quit__":
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
        self._served_url = f"http://{host}:{server.server_port}/{html_path.name}"
        return self._served_url

    def close_server(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None
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
            display(HTML(self._notebook_iframe_html(width=width, height=height)))
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
