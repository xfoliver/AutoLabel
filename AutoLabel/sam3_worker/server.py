"""Resident SAM3 HTTP worker.

Bind 127.0.0.1:9892 first, then load the model in a background thread so
/health is reachable immediately (status=loading until ready).

    C:\\dlcv\\envs\\sam3\\python.exe AutoLabel\\sam3_worker\\server.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from sam3_worker.inference import get_engine  # noqa: E402
from sam3_worker.schemas import health_payload  # noqa: E402

DEFAULT_HOST = os.environ.get("SAM3_WORKER_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("SAM3_WORKER_PORT", "9892"))
DEFAULT_RUNS = Path(os.environ.get("SAM3_WORKER_RUNS", str(HERE.parent / "runs" / "sam3")))

_LOAD_LOCK = threading.Lock()
_LOAD_STARTED = False


def _json_bytes(payload: Dict[str, Any], status: int = 200) -> tuple[int, bytes]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return status, body


def _begin_load() -> None:
    """Start model load once, in a daemon thread. Server stays up if it fails."""
    global _LOAD_STARTED
    with _LOAD_LOCK:
        if _LOAD_STARTED:
            return
        _LOAD_STARTED = True

    def _run() -> None:
        engine = get_engine()
        sys.stderr.write("[sam3_worker] loading SAM3 model (once)…\n")
        sys.stderr.flush()
        try:
            engine.load()
            sys.stderr.write(f"[sam3_worker] model loaded on {engine.device}\n")
            sys.stderr.flush()
        except Exception as e:
            traceback.print_exc()
            sys.stderr.write(f"[sam3_worker] model load failed: {e}\n")
            sys.stderr.flush()

    threading.Thread(target=_run, name="sam3-load", daemon=True).start()


class Sam3Handler(BaseHTTPRequestHandler):
    server_version = "AutoLabelSam3Worker/0.1"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stderr.write("[sam3_worker] " + (fmt % args) + "\n")
        sys.stderr.flush()

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        status, body = _json_bytes(payload, status)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in ("/health", "/"):
            engine = get_engine()
            self._send(
                200,
                health_payload(
                    model_loaded=engine.loaded,
                    device=engine.device,
                    error=engine.error,
                ),
            )
            return
        self._send(404, {"status": "error", "error": f"unknown path {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            self._send(400, {"status": "error", "error": "invalid JSON body"})
            return
        if not isinstance(data, dict):
            self._send(400, {"status": "error", "error": "JSON object required"})
            return
        if path != "/segment":
            self._send(404, {"status": "error", "error": f"unknown path {path}"})
            return
        engine = get_engine()
        if not engine.loaded:
            if engine.error:
                self._send(503, {"status": "error", "error": f"SAM3 model failed to load: {engine.error}"})
                return
            self._send(503, {"status": "loading", "error": "SAM3 model is still loading"})
            return
        try:
            result = self._segment(data)
        except FileNotFoundError as e:
            self._send(404, {"status": "error", "error": str(e)})
            return
        except ValueError as e:
            self._send(400, {"status": "error", "error": str(e)})
            return
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"status": "error", "error": str(e)})
            return
        self._send(200, result)

    def _segment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image_path = str(data.get("image_path") or "").strip()
        prompt = str(data.get("prompt") or "").strip()
        if not image_path:
            raise ValueError("image_path is required")
        if not prompt:
            raise ValueError("prompt is required")
        score = float(data.get("score_threshold") if data.get("score_threshold") is not None else 0.55)
        run_dir = str(data.get("run_dir") or "").strip()
        save_dir = Path(run_dir) if run_dir else DEFAULT_RUNS
        save_raw = bool(data.get("save_raw_mask", True))
        save_overlay = bool(data.get("save_overlay", True))

        engine = get_engine()
        packed = engine.segment(
            image_path=image_path,
            prompt=prompt,
            score_threshold=score,
            save_dir=save_dir,
            save_raw_mask=save_raw,
            save_overlay=save_overlay,
        )
        return {
            "status": "ok",
            "prompt": packed["prompt"],
            "image_path": packed["image_path"],
            "image_size": packed["image_size"],
            "instances": packed["instances"],
            "overlay_path": packed.get("overlay_path"),
            "result_json": packed.get("result_json"),
            "exif_orientation": packed.get("exif_orientation"),
            "pil_raw_size_wh": packed.get("pil_raw_size_wh"),
            "normalized_size_wh": packed.get("normalized_size_wh"),
            "applied_exif_transpose": packed.get("applied_exif_transpose"),
        }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    preload = True
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--host",) and i + 1 < len(argv):
            host = argv[i + 1]
            i += 2
            continue
        if arg in ("--port",) and i + 1 < len(argv):
            port = int(argv[i + 1])
            i += 2
            continue
        if arg == "--no-preload":
            preload = False
            i += 1
            continue
        i += 1

    httpd = ThreadingHTTPServer((host, port), Sam3Handler)
    # Allow the AutoLabel parent to reconnect after a crashed worker.
    httpd.allow_reuse_address = True
    sys.stderr.write(f"[sam3_worker] listening on http://{host}:{port}\n")
    sys.stderr.flush()
    if preload:
        _begin_load()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[sam3_worker] shutting down\n")
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
