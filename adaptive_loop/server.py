"""Local control panel server (localhost only). Standard library, no dependencies.

GET  /             the panel page
GET  /api/status   state, modulators, affect, hot items, recent actions
POST /api/toggle   the On/Off button (the only way to change the flag)
POST /api/chat     send a message: {"text": "..."}
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config
from runtime import Runtime

PANEL = Path(__file__).resolve().parent / "panel.html"


def make_handler(runtime: Runtime):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("content-type", kind)
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data) -> None:
            self._send(code, json.dumps(data, default=str).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PANEL.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self._json(200, runtime.status())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                return self._json(400, {"error": "invalid JSON"})
            try:
                if self.path == "/api/toggle":
                    self._json(200, {"on": runtime.toggle()})
                elif self.path == "/api/chat":
                    result = runtime.chat(str(body.get("text", "")))
                    self._json(409 if "error" in result else 200, result)
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:
                self._json(500, {"error": str(exc)})

        def log_message(self, fmt, *args):   # keep the terminal quiet
            pass

    return Handler


def serve(runtime: Runtime) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((config.PANEL_HOST, config.PANEL_PORT), make_handler(runtime))
