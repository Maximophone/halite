#!/usr/bin/env python3
"""Serve one evolutionary run on localhost; no dependencies or network services."""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from train import HERE, snapshot


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, run, **kwargs):
        self.run = run
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path in ("/", "/index.html"):
                data, kind = (HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8"
            elif path == "/api/data":
                data = json.dumps(snapshot(json.loads((self.run / "state.json").read_text()))).encode()
                kind = "application/json"
            elif match := re.fullmatch(r"/api/replay/(\d+)", path):
                data = (self.run / f"gen-{int(match[1]):04d}" / "replay.json").read_bytes()
                kind = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (FileNotFoundError, KeyError):
            self.send_error(404, "Waiting for the first completed generation")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "runs/first")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, run=args.run.resolve()))
    print(f"Evolution lab: http://localhost:{args.port}  ·  {args.run.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
