"""Serve existing leaf BMPs unchanged for Node-RED and Wokwi (standard library only)."""
import argparse
import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
# Support the root project and its nested sketch copy without machine-specific paths.
DATA = ROOT / "all_data" if (ROOT / "all_data").is_dir() else ROOT.parent / "all_data"
SOURCE = DATA


def build_catalog(source=SOURCE):
    """Read existing BMP files; never resize, convert, or write files."""
    catalog, images, failures = [], {}, []
    for path in sorted(source.iterdir(), key=lambda p: p.name.casefold()):
        if not path.is_file() or path.suffix.lower() != ".bmp":
            continue
        try:
            data = path.read_bytes()
            leaf_id = hashlib.sha256(path.name.encode("utf-8") + data).hexdigest()[:16]
            catalog.append({"id": leaf_id, "name": path.name, "source": path.name})
            images[leaf_id] = data
        except OSError as exc:
            failures.append(f"{path.name}: {exc}")
    return catalog, images, failures


def make_handler(catalog, images):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/api/leaves":
                self.reply(200, json.dumps(catalog).encode("utf-8"), "application/json")
            elif route.startswith("/leaves/") and route.endswith(".bmp"):
                data = images.get(route[len("/leaves/"):-4])
                if data is None:
                    self.send_error(404, "Unknown leaf. Refresh the dashboard leaf list.")
                else:
                    self.reply(200, data, "image/bmp")
            elif route == "/":
                self.reply(200, b"SGAS leaf server is running. Choose a leaf in Node-RED.", "text/plain")
            else:
                self.send_error(404, "Not found")

        def reply(self, status, data, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            print("[leaf server] " + fmt % args, flush=True)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to allow teammates on the LAN.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--directory", type=Path, default=SOURCE, help="Folder containing existing BMP files.")
    args = parser.parse_args()
    try:
        catalog, images, failures = build_catalog(args.directory)
    except OSError as exc:
        raise SystemExit(f"Cannot read BMP folder {args.directory}: {exc}")
    print(f"Loaded {len(catalog)} existing BMPs from {args.directory}. No files converted or written.", flush=True)
    for failure in failures:
        print("Skipped: " + failure, file=sys.stderr)
    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(catalog, images))
    except OSError as exc:
        raise SystemExit(f"Cannot start leaf server: {exc}")
    print(f"Leaf catalog: http://{args.host}:{args.port}/api/leaves", flush=True)
    print("Keep this terminal open. Restart after adding images, then click Refresh leaves. Ctrl+C stops it.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLeaf server stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
