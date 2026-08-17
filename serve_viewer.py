import http.server
import json
import os
import base64
import socketserver
from pathlib import Path

PORT = 5125
OUTPUT_DIR = Path(__file__).parent / "output"
VIEWER_DIR = Path(__file__).parent / "viewer"
INPUT_DIR = Path(r"D:\profile\10hinh_novabyte")


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_file(VIEWER_DIR / "index.html", "text/html")
        elif self.path == "/api/images":
            self.send_json(self._get_all_images())
        elif self.path.startswith("/api/image/"):
            name = self.path.split("/api/image/")[1]
            self.send_json(self._get_image_data(name))
        elif self.path.startswith("/output/"):
            file_path = OUTPUT_DIR / self.path[8:]
            if file_path.exists():
                ct = "image/png" if file_path.suffix == ".png" else "image/svg+xml" if file_path.suffix == ".svg" else "application/json"
                self.send_file(file_path, ct)
            else:
                self.send_error(404)
        elif self.path.startswith("/input/"):
            file_path = INPUT_DIR / self.path[7:]
            if file_path.exists():
                self.send_file(file_path, "image/png")
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def send_file(self, path, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with open(path, "rb") as f:
            self.wfile.write(f.read())

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _get_all_images(self):
        images = []
        for d in sorted(OUTPUT_DIR.iterdir()):
            if d.is_dir() and (d / "metrics.json").exists():
                metrics = json.loads((d / "metrics.json").read_text())
                images.append({"name": d.name, "metrics": metrics})
        return images

    def _get_image_data(self, name):
        img_dir = OUTPUT_DIR / name
        if not img_dir.exists():
            return {"error": "not found"}
        metrics = json.loads((img_dir / "metrics.json").read_text())
        svg_content = (img_dir / "result.svg").read_text() if (img_dir / "result.svg").exists() else ""
        return {"name": name, "metrics": metrics, "svg": svg_content}

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), ViewerHandler) as httpd:
        print(f"Viewer running at http://localhost:{PORT}")
        print(f"Serving outputs from: {OUTPUT_DIR}")
        httpd.serve_forever()
