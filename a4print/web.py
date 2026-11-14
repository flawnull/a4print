"""A small local web page for a4print: pick or drop images in the browser,
choose options, download the print-ready PDF.

    a4print-web            # opens http://127.0.0.1:8765 in your browser

It only listens on this computer and uses Python's standard library. Each
upload is processed with the same option parser and pipeline as the
command line, so both behave the same.
"""

import argparse
import contextlib
import io
import json
import re
import secrets
import shutil
import sys
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import cli, files, upscale
from .pipeline import process_file

MAX_UPLOAD_BYTES = 300 * 1024 * 1024
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

# Query parameter -> command-line flag. Flags without a value are switches.
VALUE_FLAGS = {
    "fill": "--fill", "anchor": "--anchor", "orientation": "--orientation",
    "poster": "--poster", "margin": "--margin", "overlap": "--overlap",
    "ai_strength": "--ai-strength",
}
SWITCH_FLAGS = {"trim_border": "--trim-border", "no_enhance": "--no-enhance", "ai": "--ai"}

CONTENT_TYPES = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg"}


def safe_name(name):
    """A plain file name from whatever the browser sent."""
    name = Path(name or "").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "image"
    return name[:120]


def build_argv(upload, out_dir, query):
    """Turn the page's options into the same arguments the CLI takes."""
    argv = [str(upload), "-o", str(out_dir)]
    for key, flag in VALUE_FLAGS.items():
        value = query.get(key, [""])[0].strip()
        if value:
            argv += [flag, value]
    for key, flag in SWITCH_FLAGS.items():
        if query.get(key, [""])[0] in ("1", "true", "on"):
            argv.append(flag)
    return argv


def parse_options(argv):
    """Parse with the CLI's own parser. Returns (options, error message)."""
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            args = cli.build_parser().parse_args(argv)
    except SystemExit:
        lines = [line for line in stderr.getvalue().splitlines() if "error:" in line]
        return None, (lines[-1].split("error:", 1)[1].strip() if lines else "invalid options")
    return cli.options_from_args(args), ""


class App:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.Lock()  # one image at a time: processing is CPU-heavy
        try:
            upscale.find_realesrgan()
            self.ai_available = True
        except upscale.UpscalerMissing:
            self.ai_available = False

    def process(self, name, data, query):
        name = safe_name(name)
        if Path(name).suffix.lower() not in files.EXTENSIONS:
            return HTTPStatus.BAD_REQUEST, {"error": f"{name} isn't a supported image type"}

        job = secrets.token_hex(8)
        job_dir = self.root / job
        job_dir.mkdir(parents=True)
        upload = job_dir / "upload" / name
        upload.parent.mkdir()
        upload.write_bytes(data)

        opts, error = parse_options(build_argv(upload, job_dir, query))
        if error:
            return HTTPStatus.BAD_REQUEST, {"error": error}
        if opts.ai and not self.ai_available:
            return HTTPStatus.BAD_REQUEST, {"error": "Real-ESRGAN isn't installed on this computer"}

        with self.lock:
            try:
                report, _ = process_file(upload, job_dir, opts)
            except Exception as exc:
                return HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)}

        outputs = [
            {"name": out.name, "url": f"/files/{job}/{out.name}", "kind": out.suffix.lstrip(".")}
            for out in report.outputs
        ]
        return HTTPStatus.OK, {"name": name, "lines": cli.describe(report, opts), "files": outputs}

    def output_file(self, job, name):
        if not re.fullmatch(r"[0-9a-f]{16}", job) or safe_name(name) != name:
            return None
        path = self.root / job / name
        return path if path.is_file() else None


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "a4print"

        def log_message(self, fmt, *args):
            pass  # keep the terminal quiet

        def _local_only(self):
            """Refuse requests whose Host isn't this computer (DNS rebinding)."""
            host = self.headers.get("Host", "").rsplit(":", 1)[0]
            if host not in LOCAL_HOSTS:
                self.send_error(HTTPStatus.FORBIDDEN)
                return False
            return True

        def _send(self, status, body, content_type, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, payload):
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self):
            if not self._local_only():
                return
            url = urlparse(self.path)
            if url.path == "/":
                self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8",
                           {"Content-Security-Policy": "default-src 'self'; style-src 'unsafe-inline'; "
                                                       "script-src 'unsafe-inline'; img-src 'self' blob:"})
            elif url.path == "/status":
                self._json(HTTPStatus.OK, {"ai": app.ai_available})
            elif url.path.startswith("/files/"):
                parts = url.path.split("/")
                path = app.output_file(*parts[2:4]) if len(parts) == 4 else None
                if path is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                disposition = "inline" if path.suffix == ".jpg" else f'attachment; filename="{path.name}"'
                self._send(HTTPStatus.OK, path.read_bytes(),
                           CONTENT_TYPES.get(path.suffix, "application/octet-stream"),
                           {"Content-Disposition": disposition})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if not self._local_only():
                return
            url = urlparse(self.path)
            # Other websites can't set this header without a CORS preflight,
            # which this server never approves, so they can't post here.
            if url.path != "/process" or self.headers.get("X-A4print") != "1":
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_UPLOAD_BYTES:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "file is empty or over 300 MB"})
                return
            data = self.rfile.read(length)
            query = parse_qs(url.query)
            status, payload = app.process(query.get("name", [""])[0], data, query)
            self._json(status, payload)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(prog="a4print-web", description="Run the a4print web page on this computer.")
    parser.add_argument("--port", type=int, default=8765, help="port to listen on (default: %(default)s)")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args(argv)

    root = Path(tempfile.mkdtemp(prefix="a4print-web-"))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(App(root)))
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"a4print is running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        shutil.rmtree(root, ignore_errors=True)
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>a4print</title>
<style>
  :root {
    --bg: #f6f5f2; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6a66;
    --line: #dddad3; --accent: #2f5d50; --accent-text: #ffffff; --warn: #8a5300; --error: #a12a2a;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #161615; --panel: #20201e; --text: #ecebe7; --muted: #a3a29c;
      --line: #383733; --accent: #7fb8a6; --accent-text: #10201b; --warn: #e0a54a; --error: #ef8a8a;
    }
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 860px; margin: 0 auto; padding: 32px 16px 64px; }
  h1 { font-size: 28px; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 24px; }
  .drop { border: 2px dashed var(--line); border-radius: 12px; background: var(--panel);
          padding: 32px 16px; text-align: center; cursor: pointer; }
  .drop.over { border-color: var(--accent); }
  .drop strong { display: block; font-size: 18px; }
  .drop span { color: var(--muted); font-size: 14px; }
  form { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
         gap: 12px 16px; margin: 20px 0; }
  label { display: flex; flex-direction: column; gap: 4px; font-size: 14px; color: var(--muted); }
  label.check { flex-direction: row; align-items: center; gap: 8px; color: var(--text); }
  select, input[type=number] { font: inherit; color: var(--text); background: var(--panel);
         border: 1px solid var(--line); border-radius: 8px; padding: 8px; }
  button { font: inherit; font-weight: 600; background: var(--accent); color: var(--accent-text);
           border: 0; border-radius: 8px; padding: 12px 20px; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  .result { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
            padding: 16px; margin-top: 16px; }
  .result h2 { font-size: 17px; margin: 0 0 8px; overflow-wrap: anywhere; }
  .result img { width: 100%; border-radius: 6px; margin: 8px 0; }
  .lines { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px; font-size: 14px; }
  .lines dt { color: var(--muted); }
  .lines dd { margin: 0; }
  .warning { color: var(--warn); }
  .error { color: var(--error); }
  .links { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .links a { color: var(--accent); font-weight: 600; text-decoration: none;
             border: 1px solid var(--line); border-radius: 8px; padding: 6px 12px; }
  .hint { font-size: 14px; color: var(--muted); margin-top: 24px; }
</style>
</head>
<body>
<main>
  <h1>a4print</h1>
  <p class="sub">Turn any image into a print-ready A4 page or a multi-sheet poster.</p>

  <div class="drop" id="drop" tabindex="0" role="button">
    <strong>Drop images here or click to choose</strong>
    <span>JPEG, PNG, WebP, AVIF, HEIC, TIFF</span>
  </div>
  <input type="file" id="file" accept="image/*,.heic,.heif" multiple hidden>

  <form id="opts">
    <label>Fill
      <select name="fill">
        <option value="crop">Crop to fill the page</option>
        <option value="extend">Extend the edges</option>
        <option value="fit">Fit with white border</option>
      </select>
    </label>
    <label>Keep when cropping
      <select name="anchor">
        <option value="center">Center</option><option value="top">Top</option>
        <option value="bottom">Bottom</option><option value="left">Left</option>
        <option value="right">Right</option>
      </select>
    </label>
    <label>Size
      <select name="poster">
        <option value="">One A4 page</option>
        <option value="2x2">2x2 sheets, about A2</option>
        <option value="3x2">3x2 sheets, banner</option>
        <option value="3x3">3x3 sheets, about A1</option>
        <option value="4x4">4x4 sheets</option>
      </select>
    </label>
    <label>Orientation
      <select name="orientation">
        <option value="auto">Automatic</option><option value="portrait">Portrait</option>
        <option value="landscape">Landscape</option>
      </select>
    </label>
    <label>White margin (mm, blank = default)
      <input type="number" name="margin" min="0" max="40" step="1" placeholder="0, posters 6">
    </label>
    <label class="check"><input type="checkbox" name="trim_border"> Remove plain border</label>
    <label class="check"><input type="checkbox" name="no_enhance"> No cleanup or tone changes</label>
    <label class="check" id="ai-row" hidden><input type="checkbox" name="ai"> AI upscaling (Real-ESRGAN)</label>
  </form>

  <button id="go" disabled>Make print files</button>
  <div id="results"></div>

  <p class="hint">Print the PDF at 100% / "Actual size", not "Fit to page". Everything runs on this
  computer; images are deleted when you stop a4print-web.</p>
</main>

<script>
const drop = document.getElementById("drop");
const input = document.getElementById("file");
const go = document.getElementById("go");
const results = document.getElementById("results");
let chosen = [];

fetch("/status").then(r => r.json()).then(s => {
  document.getElementById("ai-row").hidden = !s.ai;
});

function choose(list) {
  chosen = Array.from(list);
  drop.querySelector("strong").textContent =
    chosen.length ? chosen.map(f => f.name).join(", ") : "Drop images here or click to choose";
  go.disabled = !chosen.length;
}

drop.addEventListener("click", () => input.click());
drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") input.click(); });
input.addEventListener("change", () => choose(input.files));
drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", e => {
  e.preventDefault(); drop.classList.remove("over"); choose(e.dataTransfer.files);
});

function el(tag, attrs = {}, text = "") {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
  if (text) node.textContent = text;
  return node;
}

go.addEventListener("click", async () => {
  go.disabled = true;
  const params = new URLSearchParams();
  for (const [key, value] of new FormData(document.getElementById("opts"))) params.set(key, value);
  for (const file of chosen) {
    const box = el("div", {class: "result"});
    box.append(el("h2", {}, file.name), el("p", {}, "Working..."));
    results.prepend(box);
    params.set("name", file.name);
    try {
      const response = await fetch("/process?" + params, {
        method: "POST", body: file, headers: {"X-A4print": "1"}
      });
      const data = await response.json();
      box.querySelector("p").remove();
      if (!response.ok) { box.append(el("p", {class: "error"}, data.error || "Failed")); continue; }
      const preview = data.files.find(f => f.kind === "jpg");
      if (preview) box.append(el("img", {src: preview.url, alt: "Before and after preview"}));
      const dl = el("dl", {class: "lines"});
      for (const [label, text] of data.lines) {
        dl.append(el("dt", {}, label), el("dd", label === "warning" ? {class: "warning"} : {}, text));
      }
      box.append(dl);
      const links = el("div", {class: "links"});
      for (const f of data.files.filter(f => f.kind !== "jpg")) {
        links.append(el("a", {href: f.url, download: f.name}, "Download " + f.kind.toUpperCase()));
      }
      box.append(links);
    } catch (err) {
      box.querySelector("p").className = "error";
      box.querySelector("p").textContent = "Couldn't reach a4print-web. Is it still running?";
    }
  }
  go.disabled = false;
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
