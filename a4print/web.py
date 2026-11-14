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

import pikepdf

from . import calibrate, cli, files, setup_ai, upscale
from .pipeline import process_file

MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MAX_JOBS = 25  # older results are deleted, so a long session can't fill the disk
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
        self.lock = threading.Lock()  # one job at a time: processing is CPU-heavy
        self.ai_available = self._find_ai()

    @staticmethod
    def _find_ai():
        try:
            upscale.find_realesrgan()
            return True
        except upscale.UpscalerMissing:
            return False

    def status(self):
        return {"ai": self.ai_available, "ai_installable": setup_ai.supported(),
                "print_settings": calibrate.load_settings()}

    def save_print_settings(self, query):
        try:
            settings = calibrate.answers_to_settings(int(query.get("black", ["0"])[0]), query.get("gray", [""])[0])
        except ValueError as exc:
            return HTTPStatus.BAD_REQUEST, {"error": str(exc)}
        calibrate.save_settings(settings)
        return HTTPStatus.OK, self.status()

    def setup_ai(self):
        """Install Real-ESRGAN from the page (one-time ~50 MB download)."""
        with self.lock:
            if not self.ai_available:
                try:
                    setup_ai.install()
                except setup_ai.SetupError as exc:
                    return HTTPStatus.BAD_GATEWAY, {**self.status(), "error": str(exc)}
                problem = setup_ai.self_test()
                if problem:
                    return HTTPStatus.INTERNAL_SERVER_ERROR, {
                        **self.status(), "error": f"installed, but it didn't run on this computer: {problem}"}
                self.ai_available = True
        return HTTPStatus.OK, self.status()

    def process(self, name, data, query):
        name = safe_name(name)
        if Path(name).suffix.lower() not in files.EXTENSIONS:
            return HTTPStatus.BAD_REQUEST, {"error": f"{name} isn't a supported image type"}

        job = secrets.token_hex(8)
        job_dir = self.root / job
        job_dir.mkdir(parents=True)
        self._forget_old_jobs(keep=job_dir)
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
        summary = {
            "quality": report.quality.split(":")[0].split(",")[0],
            "dpi": round(report.effective_dpi),
            "low": report.effective_dpi < opts.min_dpi,
            "ai_used": "Real-ESRGAN" in report.resize_method,
            "ai_note": ("tuned for artwork" if report.content == "artwork" else "tuned for photos")
                       + ("; small text kept as is" if "small text" in report.resize_method else "")
                       if "Real-ESRGAN" in report.resize_method else "",
            "fill": report.fill_reason or report.fill,
            "poster": (f"{report.poster.cols}x{report.poster.rows} sheets, "
                       f"{report.poster.poster_mm[0] / 10:.0f} x {report.poster.poster_mm[1] / 10:.0f} cm")
                      if report.poster else "",
        }
        return HTTPStatus.OK, {"name": name, "summary": summary, "lines": cli.describe(report, opts),
                               "files": outputs}

    def _forget_old_jobs(self, keep):
        """Delete all but the newest MAX_JOBS results, never `keep` (the job
        being made now, in case timestamps tie)."""
        jobs = sorted((p for p in self.root.iterdir() if p.is_dir() and p != keep),
                      key=lambda p: p.stat().st_mtime)
        for old in jobs[:max(0, len(jobs) - (MAX_JOBS - 1))]:
            shutil.rmtree(old, ignore_errors=True)

    def combined_pdf(self, jobs):
        """One PDF with the pages of several results, in the given order."""
        merged = pikepdf.new()
        for job in jobs:
            if not re.fullmatch(r"[0-9a-f]{16}", job) or not (self.root / job).is_dir():
                continue
            for pdf_path in sorted((self.root / job).glob("*.pdf")):
                with pikepdf.open(pdf_path) as part:
                    merged.pages.extend(part.pages)
        if not merged.pages:
            return None
        merged.Root.ViewerPreferences = pikepdf.Dictionary(PrintScaling=pikepdf.Name.None_)
        out = io.BytesIO()
        merged.save(out)
        return out.getvalue()

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
                self._json(HTTPStatus.OK, app.status())
            elif url.path == "/test-page.pdf":
                path = app.root / "a4print_test_page.pdf"
                if not path.exists():
                    calibrate.write_test_page(path)
                self._send(HTTPStatus.OK, path.read_bytes(), "application/pdf",
                           {"Content-Disposition": 'attachment; filename="a4print_test_page.pdf"'})
            elif url.path == "/all.pdf":
                jobs = parse_qs(url.query).get("jobs", [""])[0].split(",")[:MAX_JOBS]
                body = app.combined_pdf(jobs)
                if body is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send(HTTPStatus.OK, body, "application/pdf",
                           {"Content-Disposition": 'attachment; filename="a4print_all.pdf"'})
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
            if url.path not in ("/process", "/setup-ai", "/settings") or self.headers.get("X-A4print") != "1":
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if url.path == "/setup-ai":
                self._json(*app.setup_ai())
                return
            if url.path == "/settings":
                self._json(*app.save_print_settings(parse_qs(url.query)))
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
    --bg: #f6f5f2; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6a66; --line: #dddad3;
    --accent: #2f5d50; --accent-text: #ffffff; --soft: #e8efec;
    --good: #2f6b3a; --ok: #8a6a00; --low: #a13a2a; --error: #a12a2a;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #161615; --panel: #20201e; --text: #ecebe7; --muted: #a3a29c; --line: #383733;
      --accent: #7fb8a6; --accent-text: #10201b; --soft: #263330;
      --good: #8fd19b; --ok: #e5c35a; --low: #f0907e; --error: #ef8a8a;
    }
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 720px; margin: 0 auto; padding: 32px 16px 64px; }
  h1 { font-size: 28px; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 24px; }
  .step { font-size: 14px; font-weight: 600; color: var(--muted); margin: 24px 0 8px; }
  .drop { border: 2px dashed var(--line); border-radius: 12px; background: var(--panel);
          padding: 36px 16px; text-align: center; cursor: pointer; }
  .drop.over, .drop:focus-visible { border-color: var(--accent); outline: none; }
  .drop strong { display: block; font-size: 18px; overflow-wrap: anywhere; }
  .drop span { color: var(--muted); font-size: 14px; }
  .sizes { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  @media (max-width: 480px) { .sizes { grid-template-columns: 1fr; } }
  .sizes label { display: block; cursor: pointer; }
  .sizes input { position: absolute; opacity: 0; }
  .sizes span { display: block; height: 100%; border: 1px solid var(--line); border-radius: 10px;
                background: var(--panel); padding: 10px 12px; }
  .sizes b { display: block; }
  .sizes small { color: var(--muted); }
  .sizes input:checked + span { border: 2px solid var(--accent); padding: 9px 11px; background: var(--soft); }
  .sizes input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: 2px; }
  .ai { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; background: var(--panel);
        border: 1px solid var(--line); border-radius: 10px; padding: 12px; margin-top: 8px; }
  .ai label { display: flex; align-items: center; gap: 8px; font-weight: 600; }
  .ai small { color: var(--muted); flex-basis: 100%; }
  button { font: inherit; font-weight: 600; border-radius: 8px; cursor: pointer; }
  button.primary { background: var(--accent); color: var(--accent-text); border: 0;
                   padding: 14px 24px; width: 100%; margin-top: 24px; font-size: 17px; }
  button.secondary { background: transparent; color: var(--accent); border: 1px solid var(--accent);
                     padding: 6px 12px; }
  button:disabled { opacity: .5; cursor: default; }
  details { margin-top: 16px; }
  summary { cursor: pointer; color: var(--muted); font-size: 15px; }
  .more { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 12px 16px; margin-top: 12px; }
  .more label { display: flex; flex-direction: column; gap: 4px; font-size: 14px; color: var(--muted); }
  .more label.check { flex-direction: row; align-items: center; gap: 8px; color: var(--text); }
  select, input[type=number] { font: inherit; color: var(--text); background: var(--panel);
         border: 1px solid var(--line); border-radius: 8px; padding: 8px; }
  .result { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
            padding: 16px; margin-top: 24px; }
  .result h2 { font-size: 17px; margin: 0 0 12px; overflow-wrap: anywhere; }
  .verdict { border-radius: 8px; padding: 10px 12px; background: var(--soft); margin-bottom: 8px; }
  .verdict b.good { color: var(--good); } .verdict b.ok { color: var(--ok); } .verdict b.low { color: var(--low); }
  .what { color: var(--muted); font-size: 15px; margin: 0 0 8px; }
  .result img { width: 100%; border-radius: 6px; margin: 8px 0; }
  .downloads { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 8px; }
  .downloads a.main { background: var(--accent); color: var(--accent-text); text-decoration: none;
                      font-weight: 600; border-radius: 8px; padding: 10px 18px; }
  .downloads a.alt { color: var(--accent); font-size: 15px; }
  .lines { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px; font-size: 14px; margin: 8px 0 0; }
  .lines dt { color: var(--muted); } .lines dd { margin: 0; overflow-wrap: anywhere; }
  .error { color: var(--error); }
  .hint { font-size: 14px; color: var(--muted); margin-top: 32px; }
  .calib { margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--line); font-size: 14px; }
  .calib a { color: var(--accent); }
  .calib-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin: 8px 0; }
  .calib-row label { display: flex; flex-direction: column; gap: 4px; color: var(--muted); }
  .calib small { color: var(--muted); }
</style>
</head>
<body>
<main>
  <h1>a4print</h1>
  <p class="sub">Turn an image into a print-ready PDF for your home printer.</p>

  <div class="step">1. Choose images</div>
  <div class="drop" id="drop" tabindex="0" role="button" aria-label="Choose images">
    <strong>Drop images here or click to choose</strong>
    <span>Photos, posters, downloads: JPEG, PNG, WebP, HEIC and more</span>
  </div>
  <input type="file" id="file" accept="image/*,.heic,.heif" multiple hidden>

  <form id="opts">
    <div class="step">2. Print size</div>
    <div class="sizes" role="radiogroup">
      <label><input type="radio" name="poster" value="" checked>
        <span><b>A4 page</b><small>One sheet, 21 x 30 cm</small></span></label>
      <label><input type="radio" name="poster" value="2x2">
        <span><b>Poster, about A2</b><small>4 sheets to glue, 56 x 39 cm</small></span></label>
      <label><input type="radio" name="poster" value="3x3">
        <span><b>Poster, about A1</b><small>9 sheets to glue, 84 x 57 cm</small></span></label>
    </div>

    <div class="ai" id="ai-box">
      <label id="ai-on" hidden><input type="checkbox" name="ai"> Sharpen with AI</label>
      <button type="button" class="secondary" id="ai-setup" hidden>Set up AI sharpening</button>
      <small id="ai-note">Makes small images (like Pinterest downloads) print noticeably sharper.</small>
    </div>

    <details>
      <summary>More options</summary>
      <div class="more">
        <label>Fit to the page
          <select name="fill">
            <option value="auto">Automatic (recommended)</option>
            <option value="crop">Fill the page, crop the edges</option>
            <option value="fit">Keep it whole, white border</option>
            <option value="extend">Keep it whole, extend the edges</option>
          </select>
        </label>
        <label>When cropping, keep the
          <select name="anchor">
            <option value="center">Center</option><option value="top">Top</option>
            <option value="bottom">Bottom</option><option value="left">Left</option>
            <option value="right">Right</option>
          </select>
        </label>
        <label>Orientation
          <select name="orientation">
            <option value="auto">Automatic</option><option value="portrait">Portrait</option>
            <option value="landscape">Landscape</option>
          </select>
        </label>
        <label>Other poster sizes
          <select id="poster-more">
            <option value="">None</option><option value="3x2">3x2 sheets, banner</option>
            <option value="2x3">2x3 sheets, tall</option><option value="4x4">4x4 sheets, 111 x 76 cm</option>
          </select>
        </label>
        <label>White margin in mm (for printers that can't print to the edge)
          <input type="number" name="margin" min="0" max="40" step="1" placeholder="automatic">
        </label>
        <label class="check"><input type="checkbox" name="trim_border"> Cut off a plain border</label>
        <label class="check"><input type="checkbox" name="no_enhance"> Don't clean up or adjust tones</label>
      </div>
      <div class="calib">
        <b>Printer calibration</b> (optional, once per printer): prints come out darker than the
        screen on many printers. <a href="/test-page.pdf" download>Print the test page</a> at 100%,
        then give your answers:
        <div class="calib-row">
          <label>First square you can tell from black
            <select id="cal-black"><option>1</option><option>2</option><option selected>3</option>
              <option>4</option><option>5</option><option>6</option><option>7</option><option>8</option>
              <option>9</option></select></label>
          <label>Gray patch that matches its stripes
            <select id="cal-gray"><option>A</option><option selected>B</option><option>C</option>
              <option>D</option><option>E</option></select></label>
          <button type="button" class="secondary" id="cal-save">Save for this printer</button>
        </div>
        <small id="cal-status"></small>
      </div>
    </details>
  </form>

  <button class="primary" id="go" disabled>Make print files</button>
  <div id="results"></div>

  <p class="hint">When printing, choose "Actual size" or 100%, not "Fit to page". Everything runs on this
  computer, and your images are deleted when you stop a4print-web.</p>
</main>

<script>
const $ = id => document.getElementById(id);
const drop = $("drop"), input = $("file"), go = $("go"), results = $("results");
let chosen = [];

const VERDICTS = {
  excellent: ["good", "Excellent", "Sharp even up close."],
  good: ["good", "Good", "Looks sharp at normal viewing distance."],
  "fine from a distance": ["ok", "OK from a distance", "Fine on a wall; a little soft up close."],
  low: ["low", "Low", "The image is small for this size, so it will look soft or blocky up close."],
};

function showAI(status) {
  $("ai-on").hidden = !status.ai;
  $("ai-setup").hidden = status.ai || !status.ai_installable;
  $("ai-box").hidden = !status.ai && !status.ai_installable;
  if (status.ai) $("ai-note").textContent =
    "Makes small images print sharper. Used only when an image needs it; the preview shows a detail at print size.";
}
function showSettings(s) {
  const p = s.print_settings || {};
  $("cal-status").textContent = p.shadows !== undefined
    ? "Saved: shadow lift " + p.shadows + ", midtone lift " + p.midtones + ". Used for every print."
    : "Not calibrated: using standard values.";
}
$("cal-save").addEventListener("click", async () => {
  const q = new URLSearchParams({black: $("cal-black").value, gray: $("cal-gray").value});
  const r = await fetch("/settings?" + q, {method: "POST", headers: {"X-A4print": "1"}});
  const s = await r.json();
  if (r.ok) showSettings(s); else $("cal-status").textContent = s.error;
});

fetch("/status").then(r => r.json()).then(s => {
  showAI(s);
  showSettings(s);
  // AI is on by default once installed; it skips images that are already sharp.
  if (s.ai) $("ai-on").querySelector("input").checked = true;
});

$("ai-setup").addEventListener("click", async () => {
  const btn = $("ai-setup");
  btn.disabled = true; btn.textContent = "Downloading (about 50 MB)...";
  try {
    const r = await fetch("/setup-ai", {method: "POST", headers: {"X-A4print": "1"}});
    const s = await r.json();
    if (!r.ok) throw new Error(s.error || "setup failed");
    showAI(s);
    $("ai-on").querySelector("input").checked = true;
  } catch (e) {
    btn.disabled = false; btn.textContent = "Try AI setup again";
    $("ai-note").textContent = "AI setup didn't work: " + e.message;
  }
});

$("poster-more").addEventListener("change", e => {
  if (e.target.value) document.querySelectorAll("input[name=poster]").forEach(r => r.checked = false);
});
document.querySelectorAll("input[name=poster]").forEach(r =>
  r.addEventListener("change", () => { $("poster-more").value = ""; }));

function choose(list) {
  chosen = Array.from(list).filter(f => f.size > 0);
  drop.querySelector("strong").textContent =
    chosen.length ? chosen.map(f => f.name).join(", ") : "Drop images here or click to choose";
  go.disabled = !chosen.length;
}
drop.addEventListener("click", () => input.click());
drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
input.addEventListener("change", () => choose(input.files));
drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("over"); choose(e.dataTransfer.files); });

function el(tag, attrs = {}, text = "") {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
  if (text) node.textContent = text;
  return node;
}

function verdict(summary, aiChecked) {
  const key = Object.keys(VERDICTS).find(k => summary.quality.startsWith(k)) || "low";
  let [cls, label, sentence] = VERDICTS[key];
  if (summary.low && summary.ai_used)
    sentence = "The original is small. AI sharpening helps a lot, but a larger version would still print better.";
  const box = el("div", {class: "verdict"});
  const b = el("b", {class: cls}, "Print quality: " + label);
  box.append(b, document.createTextNode(" (" + summary.dpi + " DPI). " + sentence));
  if (summary.low && !summary.ai_used) {
    const aiReady = !$("ai-on").hidden, aiSetup = !$("ai-setup").hidden;
    box.append(document.createTextNode(aiReady ? " Tick \"Sharpen with AI\" and make it again."
      : aiSetup ? " Set up AI sharpening above, then make it again." : " A larger version of the image would help."));
  }
  return box;
}

go.addEventListener("click", async () => {
  go.disabled = true;
  const params = new URLSearchParams();
  for (const [key, value] of new FormData($("opts"))) if (value !== "") params.set(key, value);
  if ($("poster-more").value) params.set("poster", $("poster-more").value);
  for (const file of chosen) {
    const box = el("div", {class: "result"});
    const aiChecked = params.get("ai") === "on";
    box.append(el("h2", {}, file.name), el("p", {class: "what"},
      aiChecked ? "Working... AI sharpening can take a while." : "Working..."));
    results.prepend(box);
    box.scrollIntoView({behavior: "smooth", block: "start"});
    params.set("name", file.name);
    try {
      const response = await fetch("/process?" + params, {method: "POST", body: file, headers: {"X-A4print": "1"}});
      const data = await response.json();
      box.querySelector(".what").remove();
      if (!response.ok) { box.append(el("p", {class: "error"}, data.error || "Something went wrong.")); continue; }
      const s = data.summary;
      box.append(verdict(s, aiChecked));
      box.append(el("p", {class: "what"}, (s.poster ? "Poster: " + s.poster + ". " : "") +
        s.fill.charAt(0).toUpperCase() + s.fill.slice(1) + "." +
        (s.ai_used ? " Sharpened with AI, " + s.ai_note + "." : "")));
      const preview = data.files.find(f => f.kind === "jpg");
      if (preview) box.append(el("img", {src: preview.url, alt: "Before and after"}));
      const dl = el("div", {class: "downloads"});
      const pdf = data.files.find(f => f.kind === "pdf"), png = data.files.find(f => f.kind === "png");
      if (pdf) dl.append(el("a", {class: "main", href: pdf.url, download: pdf.name}, "Download PDF"));
      if (png) dl.append(el("a", {class: "alt", href: png.url, download: png.name}, "or the PNG image"));
      box.append(dl);
      const more = el("details"); more.append(el("summary", {}, "Details"));
      const lines = el("dl", {class: "lines"});
      for (const [label, text] of data.lines) lines.append(el("dt", {}, label), el("dd", {}, text));
      more.append(lines); box.append(more);
    } catch (err) {
      const p = box.querySelector(".what") || box.appendChild(el("p"));
      p.className = "error"; p.textContent = "Couldn't reach a4print-web. Is it still running?";
    }
  }
  go.disabled = false;
  updateAll();
});

// "Download all" once there are two or more results with a PDF.
const done = [];
function updateAll() {
  done.length = 0;
  document.querySelectorAll(".result a.main").forEach(a => done.push(a.getAttribute("href").split("/")[2]));
  let bar = $("all");
  if (done.length < 2) { if (bar) bar.remove(); return; }
  if (!bar) {
    bar = el("div", {id: "all", class: "downloads"});
    results.before(bar);
  }
  bar.replaceChildren(el("a", {class: "main", href: "/all.pdf?jobs=" + done.slice().reverse().join(","),
    download: "a4print_all.pdf"}, "Download all as one PDF (" + done.length + " results)"));
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
