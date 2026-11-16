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

from . import calibrate, cli, files, layout, maxq, setup_ai, upscale
from .pipeline import process_file

MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MAX_REMOVE_REGIONS = 20
MAX_JOBS = 25  # older results are deleted, so a long session can't fill the disk
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

# Query parameter -> command-line flag. Flags without a value are switches.
VALUE_FLAGS = {
    "paper": "--paper", "fill": "--fill", "anchor": "--anchor", "orientation": "--orientation",
    "crop_at": "--crop-at",
    "poster": "--poster", "margin": "--margin", "overlap": "--overlap",
    "ai_strength": "--ai-strength",
}
SWITCH_FLAGS = {"trim_border": "--trim-border", "no_enhance": "--no-enhance", "ai": "--ai",
                "max_quality": "--max-quality"}

CONTENT_TYPES = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg"}


def safe_name(name):
    """A plain file name from whatever the browser sent."""
    path = Path(Path(name or "").name)
    # Clean the name and the extension separately, so a name without ASCII
    # letters ("фото.jpg") or a very long one still keeps its extension.
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")[:80] or "image"
    suffix = re.sub(r"[^A-Za-z0-9]+", "", path.suffix)[:8]
    return f"{stem}.{suffix}" if suffix else stem


def file_role(name):
    """What an output file is for, so the page can place it."""
    for suffix, role in (("_detail_plain.jpg", "detail_plain"), ("_detail_final.jpg", "detail_final"),
                         ("_preview.jpg", "preview"), (".pdf", "pdf"), (".png", "png")):
        if name.endswith(suffix):
            return role
    return "other"


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
    for region in query.get("remove", [])[:MAX_REMOVE_REGIONS]:   # boxes drawn on the page
        argv += ["--remove", region.strip()]
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
                "max": maxq.available(), "max_libs": maxq.libraries_available(),
                "print_settings": calibrate.load_settings()}

    def setup_max(self):
        """Download the maximum quality models (PyTorch must be installed)."""
        if not maxq.libraries_available():
            return HTTPStatus.BAD_REQUEST, {**self.status(), "error": maxq.INSTALL_HINT}
        with self.lock:
            try:
                maxq.install_models()
            except setup_ai.SetupError as exc:
                return HTTPStatus.BAD_GATEWAY, {**self.status(), "error": str(exc)}
            problem = setup_ai.max_self_test()
            if problem:
                return HTTPStatus.INTERNAL_SERVER_ERROR, {**self.status(), "error": f"didn't run: {problem}"}
        return HTTPStatus.OK, self.status()

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

    MAX_SHEET_IMAGES = 48

    def sheet_add(self, name, data, query):
        """Add one image to a photo-sheet job (made on the first call)."""
        name = safe_name(name)
        if Path(name).suffix.lower() not in files.EXTENSIONS:
            return HTTPStatus.BAD_REQUEST, {"error": f"{name} isn't a supported image type"}
        job = query.get("job", [""])[0]
        if job:
            if not re.fullmatch(r"[0-9a-f]{16}", job) or not (self.root / job / "upload").is_dir():
                return HTTPStatus.NOT_FOUND, {"error": "unknown sheet job"}
        else:
            job = secrets.token_hex(8)
            (self.root / job / "upload").mkdir(parents=True)
            self._forget_old_jobs(keep=self.root / job)
        upload = self.root / job / "upload"
        count = len(list(upload.iterdir()))
        if count >= self.MAX_SHEET_IMAGES:
            return HTTPStatus.BAD_REQUEST, {"error": f"at most {self.MAX_SHEET_IMAGES} images per sheet job"}
        (upload / f"{count:03d}_{name}").write_bytes(data)   # numbered, so the order is kept
        return HTTPStatus.OK, {"job": job}

    def sheet_make(self, query):
        """Lay out a photo-sheet job's images and make the PDF."""
        from . import layout, sheets

        job = query.get("job", [""])[0]
        upload = self.root / job / "upload"
        if not re.fullmatch(r"[0-9a-f]{16}", job) or not upload.is_dir():
            return HTTPStatus.NOT_FOUND, {"error": "unknown sheet job"}
        query = {k: v for k, v in query.items() if k not in ("poster", "job", "name")}
        opts, error = parse_options(build_argv(upload, self.root / job, query))
        if error:
            return HTTPStatus.BAD_REQUEST, {"error": error}
        try:
            per_page = int(query.get("per_page", ["4"])[0])
            sheets.plan(per_page, opts.paper)
        except ValueError as exc:
            return HTTPStatus.BAD_REQUEST, {"error": str(exc)}
        if opts.max_quality and not maxq.available():
            return HTTPStatus.BAD_REQUEST, {"error": "maximum quality isn't set up on this computer"}
        if opts.ai and not opts.max_quality and not self.ai_available:
            return HTTPStatus.BAD_REQUEST, {"error": "Real-ESRGAN isn't installed on this computer"}

        with self.lock:
            loaded, lines = [], []
            for path in sorted(upload.iterdir()):
                display = path.name[4:]
                try:
                    loaded.append((display, files.load_rgb(path), files.jpeg_quality(path)))
                except Exception as exc:
                    lines.append([display, f"skipped: {exc}"])
            if not loaded:
                return HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "none of the images could be read"}
            try:
                pages, entries, failed = sheets.make_sheets(loaded, opts, per_page)
                lines += [[name, f"skipped: {reason}"] for name, reason in failed]
                if not pages:
                    return HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "; ".join(f"{n}: {r}" for n, r in failed)}
            except Exception as exc:
                return HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)}
            pdf = self.root / job / f"a4print_sheets_{opts.paper}.pdf"
            files.save_pdf([files.pdf_page(files.to_rgb_image(p)) for p in pages], pdf, layout.paper_mm(opts.paper))
            preview = self.root / job / f"a4print_sheets_{opts.paper}_preview.jpg"
            first = files.to_rgb_image(pages[0])
            first.thumbnail((1400, 1400))
            first.save(preview, "JPEG", quality=88, icc_profile=files.SRGB_ICC)

        for sheet, name, r in entries:
            note = f"sheet {sheet}, {r.effective_dpi:.0f} DPI, {r.quality}"
            if r.orientation:
                note += ", " + r.orientation
            lines.append([name, note])
        outputs = [{"name": p.name, "url": f"/files/{job}/{p.name}", "kind": p.suffix.lstrip("."),
                    "role": "pdf" if p.suffix == ".pdf" else "sheet_preview"} for p in (pdf, preview)]
        lowest = min(r.effective_dpi for _, _, r in entries)
        return HTTPStatus.OK, {"name": f"{len(loaded)} images, {per_page} per page",
                               "summary": {"sheets": len(pages), "images": len(loaded), "lowest_dpi": round(lowest),
                                           "paper": layout.PAPER_NAMES[opts.paper]},
                               "lines": lines, "files": outputs}

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
        opts.detail_files = True   # for the before/after slider
        if opts.max_quality and not maxq.available():
            return HTTPStatus.BAD_REQUEST, {"error": "maximum quality isn't set up on this computer"}
        if opts.ai and not opts.max_quality and not self.ai_available:
            return HTTPStatus.BAD_REQUEST, {"error": "Real-ESRGAN isn't installed on this computer"}

        with self.lock:
            try:
                report, _ = process_file(upload, job_dir, opts)
            except Exception as exc:
                return HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)}

        outputs = [
            {"name": out.name, "url": f"/files/{job}/{out.name}", "kind": out.suffix.lstrip("."),
             "role": file_role(out.name)}
            for out in report.outputs
        ]
        summary = {
            "quality": report.quality.split(":")[0].split(",")[0],
            "dpi": round(report.effective_dpi),
            "low": report.effective_dpi < opts.min_dpi,
            "ai_used": bool(report.content) and "skipped" not in report.resize_method,
            "ai_note": (("maximum quality, " if opts.max_quality else "")
                        + ("tuned for artwork" if report.content == "artwork" else "tuned for photos")
                        + ("; small text kept as is" if "small text" in report.resize_method else ""))
                       if report.content and "skipped" not in report.resize_method else "",
            "fill": ("fills the " + ("poster" if report.poster else "page") + " with the part you framed"
                     if opts.crop_at is not None and report.fill == "crop"
                     else report.fill_reason or report.fill),
            "detail_cm": report.detail_cm,
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
        if not re.fullmatch(r"[0-9a-f]{16}", job) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,200}", name):
            return None
        path = self.root / job / name
        return path if path.is_file() else None


def paper_info():
    """Size labels for the page's print-size cards, per paper, computed with
    the same poster planner the prints use (inches for Letter)."""
    from . import layout, poster

    info = {}
    for paper, (w, h) in layout.PAPERS.items():
        def size(mm, paper=paper):
            if paper == "letter":
                return f"{mm[0] / 25.4:.0f} x {mm[1] / 25.4:.0f} in"
            return f"{mm[0] / 10:.0f} x {mm[1] / 10:.0f} cm"
        entry = {"name": layout.PAPER_NAMES[paper], "mm": [w, h],
                 "page": "8.5 x 11 in" if paper == "letter" else size((w, h))}
        for cols, rows in ((2, 2), (3, 3), (3, 2), (2, 3), (4, 4)):
            plan = poster.make_plan(1500, 1000, cols, rows, paper_mm=(w, h))
            iso = f", {plan.size_name}" if plan.size_name and paper != "letter" else ""
            entry[f"{cols}x{rows}"] = size(plan.poster_mm) + iso
        info[paper] = entry
    return info


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

        def end_headers(self):
            # On every response, errors included: no MIME sniffing, never
            # shown inside another site's frame (clickjacking), no referrer
            # leaks, and results aren't kept in the browser cache.
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def _send(self, status, body, content_type, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))

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
                                                       "script-src 'unsafe-inline'; img-src 'self' blob:; "
                                                       "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"})
            elif url.path == "/status":
                self._json(HTTPStatus.OK, app.status())
            elif url.path == "/test-page.pdf":
                paper = parse_qs(url.query).get("paper", [""])[0]
                if paper not in layout.PAPERS:
                    paper = calibrate.load_settings().get("paper", "a4")
                path = app.root / f"a4print_test_page_{paper}.pdf"
                if not path.exists():
                    calibrate.write_test_page(path, paper)
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
            # Second layer behind the custom header: browsers send Origin on
            # POST, and only this page's own origin is accepted.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost", "::1"):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            url = urlparse(self.path)
            # Other websites can't set this header without a CORS preflight,
            # which this server never approves, so they can't post here.
            if url.path not in ("/process", "/setup-ai", "/setup-max", "/settings", "/sheet-add", "/sheet-make") \
                    or self.headers.get("X-A4print") != "1":
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if url.path == "/setup-ai":
                self._json(*app.setup_ai())
                return
            if url.path == "/setup-max":
                self._json(*app.setup_max())
                return
            if url.path == "/sheet-make":
                self._json(*app.sheet_make(parse_qs(url.query)))
                return
            if url.path == "/settings":
                self._json(*app.save_print_settings(parse_qs(url.query)))
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 < length <= MAX_UPLOAD_BYTES:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "file is empty or over 300 MB"})
                return
            data = self.rfile.read(length)
            query = parse_qs(url.query)
            handler = app.sheet_add if url.path == "/sheet-add" else app.process
            status, payload = handler(query.get("name", [""])[0], data, query)
            self._json(status, payload)

    return Handler


PAGE_PLACEHOLDER = "__PAPER_INFO__"


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
  .step select { font: inherit; font-weight: 600; padding: 2px 6px; margin: 0 2px; }
  .drop { border: 2px dashed var(--line); border-radius: 12px; background: var(--panel);
          padding: 36px 16px; text-align: center; cursor: pointer; }
  .drop.over, .drop:focus-visible { border-color: var(--accent); outline: none; }
  .drop strong { display: block; font-size: 18px; overflow-wrap: anywhere; }
  .drop span { color: var(--muted); font-size: 14px; }
  .sizes { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .sizes select { font: inherit; font-size: 13px; padding: 0 2px; }
  @media (max-width: 720px) { .sizes { grid-template-columns: repeat(2, 1fr); } }
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
  .compare { margin: 12px 0; }
  .compare .stage { position: relative; line-height: 0; border-radius: 8px; overflow: hidden;
                    max-width: 600px; margin: 0 auto; }
  .compare .stage img { margin: 0; border-radius: 0; display: block; width: 100%; }
  .compare .stage img.before { position: absolute; inset: 0; height: 100%; object-fit: cover; }
  .compare .divider { position: absolute; top: 0; bottom: 0; width: 2px; margin-left: -1px;
                      background: #fff; box-shadow: 0 0 0 1px rgba(0,0,0,.4); pointer-events: none; }
  .compare input[type=range] { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0;
                               opacity: 0; cursor: ew-resize; }
  .compare input[type=range]:focus-visible { opacity: .25; }
  .compare figcaption { display: flex; justify-content: space-between; gap: 8px; max-width: 600px;
                        margin: 6px auto 0; font-size: 13px; color: var(--muted); }
  .whole summary { font-size: 14px; }
  .erase-help { font-size: 14px; color: var(--muted); margin: 8px 0; }
  .crop-stage { position: relative; display: inline-block; max-width: 100%; line-height: 0; overflow: hidden;
                border-radius: 6px; touch-action: none; user-select: none; }
  .crop-stage img { max-width: 100%; max-height: 520px; }
  #crop-frame { position: absolute; box-sizing: border-box; border: 2px solid #fff; cursor: grab;
                box-shadow: 0 0 0 9999px rgba(0, 0, 0, .55); }
  #crop-frame:focus-visible { outline: 3px solid var(--accent); outline-offset: -5px; }
  #crop-frame i { position: absolute; background: rgba(255, 255, 255, .75); pointer-events: none; }
  #crop-frame i.v { top: 0; bottom: 0; width: 1px; }
  #crop-frame i.h { left: 0; right: 0; height: 1px; }
  #crop-reset { display: block; margin-top: 8px; }
  .erase-stage { position: relative; display: inline-block; max-width: 100%; line-height: 0;
                 touch-action: none; cursor: crosshair; user-select: none; }
  .erase-stage img { max-width: 100%; max-height: 520px; border-radius: 6px; }
  #erase-rects { position: absolute; inset: 0; }
  #erase-rects div { position: absolute; border: 2px solid #ff00b4; background: rgba(255, 0, 180, .22); }
  #erase-clear { margin-top: 8px; }
  .result img.sheet-preview { max-width: 420px; display: block; margin: 12px auto; box-shadow: 0 2px 12px rgba(0,0,0,.15); }
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
    <div class="step">2. Print size, on
      <select id="paper" name="paper" aria-label="Paper size">
        <option value="a4">A4</option><option value="letter">Letter</option><option value="a3">A3</option>
      </select> paper</div>
    <div class="sizes" role="radiogroup">
      <label><input type="radio" name="poster" value="" checked>
        <span><b id="size-page">A4 page</b><small id="size-page-info">One sheet</small></span></label>
      <label><input type="radio" name="poster" value="2x2">
        <span><b>Poster, 4 sheets</b><small id="size-2x2-info"></small></span></label>
      <label><input type="radio" name="poster" value="3x3">
        <span><b>Poster, 9 sheets</b><small id="size-3x3-info"></small></span></label>
      <label><input type="radio" name="poster" value="sheet">
        <span><b>Photo sheet</b><small>
          <select id="per-page" aria-label="Photos per sheet">
            <option value="2">2</option><option value="4" selected>4</option><option value="6">6</option>
            <option value="9">9</option><option value="12">12</option>
          </select> per page, to cut out</small></span></label>
    </div>

    <div class="ai" id="ai-box">
      <label id="ai-on" hidden><input type="checkbox" name="ai"> Sharpen with AI</label>
      <button type="button" class="secondary" id="ai-setup" hidden>Set up AI sharpening</button>
      <label id="max-on" hidden><input type="checkbox" name="max_quality"> Maximum quality</label>
      <button type="button" class="secondary" id="max-setup" hidden>Set up maximum quality</button>
      <small id="ai-note">Makes small images (like Pinterest downloads) print noticeably sharper.</small>
      <small id="max-note" hidden></small>
    </div>

    <details id="crop-box">
      <summary>Choose what gets cropped</summary>
      <p class="erase-help" id="crop-help">Choose one image first.</p>
      <div class="crop-stage" id="crop-stage" hidden>
        <img id="crop-img" alt="The chosen image with a frame showing the part that will be printed." draggable="false">
        <div id="crop-frame" tabindex="0" role="slider" aria-label="Crop position. Drag, or use the arrow keys."
             aria-valuemin="0" aria-valuemax="100" aria-valuenow="50"></div>
      </div>
      <button type="button" class="secondary" id="crop-reset" hidden>Back to automatic</button>
    </details>

    <details id="erase-box">
      <summary>Erase a date stamp or mark</summary>
      <p class="erase-help" id="erase-help">Choose one image first.</p>
      <div class="erase-stage" id="erase-stage" hidden>
        <img id="erase-img" alt="The chosen image. Drag on it to mark areas to erase." draggable="false">
        <div id="erase-rects"></div>
      </div>
      <button type="button" class="secondary" id="erase-clear" hidden>Clear boxes</button>
    </details>

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
            <option value="">None</option><option value="3x2">3x2 sheets</option>
            <option value="2x3">2x3 sheets</option><option value="4x4">4x4 sheets</option>
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
        screen on many printers. <a href="/test-page.pdf" id="cal-link" download>Print the test page</a> at 100%,
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
  // With maximum quality set up, the basic AI setup isn't needed.
  $("ai-setup").hidden = status.ai || !status.ai_installable || status.max;
  $("ai-box").hidden = !status.ai && !status.ai_installable && !status.max_libs;
  $("max-on").hidden = !status.max;
  $("max-setup").hidden = status.max || !status.max_libs;
  const maxNote = $("max-note");
  maxNote.hidden = false;
  maxNote.textContent = status.max
    ? "Maximum quality uses newer AI models: one trained on web photos for photos, Real-ESRGAN for artwork. Slower without a graphics chip."
    : status.max_libs ? "Maximum quality: a one-time 100 MB download of two newer AI models."
    : "Maximum quality (newer AI models) needs PyTorch. To add it, run: pipx inject a4print torch spandrel";
  if (status.ai) $("ai-note").textContent =
    "Makes small images print sharper. Used only when an image needs it; the preview shows a detail at print size.";
}
// Paper: saved choice on this browser, else the setting saved with
// a4print-calibrate, else Letter in the countries that use it.
const PAPER_INFO = __PAPER_INFO__;
const LETTER_REGIONS = ["US", "CA", "MX", "PH", "PR", "CL", "CO", "VE", "GT", "CR", "PA", "DO", "SV", "HN", "NI"];
function remembered(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }
function remember(key, value) { try { localStorage.setItem(key, value); } catch (e) {} }
function showPaper(paper) {
  const info = PAPER_INFO[paper];
  $("paper").value = paper;
  $("size-page").textContent = info.name + " page";
  $("size-page-info").textContent = "One sheet, " + info.page;
  $("size-2x2-info").textContent = "To glue, " + info["2x2"];
  $("size-3x3-info").textContent = "To glue, " + info["3x3"];
  for (const opt of $("poster-more").options)
    if (opt.value) opt.textContent = opt.value + " sheets, " + info[opt.value];
  $("cal-link").href = "/test-page.pdf?paper=" + paper;
}
$("paper").addEventListener("change", () => { remember("a4print-paper", $("paper").value); showPaper($("paper").value); });
function initialPaper(saved) {
  const mine = remembered("a4print-paper");
  if (mine && PAPER_INFO[mine]) return mine;
  if (saved && PAPER_INFO[saved]) return saved;
  const region = ((navigator.language || "").split("-")[1] || "").toUpperCase();
  return LETTER_REGIONS.includes(region) ? "letter" : "a4";
}
showPaper(initialPaper(null));

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
  showPaper(initialPaper((s.print_settings || {}).paper));
  showAI(s);
  showSettings(s);
  // AI is on by default once installed; it skips images that are already sharp.
  if (s.ai) $("ai-on").querySelector("input").checked = true;
});

$("max-setup").addEventListener("click", async () => {
  const btn = $("max-setup");
  btn.disabled = true; btn.textContent = "Downloading (about 100 MB)...";
  try {
    const r = await fetch("/setup-max", {method: "POST", headers: {"X-A4print": "1"}});
    const s = await r.json();
    if (!r.ok) throw new Error(s.error || "setup failed");
    showAI(s);
    $("max-on").querySelector("input").checked = true;
  } catch (e) {
    btn.disabled = false; btn.textContent = "Try maximum quality setup again";
    $("max-note").textContent = "Setup didn't work: " + e.message;
  }
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

// Erase tool: drag boxes on the chosen image. Boxes are kept in percent of
// the image, which is what --remove takes.
let eraseBoxes = [], eraseUrl = null;
function drawEraseBoxes(extra) {
  const layer = $("erase-rects"); layer.replaceChildren();
  for (const b of extra ? eraseBoxes.concat([extra]) : eraseBoxes) {
    const d = document.createElement("div");
    d.style.left = b.x + "%"; d.style.top = b.y + "%"; d.style.width = b.w + "%"; d.style.height = b.h + "%";
    layer.append(d);
  }
  $("erase-clear").hidden = !eraseBoxes.length;
}
// Crop tool: a page-shaped frame on the chosen image. Dragging it fills the
// page with the framed part (sent as fill=crop and crop_at).
let cropPos = {x: 0.5, y: 0.5}, cropTouched = false, cropUrl = null;
const POSTER_OVERLAP = 10;   // mm, the planner's default
function printTarget() {
  // "" for one page, "2x2" style for a poster, null for photo sheets.
  const picked = document.querySelector("input[name=poster]:checked");
  const value = $("poster-more").value || (picked ? picked.value : "");
  return value === "sheet" ? null : value;
}
function singlePage() { return printTarget() !== null; }   // the crop tool covers pages and posters
function cropShape() {
  const img = $("crop-img"), target = printTarget();
  if (!img.naturalWidth || target === null) return null;
  const [pw, ph] = PAPER_INFO[$("paper").value].mm;
  const typed = parseFloat(document.querySelector("input[name=margin]").value);
  const shape = img.naturalWidth / img.naturalHeight;
  const turn = document.querySelector("select[name=orientation]").value;
  let page, lines = {x: [], y: []};
  if (!target) {
    const margin = Math.max(0, typed || 0);
    const o = turn === "auto" ? (shape > 1 ? "landscape" : "portrait") : turn;
    page = o === "landscape" ? (ph - 2 * margin) / (pw - 2 * margin) : (pw - 2 * margin) / (ph - 2 * margin);
  } else {
    // Same choice as the poster planner: portrait or landscape sheets,
    // whichever gives the poster shape closest to the image's.
    const [cols, rows] = target.split("x").map(Number), margin = Math.max(0, isNaN(typed) ? 6 : typed);
    let best = null;
    for (const [sw, sh] of [[pw, ph], [ph, pw]]) {
      const aw = sw - 2 * margin, ah = sh - 2 * margin;
      const w = cols * aw - (cols - 1) * POSTER_OVERLAP, h = rows * ah - (rows - 1) * POSTER_OVERLAP;
      if (turn !== "auto" && (h >= w) !== (turn === "portrait")) continue;
      const off = Math.abs(Math.log(w / h) - Math.log(shape));
      if (!best || off < best.off) best = {off, w, h, aw, ah, cols, rows};
    }
    if (!best) return null;
    page = best.w / best.h;
    for (let c = 1; c < best.cols; c++) lines.x.push((c * (best.aw - POSTER_OVERLAP) + POSTER_OVERLAP / 2) / best.w);
    for (let r = 1; r < best.rows; r++) lines.y.push((r * (best.ah - POSTER_OVERLAP) + POSTER_OVERLAP / 2) / best.h);
  }
  const g = shape > page ? {w: page / shape, h: 1} : {w: 1, h: shape / page};
  g.lines = lines; g.poster = !!target;
  return g;
}
function drawCrop() {
  const one = chosen.length === 1, g = one && singlePage() ? cropShape() : null;
  $("crop-stage").hidden = !g;
  $("crop-reset").hidden = !(g && cropTouched);
  const help = $("crop-help");
  if (!g) {
    help.textContent = !chosen.length ? "Choose one image first." : !one ? "This works with one image at a time."
      : printTarget() === null ? "Choosing the crop works for pages and posters, not photo sheets."
      : $("crop-img").naturalWidth ? "This poster shape isn't possible in that orientation." : help.textContent;
    return;
  }
  const frame = $("crop-frame"), cut = Math.round((1 - g.w * g.h) * 100);
  frame.style.width = g.w * 100 + "%"; frame.style.height = g.h * 100 + "%";
  frame.style.left = cropPos.x * (1 - g.w) * 100 + "%"; frame.style.top = cropPos.y * (1 - g.h) * 100 + "%";
  frame.setAttribute("aria-valuenow", Math.round((g.w < 1 ? cropPos.x : cropPos.y) * 100));
  // On a poster, thin lines show where the sheets meet.
  frame.replaceChildren();
  for (const x of g.lines.x) { const l = el("i"); l.style.left = x * 100 + "%"; l.className = "v"; frame.append(l); }
  for (const y of g.lines.y) { const l = el("i"); l.style.top = y * 100 + "%"; l.className = "h"; frame.append(l); }
  const what = g.poster ? "poster" : "page";
  help.textContent = cut < 1 ? "This image already has the " + what + "'s shape, so nothing gets cropped."
    : cropTouched ? "The print fills the " + what + " with the framed part; " + cut + "% of the image is cut off."
    : "Automatic: artwork is kept whole, and photos are cropped only when little is lost. "
      + "Drag the frame to fill the " + what + " with the part you choose (" + cut + "% would be cut off)."
      + (g.poster ? " The thin lines show where the sheets meet." : "");
}
function setupCrop() {
  cropPos = {x: 0.5, y: 0.5}; cropTouched = false;
  if (cropUrl) { URL.revokeObjectURL(cropUrl); cropUrl = null; }
  $("crop-img").removeAttribute("src");
  if (chosen.length === 1) { cropUrl = URL.createObjectURL(chosen[0]); $("crop-img").src = cropUrl; }
  drawCrop();
}
$("crop-img").addEventListener("load", drawCrop);
$("crop-img").addEventListener("error", () => {
  $("crop-stage").hidden = true;
  $("crop-help").textContent = "This browser can't show this file type, so the crop can't be chosen here.";
});
(() => {
  const stage = $("crop-stage"), frame = $("crop-frame"); let from = null;
  const clamp = v => Math.min(1, Math.max(0, v));
  const move = (dx, dy) => {
    const g = cropShape(); if (!g) return;
    if (g.w < 1) cropPos.x = clamp(cropPos.x + dx / (1 - g.w));
    if (g.h < 1) cropPos.y = clamp(cropPos.y + dy / (1 - g.h));
    cropTouched = true; drawCrop();
  };
  stage.addEventListener("pointerdown", e => { e.preventDefault(); from = {x: e.clientX, y: e.clientY}; stage.setPointerCapture(e.pointerId); frame.focus(); });
  stage.addEventListener("pointermove", e => {
    if (!from) return;
    const r = $("crop-img").getBoundingClientRect();
    move((e.clientX - from.x) / r.width, (e.clientY - from.y) / r.height);
    from = {x: e.clientX, y: e.clientY};
  });
  for (const type of ["pointerup", "pointercancel"]) stage.addEventListener(type, () => { from = null; });
  frame.addEventListener("keydown", e => {
    const step = {ArrowLeft: [-0.02, 0], ArrowRight: [0.02, 0], ArrowUp: [0, -0.02], ArrowDown: [0, 0.02]}[e.key];
    if (step) { e.preventDefault(); move(step[0], step[1]); }
  });
  $("crop-reset").addEventListener("click", () => { cropPos = {x: 0.5, y: 0.5}; cropTouched = false; drawCrop(); });
  for (const sel of ["#paper", "select[name=orientation]", "#poster-more"])
    document.querySelector(sel).addEventListener("change", drawCrop);
  document.querySelector("input[name=margin]").addEventListener("input", drawCrop);
  document.querySelectorAll("input[name=poster]").forEach(r => r.addEventListener("change", drawCrop));
})();

function setupErase() {
  eraseBoxes = []; drawEraseBoxes();
  if (eraseUrl) { URL.revokeObjectURL(eraseUrl); eraseUrl = null; }
  const one = chosen.length === 1;
  $("erase-stage").hidden = !one;
  $("erase-help").textContent = one ? "Drag a box over each thing to erase. It is filled in from its surroundings; "
    + "use it for your own photos or images you may edit." : chosen.length ? "This works with one image at a time."
    : "Choose one image first.";
  if (one) { eraseUrl = URL.createObjectURL(chosen[0]); $("erase-img").src = eraseUrl; }
}
$("erase-img").addEventListener("error", () => {
  $("erase-stage").hidden = true;
  $("erase-help").textContent = "This browser can't show this file type, so boxes can't be drawn on it here.";
});
(() => {
  const stage = $("erase-stage"); let start = null;
  const at = e => {
    const r = $("erase-img").getBoundingClientRect();
    return {x: Math.min(100, Math.max(0, (e.clientX - r.left) / r.width * 100)),
            y: Math.min(100, Math.max(0, (e.clientY - r.top) / r.height * 100))};
  };
  const box = (a, b) => ({x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y)});
  stage.addEventListener("pointerdown", e => { e.preventDefault(); start = at(e); stage.setPointerCapture(e.pointerId); });
  stage.addEventListener("pointermove", e => { if (start) drawEraseBoxes(box(start, at(e))); });
  stage.addEventListener("pointerup", e => {
    if (!start) return;
    const b = box(start, at(e)); start = null;
    if (b.w >= 0.5 && b.h >= 0.5 && eraseBoxes.length < 20) eraseBoxes.push(b);
    drawEraseBoxes();
  });
  stage.addEventListener("pointercancel", () => { start = null; drawEraseBoxes(); });
  $("erase-clear").addEventListener("click", () => { eraseBoxes = []; drawEraseBoxes(); });
})();

function choose(list) {
  chosen = Array.from(list).filter(f => f.size > 0);
  drop.querySelector("strong").textContent =
    chosen.length ? chosen.map(f => f.name).join(", ") : "Drop images here or click to choose";
  go.disabled = !chosen.length;
  setupErase();
  setupCrop();
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
  const box = el("div", {class: "verdict"});
  if (summary.ai_used && summary.dpi < 240) {
    // Separate what the file had from what the print gets.
    box.append(el("b", {class: "ok"}, "Sharpened with AI"),
      document.createTextNode(". The original is " + label.toLowerCase() + " resolution (" + summary.dpi +
        " DPI on this size); edges, lettering and faces now print crisp, and the finest detail is partly " +
        "drawn by the AI. Compare below."));
    return box;
  }
  const b = el("b", {class: cls}, "Print quality: " + label);
  box.append(b, document.createTextNode(" (" + summary.dpi + " DPI). " + sentence));
  if (summary.low && !summary.ai_used) {
    const aiReady = !$("ai-on").hidden, aiSetup = !$("ai-setup").hidden;
    box.append(document.createTextNode(aiReady ? " Tick \"Sharpen with AI\" and make it again."
      : aiSetup ? " Set up AI sharpening above, then make it again." : " A larger version of the image would help."));
  }
  return box;
}

// Before/after comparison at print size: drag (or use the arrow keys) to move the divider.
function slider(beforeUrl, afterUrl, cm) {
  if (Math.abs(cm - Math.round(cm)) <= 0.15) cm = Math.round(cm);   // "5 x 5 cm", not "5.1 x 5.1 cm"
  const wrap = el("figure", {class: "compare"});
  const stage = el("div", {class: "stage"});
  const after = el("img", {src: afterUrl, alt: "a4print result"});
  const before = el("img", {src: beforeUrl, alt: "Plain enlargement", class: "before"});
  const line = el("div", {class: "divider"});
  const range = el("input", {type: "range", min: "0", max: "100", value: "50",
                             "aria-label": "Move the divider between plain enlargement and a4print"});
  const move = v => { before.style.clipPath = "inset(0 " + (100 - v) + "% 0 0)"; line.style.left = v + "%"; };
  range.addEventListener("input", () => move(range.value));
  stage.append(after, before, line, range);
  move(50);
  const cap = el("figcaption");
  cap.append(el("span", {}, "Plain enlargement"), el("span", {}, "Detail at print size (" + cm + " x " + cm + " cm)"),
             el("span", {}, "a4print"));
  wrap.append(stage, cap);
  return wrap;
}

async function makeSheets(params) {
  // Photo sheet: upload every image into one job, then lay them out.
  const box = el("div", {class: "result"});
  box.append(el("h2", {}, chosen.length + " images on photo sheets"), el("p", {class: "what"}, "Uploading..."));
  results.prepend(box);
  const say = text => { box.querySelector(".what").textContent = text; };
  try {
    let job = "";
    for (const [i, file] of chosen.entries()) {
      say("Uploading " + (i + 1) + " of " + chosen.length + "...");
      const q = new URLSearchParams({name: file.name}); if (job) q.set("job", job);
      const r = await fetch("/sheet-add?" + q, {method: "POST", body: file, headers: {"X-A4print": "1"}});
      const d = await r.json(); if (!r.ok) throw new Error(file.name + ": " + d.error); job = d.job;
    }
    say(params.get("ai") === "on" || params.get("max_quality") === "on"
        ? "Laying out and sharpening with AI. This can take a while..." : "Laying out the sheets...");
    params.set("job", job); params.set("per_page", $("per-page").value); params.delete("poster");
    const r = await fetch("/sheet-make?" + params, {method: "POST", headers: {"X-A4print": "1"}});
    const data = await r.json(); if (!r.ok) throw new Error(data.error);
    const s = data.summary;
    say(s.images + " images on " + s.sheets + " " + s.paper + " sheet" + (s.sheets > 1 ? "s" : "") + ", " +
        $("per-page").value + " per page. Cut along the dashed lines. Lowest sharpness: " + s.lowest_dpi + " DPI.");
    const preview = data.files.find(f => f.role === "sheet_preview");
    if (preview) box.append(el("img", {src: preview.url, alt: "First sheet", class: "sheet-preview"}));
    const pdf = data.files.find(f => f.role === "pdf");
    const dl = el("div", {class: "downloads"});
    if (pdf) dl.append(el("a", {class: "main", href: pdf.url, download: pdf.name}, "Download PDF"));
    box.append(dl);
    const more = el("details"); more.append(el("summary", {}, "Details"));
    const lines = el("dl", {class: "lines"});
    for (const [label, text] of data.lines) lines.append(el("dt", {}, label), el("dd", {}, text));
    more.append(lines); box.append(more);
  } catch (err) {
    const p = box.querySelector(".what"); p.className = "error"; p.textContent = err.message || "Something went wrong.";
  }
}

go.addEventListener("click", async () => {
  go.disabled = true;
  const params = new URLSearchParams();
  for (const [key, value] of new FormData($("opts"))) if (value !== "") params.set(key, value);
  if ($("poster-more").value) params.set("poster", $("poster-more").value);
  if (chosen.length === 1 && cropTouched && singlePage()) {
    params.set("fill", "crop");
    params.set("crop_at", (cropPos.x * 100).toFixed(1) + "," + (cropPos.y * 100).toFixed(1));
  }
  if (chosen.length === 1) for (const b of eraseBoxes)
    params.append("remove", [b.x, b.y, b.w, b.h].map(v => v.toFixed(2) + "%").join(","));
  if (params.get("poster") === "sheet") {
    await makeSheets(params);
    go.disabled = false;
    updateAll();
    return;
  }
  for (const file of chosen) {
    const box = el("div", {class: "result"});
    const aiChecked = params.get("ai") === "on" || params.get("max_quality") === "on";
    box.append(el("h2", {}, file.name), el("p", {class: "what"},
      aiChecked ? "Working... AI sharpening can take a while." : "Working..."));
    results.prepend(box);
    box.scrollIntoView({behavior: "smooth", block: "start"});
    params.set("name", file.name);
    // Show elapsed time, so a long AI job doesn't look frozen.
    const waiting = box.querySelector(".what"), base = waiting.textContent, began = Date.now();
    const timer = setInterval(() => {
      const sec = Math.round((Date.now() - began) / 1000);
      waiting.textContent = base + " " + Math.floor(sec / 60) + ":" + String(sec % 60).padStart(2, "0");
    }, 1000);
    try {
      let response, data;
      try {
        response = await fetch("/process?" + params, {method: "POST", body: file, headers: {"X-A4print": "1"}});
        data = await response.json();
      } finally { clearInterval(timer); }
      box.querySelector(".what").remove();
      if (!response.ok) { box.append(el("p", {class: "error"}, data.error || "Something went wrong.")); continue; }
      const s = data.summary;
      box.append(verdict(s, aiChecked));
      box.append(el("p", {class: "what"}, (s.poster ? "Poster: " + s.poster + ". " : "") +
        s.fill.charAt(0).toUpperCase() + s.fill.slice(1) + "." +
        (s.ai_used ? " Sharpened with AI, " + s.ai_note + "." : "")));
      const role = r => data.files.find(f => f.role === r);
      const plain = role("detail_plain"), final = role("detail_final");
      if (plain && final) box.append(slider(plain.url, final.url, s.detail_cm || 5));
      const preview = role("preview");
      if (preview) {
        const whole = el("details", {class: "whole"});
        whole.append(el("summary", {}, "Whole image, before and after"),
                     el("img", {src: preview.url, alt: "Before and after"}));
        box.append(whole);
      }
      const dl = el("div", {class: "downloads"});
      const pdf = role("pdf"), png = role("png");
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


PAGE = PAGE.replace(PAGE_PLACEHOLDER, json.dumps(paper_info()))


if __name__ == "__main__":
    sys.exit(main())
