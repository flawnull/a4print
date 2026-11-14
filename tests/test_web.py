"""The local web page, driven over real HTTP."""

import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image

from a4print import web


@pytest.fixture
def server(tmp_path):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(web.App(tmp_path)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def png_bytes(arr):
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, "PNG")
    return buffer.getvalue()


def post(url, data, headers=None):
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"X-A4print": "1", **(headers or {})})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        body = err.read()
        return err.code, (json.loads(body) if body.startswith(b"{") else {})


def get(url, headers=None):
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers, b""


def test_page_loads(server):
    status, headers, body = get(server + "/")
    assert status == 200 and b"Drop images here" in body
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_upload_makes_a4_files(server, portrait):
    status, data = post(server + "/process?name=holiday.png&fill=fit", png_bytes(portrait))
    assert status == 200
    kinds = sorted(f["kind"] for f in data["files"])
    assert kinds == ["jpg", "pdf", "png"]
    assert any(label == "page" and "A4 portrait, fit" in text for label, text in data["lines"])

    pdf = next(f for f in data["files"] if f["kind"] == "pdf")
    status, headers, body = get(server + pdf["url"])
    assert status == 200 and body.startswith(b"%PDF")
    assert "attachment" in headers["Content-Disposition"]


def test_upload_makes_poster(server, landscape):
    status, data = post(server + "/process?name=wall.png&poster=2x2", png_bytes(landscape))
    assert status == 200
    assert sorted(f["name"] for f in data["files"]) == ["wall_poster.pdf", "wall_poster_preview.jpg"]


def test_bad_options_use_the_cli_messages(server, portrait):
    status, data = post(server + "/process?name=a.png&poster=9x9", png_bytes(portrait))
    assert status == 400 and "poster needs" in data["error"]


def test_rejects_non_images(server):
    status, data = post(server + "/process?name=notes.txt", b"hello")
    assert status == 400
    status, data = post(server + "/process?name=fake.jpg", b"hello")
    assert status == 422 and "not an image" in data["error"]


def test_requires_the_page_header(server, portrait):
    """Other websites can't send this header without a CORS preflight."""
    status, _ = post(server + "/process?name=a.png", png_bytes(portrait), {"X-A4print": "0"})
    assert status == 403


def test_rejects_foreign_host_header(server):
    """Protects against DNS rebinding: a remote site pointing its name at 127.0.0.1."""
    status, _, _ = get(server + "/", {"Host": "evil.example:8765"})
    assert status == 403


def test_files_cannot_escape_the_job_folder(server, portrait):
    status, data = post(server + "/process?name=a.png", png_bytes(portrait))
    job = data["files"][0]["url"].split("/")[2]
    for path in (f"/files/{job}/..%2F..%2Fetc%2Fpasswd", f"/files/{job}/upload", "/files/nothex/a_a4.png"):
        assert get(server + path)[0] == 404


def test_safe_name():
    assert web.safe_name("../../my photo (1).JPG") == "my_photo_1_.JPG"
    assert web.safe_name("") == "image"


def test_hidden_options_stay_hidden():
    """label.check sets display: flex, which would override the hidden
    attribute and show the AI option when Real-ESRGAN isn't installed."""
    assert "[hidden] { display: none !important; }" in web.PAGE


def test_status_reports_ai(server, monkeypatch):
    status, _, body = get(server + "/status")
    assert status == 200 and json.loads(body)["ai"] in (True, False)


def test_result_has_a_plain_summary(server, portrait):
    status, data = post(server + "/process?name=a.png", png_bytes(portrait))
    s = data["summary"]
    assert s["quality"] == "low" and s["low"] is True and s["ai_used"] is False
    assert s["dpi"] > 0 and s["fill"]


def test_setup_ai_from_the_page(tmp_path, monkeypatch, portrait):
    """The page's setup button installs, self-tests and turns AI on."""
    from a4print import setup_ai

    calls = []
    monkeypatch.setattr(setup_ai, "install", lambda *a, **k: calls.append("install"))
    monkeypatch.setattr(setup_ai, "self_test", lambda: "")
    monkeypatch.setattr(setup_ai, "supported", lambda: True)
    app = web.App(tmp_path)
    app.ai_available = False
    status, payload = app.setup_ai()
    assert status == 200 and payload["ai"] is True and calls == ["install"]


def test_setup_ai_reports_failures(tmp_path, monkeypatch):
    from a4print import setup_ai

    def broken(*a, **k):
        raise setup_ai.SetupError("couldn't download Real-ESRGAN: offline")

    monkeypatch.setattr(setup_ai, "install", broken)
    app = web.App(tmp_path)
    app.ai_available = False
    status, payload = app.setup_ai()
    assert status == 502 and "offline" in payload["error"] and payload["ai"] is False


def test_setup_ai_needs_the_page_header(server):
    status, _ = post(server + "/setup-ai", b"", {"X-A4print": "0"})
    assert status == 403


def test_download_all_combines_results(server, portrait, landscape):
    import pikepdf

    jobs = []
    for name, arr, extra in (("a.png", portrait, ""), ("b.png", landscape, "&poster=2x2")):
        _, data = post(server + f"/process?name={name}{extra}", png_bytes(arr))
        jobs.append(data["files"][0]["url"].split("/")[2])
    status, headers, body = get(server + "/all.pdf?jobs=" + ",".join(jobs))
    assert status == 200 and "a4print_all.pdf" in headers["Content-Disposition"]
    with pikepdf.open(io.BytesIO(body)) as pdf:
        assert len(pdf.pages) == 1 + 4                      # one A4 page + a 2x2 poster
        assert pdf.Root.ViewerPreferences.PrintScaling == pikepdf.Name.None_
    assert get(server + "/all.pdf?jobs=nothex,../x")[0] == 404


def test_old_jobs_are_deleted(tmp_path, portrait, monkeypatch):
    monkeypatch.setattr(web, "MAX_JOBS", 3)
    app = web.App(tmp_path)
    for i in range(5):
        status, _ = app.process(f"p{i}.png", png_bytes(portrait), {"no_enhance": ["1"]})
        assert status == 200
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 3


def test_calibration_from_the_page(server):
    status, _, body = get(server + "/test-page.pdf")
    assert status == 200 and body.startswith(b"%PDF")
    status, data = post(server + "/settings?black=5&gray=C", b"")
    assert status == 200 and data["print_settings"] == {"shadows": round(16 / 255, 3), "midtones": 0.1}
    status, data = post(server + "/settings?black=12&gray=C", b"")
    assert status == 400
    status, _ = post(server + "/settings?black=5&gray=C", b"", {"X-A4print": "0"})
    assert status == 403
