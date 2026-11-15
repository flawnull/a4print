"""The protections described in SECURITY.md."""

import math

import pytest
from PIL import Image

from a4print import calibrate, files, maxq, setup_ai

from .test_web import get, png_bytes, post, server  # noqa: F401  (server is a fixture)

# --- untrusted images ---------------------------------------------------

def test_oversized_images_are_refused_before_decoding(tmp_path, monkeypatch):
    path = tmp_path / "big.png"
    Image.new("RGB", (300, 300)).save(path)
    monkeypatch.setattr(files, "MAX_PIXELS", 50_000)

    def no_decoding(*args, **kwargs):
        raise AssertionError("pixels were decoded")

    monkeypatch.setattr(files, "_to_srgb_array", no_decoding)
    with pytest.raises(ValueError, match="too large"):
        files.load_rgb(path)


# --- web page ------------------------------------------------------------

def test_every_response_has_protective_headers(server, portrait):  # noqa: F811
    for path in ("/", "/status", "/files/0000000000000000/x.png", "/nothing"):
        _, headers, _ = get(server + path)
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Cache-Control"] == "no-store"
    _, headers, _ = get(server + "/")
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


def test_foreign_origin_is_refused(server, portrait):  # noqa: F811
    status, _ = post(server + "/process?name=a.png", png_bytes(portrait), {"Origin": "https://evil.example"})
    assert status == 403
    status, _ = post(server + "/setup-ai", b"", {"Origin": "https://evil.example"})
    assert status == 403
    status, _ = post(server + "/process?name=a.png", png_bytes(portrait), {"Origin": server})
    assert status == 200


def test_malformed_length_is_rejected(server):  # noqa: F811
    import http.client
    from urllib.parse import urlparse

    u = urlparse(server)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    conn.putrequest("POST", "/process?name=a.png")
    conn.putheader("X-A4print", "1")
    conn.putheader("Content-Length", "not-a-number")
    conn.endheaders()
    assert conn.getresponse().status == 413


# --- downloads -------------------------------------------------------------

def test_only_https_downloads(tmp_path):
    with pytest.raises(setup_ai.SetupError, match="non-https"):
        setup_ai.download("http://example.com/model.pth", tmp_path / "x", "0" * 64)


def test_oversized_downloads_are_stopped(tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 5000)
    with pytest.raises(setup_ai.SetupError, match="larger than expected"):
        setup_ai.download(big.as_uri(), tmp_path / "out", "0" * 64, max_bytes=1000)


def test_changed_model_file_is_refused(tmp_path):
    m = maxq.MODELS["photo"]
    maxq.models_dir().mkdir(parents=True, exist_ok=True)
    (maxq.models_dir() / m.file).write_bytes(b"not the tested model")
    maxq._loaded.pop("photo", None)
    with pytest.raises(maxq.MaxQualityUnavailable, match="checksum"):
        maxq._model("photo")


# --- saved settings ----------------------------------------------------------

@pytest.mark.parametrize("content,expected", [
    ('{"shadows": NaN, "midtones": 0.1}', {"midtones": 0.1}),
    ('{"shadows": 99, "midtones": -5}', {"shadows": 0.2, "midtones": 0.0}),
    ('{"shadows": "lots", "midtones": [1]}', {}),
    ('[1, 2, 3]', {}),
])
def test_damaged_settings_are_made_safe(content, expected):
    calibrate.settings_path().parent.mkdir(parents=True, exist_ok=True)
    calibrate.settings_path().write_text(content)
    settings = calibrate.load_settings()
    assert settings == expected
    assert all(math.isfinite(v) for v in settings.values())
