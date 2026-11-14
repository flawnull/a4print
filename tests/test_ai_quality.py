"""Choosing how hard to apply the AI, and keeping small text intact."""

import sys

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from a4print import enhance, upscale
from a4print.pipeline import Options, detail_pair, prepare

from .conftest import gradient, with_noise


def flat_graphic(small_text=True):
    """A Pinterest-style graphic: flat shapes, a big title, a tiny credit line."""
    img = Image.new("RGB", (236, 320), (238, 226, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 226, 310], outline=(25, 25, 25), width=2)
    d.ellipse([40, 40, 196, 196], fill=(214, 72, 52))
    d.polygon([(30, 250), (118, 150), (206, 250)], fill=(30, 60, 90))
    d.text((32, 262), "SUMMER", fill=(25, 25, 25), font=ImageFont.load_default(size=26))
    if small_text:
        d.text((20, 297), "Designer: example 2025", fill=(40, 40, 40), font=ImageFont.load_default(size=9))
    return np.asarray(img)


def test_content_type():
    assert enhance.content_type(flat_graphic()) == "artwork"
    assert enhance.content_type(with_noise(gradient(600, 400), 6)) == "photo"
    y, x = np.mgrid[0:400, 0:600]
    smooth_photo = np.dstack([128 + 60 * np.sin(x / 40) * np.cos(y / 55)] * 3).astype(np.uint8)
    assert enhance.content_type(with_noise(smooth_photo, 3)) == "photo"


def test_tiny_text_mask_finds_only_the_credit_line():
    mask = enhance.tiny_text_mask(flat_graphic())
    ys, xs = np.nonzero(mask)
    assert ys.min() >= 290 and ys.max() <= 315       # just the credit line
    assert xs.min() <= 25 and xs.max() >= 120        # the whole line
    assert not enhance.tiny_text_mask(flat_graphic(small_text=False)).any()


FAKE = """#!{python}
import sys
import numpy as np
from PIL import Image
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
img = Image.open(args["-i"])
out = np.asarray(img.resize((img.width * 4, img.height * 4))).copy()
out[::7] = 0   # an obvious "AI" signature: stripes everywhere
Image.fromarray(out).save(args["-o"])
"""


@pytest.fixture
def fake_ai(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("the stand-in binary is a Unix script")
    fake = tmp_path / "realesrgan-ncnn-vulkan"
    fake.write_text(FAKE.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv(upscale.ENV_VAR, str(fake))


def test_artwork_gets_full_strength_and_text_protection(fake_ai):
    _, report = prepare(flat_graphic(), Options(ai=True, enhance=False))
    assert report.content == "artwork"
    assert "100%, artwork, small text kept as is" in report.resize_method


def test_photos_get_full_strength(fake_ai):
    _, report = prepare(with_noise(gradient(300, 200), 6), Options(ai=True, enhance=False))
    assert report.content == "photo"
    assert "100%, photo" in report.resize_method


def test_protected_area_is_the_plain_enlargement(fake_ai):
    arr = flat_graphic()
    protect = enhance.tiny_text_mask(arr)
    size = (944, 1280)
    out, _ = upscale.resize(arr, size, ai=True, ai_strength=1.0, protect=protect)
    plain = upscale.lanczos(arr, size).astype(int)
    text_rows = out[1200:1225, 120:450].astype(int)
    shapes = out[300:600, 300:600].astype(int)
    # Inside the credit line: no AI stripes. Elsewhere: the AI result.
    assert np.abs(text_rows - plain[1200:1225, 120:450]).mean() < 3
    assert np.abs(shapes - plain[300:600, 300:600]).mean() > 10


def test_detail_pair_and_preview(tmp_path):
    src = flat_graphic()
    page, report = prepare(src, Options(enhance=False))
    plain, final = report.detail
    assert plain.shape[:2] == final.shape[:2] == (600, 600)
    # The detail sits where the image is busiest, not in empty paper.
    assert final.astype(float).std() > 30

    from a4print.pipeline import process_file
    path = tmp_path / "g.png"
    Image.fromarray(src).save(path)
    process_file(path, tmp_path / "o", Options(pdf=False))
    assert Image.open(tmp_path / "o" / "g_preview.jpg").height > 1200 + 600


def test_detail_pair_handles_tiny_areas():
    assert detail_pair(np.zeros((10, 10, 3), np.uint8), np.zeros((20, 20, 3), np.uint8),
                       ((0, 0, 10, 10), (0, 0, 20, 20))) is None


def _upscaled_pair():
    """A plain enlargement and a fake 'AI' result that brightens everything
    (tone drift), sharpens an edge, and wipes out fine texture."""
    rng = np.random.default_rng(0)
    plain = np.full((400, 400, 3), 120, np.float32)
    plain[:, 200:] = 180
    plain = cv2.GaussianBlur(plain, (0, 0), 6)          # soft edge at x=200
    texture = rng.normal(0, 6, (400, 400, 1)).astype(np.float32)
    plain[:, :150] += texture[:, :150]                  # "skin" texture on the left
    ai = np.full((400, 400, 3), 135, np.float32)         # +15 brighter, no texture
    ai[:, 200:] = 195                                   # hard edge
    return np.clip(plain, 0, 255).astype(np.uint8), np.clip(ai, 0, 255).astype(np.uint8)


def test_merge_keeps_original_tones():
    plain, ai = _upscaled_pair()
    out = upscale.merge_ai(plain, ai, scale=4)
    for x in (60, 330):  # inside the two flat-ish halves
        assert abs(out[100:300, x - 20:x + 20].mean() - plain[100:300, x - 20:x + 20].mean()) < 2


def test_merge_takes_edges_from_ai():
    plain, ai = _upscaled_pair()
    out = upscale.merge_ai(plain, ai, scale=4).astype(int)
    edge = lambda img: np.abs(np.diff(img[200, 185:215, 0])).max()
    assert edge(out) > 2 * edge(plain.astype(int))


def test_merge_restores_texture_the_ai_removed():
    plain, ai = _upscaled_pair()
    out = upscale.merge_ai(plain, ai, scale=4).astype(np.float32)
    region = (slice(100, 300), slice(20, 130))
    hf = lambda img: (img - cv2.GaussianBlur(img, (0, 0), 3))[region].std()
    assert hf(out) > 0.8 * hf(plain.astype(np.float32))
    assert hf(ai.astype(np.float32)) < 0.1


def test_merge_strips_match_whole_image():
    plain, ai = _upscaled_pair()
    whole = upscale.merge_ai(plain, ai, scale=4, rows=10_000).astype(int)
    strips = upscale.merge_ai(plain, ai, scale=4, rows=64).astype(int)
    assert np.abs(whole - strips).max() <= 1
    assert np.array_equal(upscale.merge_ai(plain, ai, scale=4, strength=0), plain)


def test_detail_centers_on_a_face():
    from skimage import data

    portrait = data.astronaut()           # NASA portrait, public domain
    face = enhance.find_face(portrait)
    assert face is not None
    x, y, w, h = face
    assert 150 < x + w / 2 < 300 and 60 < y + h / 2 < 200
    assert enhance.find_face(flat_graphic()) is None
