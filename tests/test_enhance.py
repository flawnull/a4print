import numpy as np
import pytest

from a4print import enhance

from .conftest import gradient, with_noise


def test_grayscale_detection(portrait, bw_photo):
    assert enhance.is_grayscale(bw_photo)
    assert not enhance.is_grayscale(portrait)


def test_noise_estimate_and_auto_skip():
    clean = gradient(800, 800)
    noisy = with_noise(clean, 8)
    assert enhance.estimate_noise(noisy) > 4 * enhance.estimate_noise(clean)

    out, reason = enhance.denoise(clean)
    assert reason == "" and out is clean

    out, reason = enhance.denoise(noisy)
    assert "noise" in reason
    assert enhance.estimate_noise(out) < enhance.estimate_noise(noisy)


def test_texture_is_not_mistaken_for_noise():
    """Fine, clean detail (fabric, hair, foliage) must not trigger denoise."""
    y, x = np.mgrid[0:800, 0:800]
    texture = 128 + 60 * np.sign(np.sin(x / 1.7) * np.sin(y / 2.3))
    texture = np.dstack([texture] * 3).astype(np.uint8)
    texture[:, :400] = 140  # a flat area, like sky or a wall
    _, reason = enhance.denoise(texture)
    assert reason == ""


def test_denoise_keeps_some_grain():
    noisy = with_noise(gradient(600, 600), 8)
    full, _ = enhance.denoise(noisy, keep_grain=0)
    gentle, _ = enhance.denoise(noisy)
    assert enhance.estimate_noise(full) < enhance.estimate_noise(gentle) < enhance.estimate_noise(noisy)


def test_low_jpeg_quality_triggers_cleanup():
    clean = gradient(400, 400)
    _, reason = enhance.denoise(clean, jpeg_quality=45)
    assert "JPEG quality 45" in reason
    _, reason = enhance.denoise(clean, jpeg_quality=92)
    assert reason == ""


def test_small_color_logo_keeps_image_in_color(bw_photo):
    img = bw_photo.copy()
    img[20:50, 20:70] = (220, 30, 30)  # 0.2% of the image
    assert not enhance.is_grayscale(img)


def test_tone_curve_lifts_shadows_keeps_white():
    lut = enhance.tone_curve(shadows=0.03, midtones=0.05)
    assert 5 <= lut[0] <= 12
    assert lut[255] == 255
    assert lut[128] > 128
    assert (np.diff(lut.astype(int)) >= 0).all()


def test_finish_keeps_gray_neutral(bw_photo):
    gray = enhance.to_gray(bw_photo)
    out = enhance.finish_for_print(gray)
    assert out.ndim == 2


def test_finish_keeps_flat_colors():
    flat = np.zeros((200, 200, 3), np.uint8)
    flat[:] = (200, 60, 40)
    out = enhance.finish_for_print(flat, shadows=0, midtones=0)
    assert np.abs(out.astype(int) - flat).max() <= 3


def test_inpaint_removes_text_box():
    img = np.zeros((300, 400, 3), np.uint8)
    img[:] = (90, 140, 60)
    img[200:230, 250:380] = 255  # a white "date stamp"
    out = enhance.inpaint_regions(img, [(250, 200, 130, 30)])
    assert np.abs(out[215, 315].astype(int) - (90, 140, 60)).max() < 10


def _poster_with_credit():
    """Paper with a dark frame line and a small credit line just below it."""
    from PIL import Image, ImageDraw, ImageFont

    rng = np.random.default_rng(0)
    paper = np.clip(np.array([226, 221, 204]) + rng.normal(0, 4, (400, 600, 1)), 0, 255).astype(np.uint8)
    img = Image.fromarray(paper)
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 589, 360], outline=(30, 30, 30), width=3)
    d.text((20, 372), "Designer: example  9/25/25", fill=(40, 40, 40), font=ImageFont.load_default(size=13))
    return np.asarray(img)


def test_erasing_text_keeps_the_neighboring_frame_line():
    """Filling the whole box smeared the frame line into it; only the text
    strokes are filled now."""
    img = _poster_with_credit()
    box = (12, 363, 300, 30)                       # touches the frame line above
    out = enhance.inpaint_regions(img, [box]).astype(int)
    band = out[368:392, 15:300]
    assert band.min() > 150                         # no text, no dark smear
    assert np.abs(out[358:361, 50:550] - img[358:361, 50:550]).max() <= 2      # frame line untouched
    untouched = np.abs(out[363:393, 12:312] - img[363:393, 12:312]).max(axis=2) == 0
    assert untouched.mean() > 0.6                   # most of the paper grain is original


@pytest.mark.parametrize("box", [(250, 200, 130, 30), (240, 190, 150, 50), (200, 150, 230, 130)])
def test_erasing_a_solid_patch_with_tight_or_loose_boxes(box):
    img = np.zeros((300, 400, 3), np.uint8)
    img[:] = (90, 140, 60)
    img[200:230, 250:380] = 255
    out = enhance.inpaint_regions(img, [box])
    assert np.abs(out.astype(int) - (90, 140, 60)).max() < 12


def test_erasing_works_on_gray_images():
    gray = enhance.to_gray(_poster_with_credit())
    out = enhance.inpaint_regions(gray, [(12, 363, 300, 30)])
    assert out.ndim == 2 and out[368:392, 15:300].min() > 150
