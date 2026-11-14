import numpy as np

from a4print import enhance

from .conftest import gradient, with_noise


def test_grayscale_detection(portrait, bw_photo):
    assert enhance.is_grayscale(bw_photo)
    assert not enhance.is_grayscale(portrait)


def test_noise_estimate_and_auto_skip():
    clean = gradient(800, 800)
    noisy = with_noise(clean, 8)
    assert enhance.estimate_noise(noisy) > 4 * enhance.estimate_noise(clean)

    out, _, applied = enhance.denoise(clean)
    assert not applied and out is clean

    out, _, applied = enhance.denoise(noisy)
    assert applied
    assert enhance.estimate_noise(out) < enhance.estimate_noise(noisy)


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
