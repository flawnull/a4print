"""Print grain on strongly enlarged photos."""

import numpy as np

from a4print import enhance
from a4print.pipeline import Options, prepare

from .conftest import gradient, with_noise


def test_grain_amount_rises_with_enlargement():
    assert enhance.grain_amount(1.5) == 0 and enhance.grain_amount(2) == 0
    assert 0 < enhance.grain_amount(3) < enhance.grain_amount(4) < enhance.grain_amount(5) == 3
    assert enhance.grain_amount(12) == 3


def test_grain_is_fine_mid_tone_and_the_right_strength():
    flat = np.full((1500, 900, 3), 128, np.uint8)
    out = enhance.add_grain(flat, 3.0).astype(float)
    assert abs(out.mean() - 128) < 0.2
    assert 2.4 < out[..., 0].std() < 3.4
    assert np.array_equal(out[..., 0], out[..., 1])           # luminance only: no color speckles
    white = enhance.add_grain(np.full((300, 300, 3), 255, np.uint8), 3.0)
    assert (white == 255).all()                                # paper white stays clean
    # Strips join without a seam: neighboring rows across a strip edge
    # differ no more than rows elsewhere.
    jump = np.abs(np.diff(out[..., 0], axis=0)).mean(axis=1)
    assert jump[1023] < 2 * np.median(jump)


def test_grain_is_the_same_every_run():
    arr = np.full((200, 200, 3), 100, np.uint8)
    assert np.array_equal(enhance.add_grain(arr, 2), enhance.add_grain(arr, 2))


def small_photo():
    return with_noise(gradient(240, 170), 4)


def test_strongly_enlarged_photos_get_grain():
    _, report = prepare(small_photo(), Options())
    assert report.content == "photo" and report.grain
    _, report = prepare(small_photo(), Options(grain=False))
    assert not report.grain
    _, report = prepare(small_photo(), Options(enhance=False))
    assert not report.grain


def test_no_grain_on_artwork_or_large_images():
    from .test_ai_quality import flat_graphic

    _, report = prepare(flat_graphic(), Options())
    assert report.content == "artwork" and not report.grain
    _, report = prepare(with_noise(gradient(1800, 1270), 4), Options())
    assert not report.grain
