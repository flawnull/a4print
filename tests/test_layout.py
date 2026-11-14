import numpy as np
import pytest

from a4print import layout


def test_a4_pixel_size():
    assert layout.A4_PORTRAIT == (2480, 3508)
    assert layout.A4_LANDSCAPE == (3508, 2480)


@pytest.mark.parametrize("w,h,override,expected", [
    (600, 900, "auto", "portrait"),
    (1500, 1000, "auto", "landscape"),
    (1000, 1000, "auto", "portrait"),
    (1500, 1000, "portrait", "portrait"),
])
def test_choose_orientation(w, h, override, expected):
    assert layout.choose_orientation(w, h, override) == expected


def test_crop_box_keeps_a4_ratio_and_anchor():
    # Tall source: overflow is vertical.
    x0, y0, x1, y1 = layout.crop_box(1000, 2000, 2480, 3508, "top")
    assert (x0, y0, x1) == (0, 0, 1000)
    assert abs((x1 - x0) / (y1 - y0) - 2480 / 3508) < 0.01

    _, y0, _, y1 = layout.crop_box(1000, 2000, 2480, 3508, "bottom")
    assert y1 == 2000

    # Wide source: overflow is horizontal.
    x0, _, x1, _ = layout.crop_box(3000, 1000, 2480, 3508, "right")
    assert x1 == 3000 and x0 > 0


def test_effective_dpi():
    # 1240 px across 2480 px (one A4 width at 300 DPI) prints at 150 DPI.
    assert layout.effective_dpi(1240, 2480) == pytest.approx(150)


def test_place_on_white_has_white_border(portrait):
    w, h = layout.fit_size(600, 900, 2480, 3508)
    fitted = np.zeros((h, w, 3), np.uint8)
    page = layout.place_on_white(fitted, 2480, 3508)
    assert page.shape == (3508, 2480, 3)
    assert (page[:, 0] == 255).all()       # side border is white
    assert (page[1754, 1240] == 0).all()   # image in the middle


def test_extend_fills_border_from_image():
    fitted = np.zeros((1000, 2480, 3), np.uint8)
    fitted[:] = (30, 120, 200)
    page = layout.extend_background(fitted, 2480, 3508)
    assert page.shape == (3508, 2480, 3)
    # The border continues the image color instead of going white.
    assert np.abs(page[10, 1240].astype(int) - (30, 120, 200)).max() < 5
    # The image area is untouched.
    assert (page[1754, 1240] == (30, 120, 200)).all()


def test_trim_border_removes_beige_margin(portrait):
    page = np.full((1100, 800, 3), (238, 228, 205), np.uint8)
    page[100:1000, 100:700] = portrait
    trimmed, box = layout.trim_border(page)
    assert box == (100, 100, 700, 1000)
    assert trimmed.shape == portrait.shape


def test_trim_border_leaves_busy_edges_alone(portrait):
    trimmed, box = layout.trim_border(portrait)
    assert box == (0, 0, 600, 900)
