"""The default fill: keep designed artwork whole, crop photos only when
little is lost."""

import numpy as np

from a4print import layout
from a4print.pipeline import Options, prepare

from .conftest import gradient

PAPER = (226, 221, 204)


def designed_poster(w=736, h=980, grain=4, seed=3):
    """Like a Pinterest poster: textured paper, a frame line near the edge,
    dark artwork inside."""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)
    img[:] = PAPER
    img += rng.normal(0, grain, (h, w, 1))
    img = np.clip(img, 0, 255).astype(np.uint8)
    img[14:17, 14:w - 14] = 30          # frame line, top
    img[h - 43:h - 40, 14:w - 14] = 30  # bottom
    img[14:h - 40, 14:17] = 30          # left
    img[14:h - 40, w - 17:w - 14] = 30  # right
    img[40:500, 40:w - 40] = (40, 40, 40)
    return img


def test_border_color_and_grain():
    color, grain = layout.border_color(designed_poster())
    assert np.abs(np.array(color) - PAPER).max() <= 3
    assert 2.5 < grain < 6
    assert layout.border_color(gradient(600, 400)) is None


def test_designed_poster_is_kept_whole():
    poster = designed_poster()
    page, report = prepare(poster, Options(enhance=False))
    assert page.shape == (3508, 2480, 3)
    assert report.fill == "fit" and "border color" in report.fill_reason
    # Both side frame lines survive (cropping would cut them off).
    middle = page[1754]
    dark = np.flatnonzero(middle.mean(axis=1) < 90)
    assert dark.min() < 120 and dark.max() > 2480 - 120
    # The added space at the top is paper colored, not white.
    assert np.abs(page[20, 1240].astype(int) - PAPER).max() < 20


def test_padding_matches_the_paper_grain():
    poster = designed_poster(grain=6)
    page, _ = prepare(poster, Options(enhance=False))
    added = page[5:60, 400:2000].astype(float).std()
    paper = page[1500:1700, 60:100].astype(float).std()   # paper inside the frame line
    assert abs(added - paper) < 0.5 * paper


def test_photo_with_small_mismatch_is_cropped():
    _, report = prepare(gradient(1600, 1000), Options())
    assert report.fill == "crop"


def test_photo_with_big_mismatch_is_kept_whole_on_white():
    page, report = prepare(gradient(1000, 1000), Options(enhance=False))
    assert report.fill == "fit"
    assert (page[:100] == 255).all()


def test_pad_to_shape():
    out = layout.pad_to_shape(np.zeros((100, 300, 3), np.uint8), 2480, 3508, (200, 190, 180))
    assert abs(out.shape[1] / out.shape[0] - 2480 / 3508) < 0.01
    assert out.shape[1] == 300
