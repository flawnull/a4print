"""Free crop position, the web crop parameters, and face-aware cropping."""

import numpy as np
import pytest

from a4print import cli, layout, web
from a4print.pipeline import Options, prepare

from .conftest import gradient


def test_crop_position():
    assert layout.crop_box(3000, 1000, 2480, 3508, position=(0, 0.5))[0] == 0
    assert layout.crop_box(3000, 1000, 2480, 3508, position=(1, 0.5))[2] == 3000
    assert layout.crop_box(3000, 1000, 2480, 3508, position=(0.5, 0.5)) == layout.crop_box(3000, 1000, 2480, 3508)
    assert layout.crop_box(1000, 3000, 3508, 2480, position=(0.5, 1))[3] == 3000
    assert layout.crop_box(3000, 1000, 2480, 3508, position=(7, -2))[2] == 3000      # clamped


@pytest.mark.parametrize("text,expected", [("30,70", (0.3, 0.7)), ("0%, 100%", (0.0, 1.0))])
def test_crop_at_flag(text, expected):
    assert cli.build_parser().parse_args(["x.png", "--crop-at", text]).crop_at == expected


@pytest.mark.parametrize("text", ["50", "a,b", "120,50", "-1,0"])
def test_bad_crop_at(text):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["x.png", "--crop-at", text])


def test_crop_at_fills_the_page_with_that_part():
    wide = gradient(1500, 500)                      # red rises left to right
    left, r1 = prepare(wide, Options(crop_at=(0.0, 0.5), orientation="portrait", enhance=False))
    right, r2 = prepare(wide, Options(crop_at=(1.0, 0.5), orientation="portrait", enhance=False))
    assert r1.fill == r2.fill == "crop"            # auto fill gives way to the chosen crop
    assert left.shape == (3508, 2480, 3)
    assert left[..., 0].mean() + 100 < right[..., 0].mean()


def test_web_passes_crop_at():
    argv = web.build_argv("in.png", "out", {"fill": ["crop"], "crop_at": ["100.0,50.0"]})
    opts, error = web.parse_options(argv)
    assert not error and opts.crop_at == (1.0, 0.5) and opts.fill == "crop"
    assert web.paper_info()["letter"]["mm"] == [215.9, 279.4]


def test_automatic_crop_keeps_a_face_in_the_frame():
    from skimage import data

    portrait = data.astronaut()                     # NASA, public domain; face around x 180-270
    wide = np.full((512, 1400, 3), (120, 140, 160), np.uint8)
    wide[:, :512] = portrait                        # the person is at the far left
    page, report = prepare(wide, Options(fill="crop", orientation="portrait", enhance=False))
    assert "faces" in report.crop_note
    # A centered crop would show only the plain background.
    assert page[:, :, 0].std() > 30

    # Choosing a side or a position yourself switches this off.
    _, report = prepare(wide, Options(fill="crop", orientation="portrait", anchor="right", enhance=False))
    assert report.crop_note == ""
    _, report = prepare(wide, Options(crop_at=(0.5, 0.5), orientation="portrait", enhance=False))
    assert report.crop_note == ""


def test_no_faces_means_a_centered_crop():
    _, report = prepare(gradient(1500, 500), Options(fill="crop", orientation="portrait", enhance=False))
    assert report.crop_note == ""
