import numpy as np
import pytest
from PIL import Image

from a4print import cli, poster
from a4print.layout import DPI
from a4print.pipeline import Options, prepare_poster

from .conftest import gradient


def test_parse_grid():
    assert poster.parse_grid("2x3") == (2, 3)
    assert poster.parse_grid("3X3") == (3, 3)
    for bad in ("1x1", "2", "axb", "7x2", "0x3"):
        with pytest.raises(ValueError):
            poster.parse_grid(bad)


def test_plan_sizes_and_names():
    plan = poster.make_plan(3000, 2000, 2, 2, margin_mm=6, overlap_mm=10)
    assert plan.sheet_orientation == "landscape"
    # 2 printed areas of 285 mm minus one 10 mm overlap = 560 mm wide.
    assert plan.poster_mm == pytest.approx((560, 386))
    assert plan.size_name == "about A2"
    assert poster.make_plan(2000, 3000, 3, 3).size_name == "about A1"


def test_plan_picks_sheet_direction_closest_to_image():
    tall = poster.make_plan(1000, 1500, 2, 2)
    wide = poster.make_plan(1500, 1000, 2, 2)
    assert tall.sheet_orientation == "portrait"
    assert wide.sheet_orientation == "landscape"


def test_plan_orientation_override():
    plan = poster.make_plan(1500, 1000, 2, 2, orientation="portrait")
    assert plan.poster_mm[1] > plan.poster_mm[0]
    with pytest.raises(ValueError, match="try 1x3"):
        poster.make_plan(1000, 1000, 3, 1, orientation="portrait")


def test_big_posters_render_below_300_dpi():
    plan = poster.make_plan(3000, 2000, 5, 5)
    w, h = plan.render_size
    assert plan.render_dpi < DPI
    assert w * h <= poster.MAX_POSTER_PIXELS * 1.01


def assemble(sheets, plan):
    """Put sheets together the way a person would: trim each sheet's margins
    and lay later sheets over earlier ones' overlap strips."""
    m = poster._mm(plan.margin_mm)
    aw, ah = poster._mm(plan.area_mm[0]), poster._mm(plan.area_mm[1])
    w, h = plan.render_size
    canvas = np.zeros((h + ah, w + aw, 3), np.uint8)
    for i, sheet in enumerate(sheets):
        row, col = divmod(i, plan.cols)
        x0, y0, _, _ = poster._area_box(plan, row, col, DPI)
        canvas[y0:y0 + ah, x0:x0 + aw] = sheet[m:m + ah, m:m + aw]
    return canvas[:h, :w]


@pytest.mark.parametrize("cols,rows", [(2, 2), (3, 2), (2, 3)])
def test_sheets_reassemble_into_the_poster(cols, rows):
    """Every poster pixel survives, and the dashed guide lines end up hidden
    under the neighboring sheet."""
    plan = poster.make_plan(1200, 900, cols, rows)
    assert plan.render_dpi == DPI
    w, h = plan.render_size
    art = gradient(w, h)
    art[::97] = (0, 0, 0)  # thin lines, so any misalignment shows
    art[:, ::89] = (255, 255, 255)

    sheets = poster.slice_sheets(art, plan)
    assert len(sheets) == cols * rows
    sheet_shape = (poster._mm(plan.sheet_mm[1]), poster._mm(plan.sheet_mm[0]), 3)
    assert all(s.shape == sheet_shape for s in sheets)

    rebuilt = assemble(sheets, plan)
    assert np.array_equal(rebuilt[:-2, :-2], art[:-2, :-2])


def test_sheets_have_white_margins_and_labels():
    plan = poster.make_plan(1200, 900, 2, 2)
    sheets = poster.slice_sheets(gradient(*plan.render_size), plan)
    m = poster._mm(plan.margin_mm)
    first = sheets[0]
    assert (first[:m - 10] == 255).all()          # top margin is blank
    bottom = first[-m:]
    assert (bottom < 200).any()                    # the label is printed there


def test_poster_cli_writes_multipage_pdf(tmp_path, save, landscape):
    pikepdf = pytest.importorskip("pikepdf")
    out = tmp_path / "out"
    assert cli.main([str(save(landscape, "art.png")), "-o", str(out), "--poster", "3x2"]) == 0
    with pikepdf.open(out / "art_poster.pdf") as pdf:
        sizes = [tuple(round(float(p.mediabox[i]) / 72 * 25.4) for i in (2, 3)) for p in pdf.pages]
    assert len(sizes) == 6
    assert set(sizes) in ({(297, 210)}, {(210, 297)})
    assert (out / "art_poster_preview.jpg").exists()
    assert not (out / "art_a4.png").exists()


def test_poster_keeps_bw_neutral(bw_photo):
    sheets, _, report = prepare_poster(bw_photo, Options(poster=(2, 2)))
    assert report.grayscale
    for sheet in sheets:
        assert (sheet[..., 0] == sheet[..., 1]).all() and (sheet[..., 1] == sheet[..., 2]).all()


def test_poster_resolution_is_for_the_big_size(landscape):
    _, _, a4_like = prepare_poster(landscape, Options(poster=(2, 2)))
    _, _, bigger = prepare_poster(landscape, Options(poster=(3, 3)))
    assert bigger.effective_dpi < a4_like.effective_dpi
    assert a4_like.poster.size_name == "about A2"


def test_bad_poster_flag():
    with pytest.raises(SystemExit):
        cli.main(["x.png", "--poster", "9x9"])
