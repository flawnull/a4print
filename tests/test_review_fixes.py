"""Regression tests for the bugs found in the code review."""

import numpy as np
import pikepdf
import pytest
from PIL import Image

from a4print import cli, enhance, files, sheets, web
from a4print.pipeline import Options, prepare, prepare_poster

from .conftest import gradient
from .test_web import get, png_bytes, post, server  # noqa: F401  (server is a fixture)


@pytest.mark.parametrize("name,expected", [
    ("фото.jpg", "image.jpg"),                    # no ASCII letters: keep the extension
    ("写真.PNG", "image.PNG"),
    ("x" * 125 + ".jpg", "x" * 80 + ".jpg"),      # long names keep it too
    ("../../etc/passwd", "passwd"),
])
def test_safe_name_keeps_the_extension(name, expected):
    assert web.safe_name(name) == expected


def test_non_ascii_and_long_names_upload(server, portrait):  # noqa: F811
    from urllib.parse import quote

    for name in ("фото.png", "y" * 125 + ".png"):
        status, data = post(server + "/process?no_enhance=1&name=" + quote(name), png_bytes(portrait))
        assert status == 200
        for f in data["files"]:                    # every output can be fetched, long names included
            assert get(server + f["url"])[0] == 200


def test_same_name_in_two_folders(tmp_path, portrait, landscape):
    for folder, arr in (("a", portrait), ("b", landscape)):
        (tmp_path / folder).mkdir()
        Image.fromarray(arr).save(tmp_path / folder / "photo.jpg")
    out = tmp_path / "out"
    assert cli.main([str(tmp_path / "a"), str(tmp_path / "b"), "-o", str(out), "--no-pdf", "--no-preview"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["photo_jpg_2_a4.png", "photo_jpg_a4.png"]


def test_outputs_of_every_paper_are_not_reprocessed(tmp_path):
    for name in ("cat.jpg", "cat_letter.png", "cat_a3.png", "cat_detail_final.jpg", "film_poster.jpg"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in files.list_images(tmp_path)] == ["cat.jpg", "film_poster.jpg"]


def test_sheet_errors_are_reported_not_raised(tmp_path, save, portrait, capsys):
    path = str(save(portrait, "p.png"))
    assert cli.main([path, "-o", str(tmp_path / "o"), "--per-page", "4", "--margin", "150"]) == 2
    assert "no room" in capsys.readouterr().err
    assert cli.main([path, "-o", str(tmp_path / "o"), "--per-page", "2", "--remove", "5000,5000,10,10"]) == 1
    assert "outside" in capsys.readouterr().err


def test_one_bad_image_does_not_lose_the_sheet():
    imgs = [("ok.png", gradient(600, 400), None), ("ok2.png", gradient(500, 400), None)]
    opts = Options(enhance=False, remove=[(550, 10, 20, 20)])       # outside the 500 px wide image only
    pages, entries, failed = sheets.make_sheets(imgs, opts, 2)
    assert len(pages) == 1 and [e[1] for e in entries] == ["ok.png"]
    assert failed and failed[0][0] == "ok2.png"


def test_grid_reports_unreadable_files(tmp_path, save, portrait, capsys):
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    good = save(portrait, "good.png")
    assert cli.main([str(bad), str(good), "-o", str(tmp_path / "o"), "--grid"]) == 1
    assert "not an image" in capsys.readouterr().err
    assert (tmp_path / "o" / "good_grid.jpg").exists()


def test_negative_margin_is_rejected_everywhere(server, portrait):  # noqa: F811
    with pytest.raises(ValueError, match="negative"):
        prepare(portrait, Options(margin=-5))
    with pytest.raises(ValueError, match="negative"):
        prepare_poster(portrait, Options(poster=(2, 2), margin=-5))
    with pytest.raises(ValueError, match="negative"):
        sheets.make_sheets([("a", portrait, None)], Options(margin=-5), 2)
    status, data = post(server + "/process?name=a.png&margin=-5", png_bytes(portrait))
    assert status == 422 and "negative" in data["error"]


def test_detail_size_follows_the_render_dpi(landscape):
    _, report = prepare(landscape, Options(enhance=False))
    assert report.detail_cm == pytest.approx(5.1, abs=0.1)          # 600 px at 300 DPI
    _, _, report = prepare_poster(landscape, Options(poster=(4, 4), paper="a3", enhance=False))
    assert report.poster.render_dpi < 300
    assert report.detail_cm == pytest.approx(600 / report.poster.render_dpi * 2.54, abs=0.1)
    assert report.detail_cm > 8


def test_other_poster_sizes_follow_the_paper():
    info = web.paper_info()
    assert info["a4"]["4x4"].startswith("111 x 76 cm")
    assert info["a3"]["4x4"].startswith("160 x 111 cm")
    assert info["letter"]["3x2"].endswith("in")
    assert 'info[opt.value]' in web.PAGE


def test_web_test_page_matches_the_paper(server):  # noqa: F811
    for paper, inches in (("letter", [8.5, 11.0]), ("a4", [8.27, 11.69])):
        status, _, body = get(server + f"/test-page.pdf?paper={paper}")
        assert status == 200
        import io
        with pikepdf.open(io.BytesIO(body)) as pdf:
            assert [round(float(v) / 72, 2) for v in pdf.pages[0].mediabox[2:4]] == inches


@pytest.mark.parametrize("shape", [(1, 3000), (3000, 1)])
def test_very_thin_images(shape):
    img = np.full(shape + (3,), 120, np.uint8)
    page, _ = prepare(img, Options())
    assert page.shape[:2] in ((3508, 2480), (2480, 3508))


def test_tiny_percent_box_on_a_small_image():
    assert enhance.resolve_region(("10%", "10%", "0.5%", "0.5%"), 60, 60) == (6, 6, 1, 1)
    out = enhance.inpaint_regions(np.full((60, 60, 3), 100, np.uint8), [("10%", "10%", "0.5%", "0.5%")])
    assert out.shape == (60, 60, 3)


def test_sheet_ignored_flags_are_mentioned(tmp_path, save, portrait, capsys):
    assert cli.main([str(save(portrait, "p.png")), "-o", str(tmp_path / "o"), "--per-page", "2",
                     "--orientation", "landscape", "--combine"]) == 0
    assert "--orientation, --combine do not apply" in capsys.readouterr().err
