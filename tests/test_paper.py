"""Letter and A3 paper alongside A4."""

import pikepdf
import pytest
from PIL import Image

from a4print import calibrate, cli, layout, web
from a4print.pipeline import Options, prepare, prepare_poster


def test_page_sizes_in_pixels():
    assert layout.target_size("portrait", "a4") == (2480, 3508)
    assert layout.target_size("portrait", "letter") == (2550, 3300)    # 8.5 x 11 in at 300 DPI
    assert layout.target_size("landscape", "a3") == (4961, 3508)
    with pytest.raises(ValueError, match="unknown paper"):
        layout.paper_mm("tabloid")


def test_letter_page(portrait):
    page, _ = prepare(portrait, Options(paper="letter", enhance=False))
    assert page.shape[:2] == (3300, 2550)


def pdf_inches(path):
    with pikepdf.open(path) as pdf:
        return [tuple(round(float(p.mediabox[i]) / 72, 2) for i in (2, 3)) for p in pdf.pages]


def test_letter_pdf_is_8_5_by_11_inches(tmp_path, save, portrait):
    assert cli.main([str(save(portrait, "p.png")), "-o", str(tmp_path / "o"), "--paper", "letter",
                     "--no-preview"]) == 0
    assert pdf_inches(tmp_path / "o" / "p_letter.pdf") == [(8.5, 11.0)]
    assert Image.open(tmp_path / "o" / "p_letter.png").size == (2550, 3300)


def test_letter_poster_sheets(tmp_path, save, landscape):
    sheets, _, report = prepare_poster(landscape, Options(poster=(2, 2), paper="letter"))
    assert all(s.shape[:2] in ((3300, 2550), (2550, 3300)) for s in sheets)
    assert report.poster.sheet_mm in ((215.9, 279.4), (279.4, 215.9))
    assert cli.main([str(save(landscape, "w.png")), "-o", str(tmp_path / "o"), "--paper", "letter",
                     "--poster", "2x2", "--no-preview"]) == 0
    assert set(pdf_inches(tmp_path / "o" / "w_poster.pdf")) <= {(8.5, 11.0), (11.0, 8.5)}


def test_saved_paper_becomes_the_default(tmp_path):
    assert cli.build_parser().parse_args(["x.png"]).paper == "a4"
    assert calibrate.main(["--paper", "letter", "-o", str(tmp_path)]) == 0
    assert cli.build_parser().parse_args(["x.png"]).paper == "letter"
    assert pdf_inches(tmp_path / "a4print_test_page.pdf") == [(8.5, 11.0)]
    # Saving answers later keeps the saved paper.
    calibrate.save_settings(calibrate.answers_to_settings(3, "B"))
    assert calibrate.load_settings()["paper"] == "letter"


def test_invalid_saved_paper_is_ignored():
    calibrate.settings_path().parent.mkdir(parents=True, exist_ok=True)
    calibrate.settings_path().write_text('{"paper": "../../etc"}')
    assert "paper" not in calibrate.load_settings()


def test_web_page_paper():
    info = web.paper_info()
    assert info["letter"]["page"] == "8.5 x 11 in" and info["a4"]["2x2"].endswith("about A2")
    assert "about" not in info["letter"]["2x2"]           # no A-series names for Letter
    assert "__PAPER_INFO__" not in web.PAGE
    argv = web.build_argv("in.png", "out", {"paper": ["letter"]})
    assert argv[argv.index("--paper") + 1] == "letter"
