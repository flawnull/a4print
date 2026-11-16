"""Photo sheets: several images per page."""

import pikepdf
import pytest

from a4print import cli, sheets, web
from a4print.pipeline import Options

from .conftest import gradient
from .test_web import get, png_bytes, post, server  # noqa: F401  (server is a fixture)


@pytest.mark.parametrize("n", sorted(sheets.GRIDS))
def test_slots_fit_on_the_page_without_overlap(n):
    (w, h), slots, gap = sheets.plan(n, "a4")
    assert len(slots) == n
    for s in slots:
        assert 0 < s.x and s.x + s.w < w and 0 < s.y and s.y + s.h < h
    for a in slots:
        for b in slots:
            if a is not b:
                apart = a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y
                assert apart


def test_bad_count():
    with pytest.raises(ValueError, match="per-page"):
        sheets.plan(5)


def test_landscape_photo_turns_into_a_portrait_slot():
    slot = sheets.Slot(0, 0, 1000, 1400)
    out, report = sheets.fit_into_slot(gradient(1400, 1000), Options(enhance=False), slot)
    assert out.shape[:2] == (1400, 1000)
    assert report.orientation == "turned 90 degrees"
    _, report = sheets.fit_into_slot(gradient(1000, 1400), Options(enhance=False), slot)
    assert report.orientation == ""


def test_five_images_make_two_sheets_with_guides():
    imgs = [(f"{i}.png", gradient(600 + 40 * i, 400), None) for i in range(5)]
    pages, entries, failed = sheets.make_sheets(imgs, Options(enhance=False), 4)
    assert failed == []
    assert len(pages) == 2 and pages[0].shape == (3508, 2480, 3)
    assert [e[0] for e in entries] == [1, 1, 1, 1, 2]
    (_, _), slots, gap = sheets.plan(4)
    guide_x = slots[0].x + slots[0].w + gap // 2
    column = pages[0][slots[0].y:slots[0].y + slots[0].h, guide_x]
    assert (column == 200).all(axis=1).any()          # dashed guide between the two columns


def test_cli_several_files_on_letter(tmp_path, save, portrait, landscape):
    paths = [str(save(a, f"{i}.png")) for i, a in enumerate((portrait, landscape, portrait))]
    out = tmp_path / "out"
    assert cli.main(paths + ["-o", str(out), "--per-page", "2", "--paper", "letter"]) == 0
    with pikepdf.open(out / "a4print_sheets_letter.pdf") as pdf:
        assert len(pdf.pages) == 2
        assert [round(float(v) / 72, 2) for v in pdf.pages[0].mediabox[2:4]] == [8.5, 11.0]
    assert (out / "a4print_sheets_letter_preview.jpg").exists()


def test_cli_rejects_sheet_and_poster(tmp_path, save, portrait, capsys):
    assert cli.main([str(save(portrait, "p.png")), "--per-page", "4", "--poster", "2x2"]) == 2
    assert "can't be combined" in capsys.readouterr().err


def test_web_sheet_flow(server, portrait, landscape):  # noqa: F811
    status, data = post(server + "/sheet-add?name=a.png", png_bytes(portrait))
    assert status == 200
    job = data["job"]
    status, _ = post(server + f"/sheet-add?name=b.png&job={job}", png_bytes(landscape))
    assert status == 200
    status, data = post(server + f"/sheet-make?job={job}&per_page=2&poster=3x3", b"")
    assert status == 200 and data["summary"]["sheets"] == 1 and data["summary"]["images"] == 2
    assert [line[0] for line in data["lines"]] == ["a.png", "b.png"]
    pdf = next(f for f in data["files"] if f["role"] == "pdf")
    assert get(server + pdf["url"])[2].startswith(b"%PDF")


def test_web_sheet_safety(server, portrait, monkeypatch):  # noqa: F811
    assert post(server + "/sheet-add?name=a.png&job=../../etc", png_bytes(portrait))[0] == 404
    assert post(server + "/sheet-make?job=0000000000000000", b"")[0] == 404
    assert post(server + "/sheet-add?name=a.png", png_bytes(portrait), {"X-A4print": "0"})[0] == 403
    monkeypatch.setattr(web.App, "MAX_SHEET_IMAGES", 2)
    _, data = post(server + "/sheet-add?name=a.png", png_bytes(portrait))
    post(server + f"/sheet-add?name=b.png&job={data['job']}", png_bytes(portrait))
    status, data = post(server + f"/sheet-add?name=c.png&job={data['job']}", png_bytes(portrait))
    assert status == 400 and "at most" in data["error"]


def test_bad_per_page_from_the_page(server, portrait):  # noqa: F811
    _, data = post(server + "/sheet-add?name=a.png", png_bytes(portrait))
    status, data = post(server + f"/sheet-make?job={data['job']}&per_page=7", b"")
    assert status == 400
