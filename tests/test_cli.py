import io
import sys

import numpy as np
import pytest
from PIL import Image, ImageCms

from a4print import cli, enhance, upscale
from tests.conftest import gradient, with_noise
from a4print.pipeline import Options, prepare, process_file


def test_folder_run_writes_png_pdf_preview(tmp_path, save, portrait, landscape):
    save(portrait, "tall.png")
    save(landscape, "wide.jpg")
    out = tmp_path / "out"

    assert cli.main([str(tmp_path), "-o", str(out)]) == 0

    tall = Image.open(out / "tall_a4.png")
    wide = Image.open(out / "wide_a4.png")
    assert tall.size == (2480, 3508)
    assert wide.size == (3508, 2480)
    assert round(tall.info["dpi"][0]) == 300
    profile = ImageCms.ImageCmsProfile(io.BytesIO(tall.info["icc_profile"]))
    assert "sRGB" in ImageCms.getProfileDescription(profile)
    assert (out / "tall_preview.jpg").exists()

    pdf = (out / "wide_a4.pdf").read_bytes()
    assert pdf.startswith(b"%PDF")


def test_pdf_page_is_a4(tmp_path, save, portrait):
    pikepdf = pytest.importorskip("pikepdf")
    cli.main([str(save(portrait, "p.png")), "-o", str(tmp_path / "o"), "--no-preview"])
    with pikepdf.open(tmp_path / "o" / "p_a4.pdf") as pdf:
        box = [float(v) for v in pdf.pages[0].mediabox]
    width_mm, height_mm = (v / 72 * 25.4 for v in (box[2], box[3]))
    assert width_mm == pytest.approx(210, abs=0.5)
    assert height_mm == pytest.approx(297, abs=0.5)


@pytest.mark.parametrize("fill", ["crop", "extend", "fit"])
def test_fill_modes_make_a4(portrait, fill):
    page, report = prepare(portrait, Options(fill=fill))
    assert page.shape == (3508, 2480, 3)
    assert report.orientation == "portrait"


def test_fit_leaves_white_side_bars(portrait):
    page, _ = prepare(portrait, Options(fill="fit", enhance=False))
    assert (page[:, :20] == 255).all()


def test_low_dpi_warning(portrait):
    _, report = prepare(portrait, Options())
    assert report.effective_dpi < 150
    assert any("below 150 DPI" in w for w in report.warnings)
    assert report.quality.startswith("low")


def test_bw_output_is_neutral(bw_photo):
    page, report = prepare(bw_photo, Options())
    assert report.grayscale
    assert page.ndim == 2  # saved later as R = G = B


def test_ai_without_binary_fails_clearly(tmp_path, save, portrait, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv(upscale.ENV_VAR, raising=False)
    code = cli.main([str(save(portrait, "p.png")), "-o", str(tmp_path / "o"), "--ai"])
    assert code == 3
    assert "a4print-setup-ai" in capsys.readouterr().err


def test_bad_region_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["x.png", "--remove", "10,10,abc"])


FAKE_REALESRGAN = """#!{python}
import sys
from PIL import Image
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
img = Image.open(args["-i"])
img.resize((img.width * 4, img.height * 4)).save(args["-o"])
"""


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in binary is a Unix script")
def test_ai_calls_realesrgan(tmp_path, portrait, monkeypatch):
    """A stand-in binary with the same command line as realesrgan-ncnn-vulkan."""

    fake = tmp_path / "realesrgan-ncnn-vulkan"
    fake.write_text(FAKE_REALESRGAN.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv(upscale.ENV_VAR, str(fake))

    page, report = prepare(portrait, Options(ai=True, ai_strength=1.0, enhance=False))
    assert page.shape == (3508, 2480, 3)
    assert report.resize_method == "Real-ESRGAN (100%) + Lanczos"

    # Strength 0 is a plain Lanczos resize; 0.5 lands between the two.
    plain, _ = prepare(portrait, Options(enhance=False))
    zero, _ = prepare(portrait, Options(ai=True, ai_strength=0.0, enhance=False))
    half, _ = prepare(portrait, Options(ai=True, ai_strength=0.5, enhance=False))
    assert np.abs(zero.astype(int) - plain).max() <= 1
    gap_full = np.abs(page.astype(int) - plain).mean()
    assert 0.3 * gap_full < np.abs(half.astype(int) - plain).mean() < 0.7 * gap_full


def test_margin_adds_white_frame(landscape):
    page, report = prepare(landscape, Options(margin=5, enhance=False))
    assert page.shape == (2480, 3508, 3)
    frame = round(5 * 300 / 25.4)  # 59 px
    assert (page[:frame] == 255).all() and (page[:, -frame:] == 255).all()
    assert not (page[frame + 5, frame + 5] == 255).all()
    # Resolution is measured on the printable area, which is smaller than the sheet.
    _, borderless = prepare(landscape, Options(enhance=False))
    assert report.effective_dpi > borderless.effective_dpi


def test_combine_writes_one_pdf_with_mixed_orientations(tmp_path, save, portrait, landscape):
    pikepdf = pytest.importorskip("pikepdf")
    save(portrait, "a.png")
    save(landscape, "b.png")
    out = tmp_path / "out"
    assert cli.main([str(tmp_path), "-o", str(out), "--combine", "--no-preview"]) == 0
    with pikepdf.open(out / "a4print_all.pdf") as pdf:
        sizes = [tuple(round(float(p.mediabox[i]) / 72 * 25.4) for i in (2, 3)) for p in pdf.pages]
    assert sizes == [(210, 297), (297, 210)]


def test_jpeg_cleanup_reported(tmp_path, landscape):
    path = tmp_path / "pin.jpg"
    Image.fromarray(landscape).save(path, quality=40)
    report, _ = process_file(path, tmp_path / "o", Options(pdf=False, preview=False))
    assert "JPEG quality 40" in report.cleanup


def test_same_name_inputs_do_not_overwrite(tmp_path, portrait, landscape):
    Image.fromarray(portrait).save(tmp_path / "photo.png")
    Image.fromarray(landscape).save(tmp_path / "photo.jpg")
    out = tmp_path / "out"
    assert cli.main([str(tmp_path), "-o", str(out), "--no-pdf", "--no-preview"]) == 0
    assert Image.open(out / "photo_png_a4.png").size == (2480, 3508)
    assert Image.open(out / "photo_jpg_a4.png").size == (3508, 2480)


@pytest.mark.parametrize("text,expected", [
    ("10,20,30,40", (10, 20, 30, 40)),
    ("70%,90%,30%,10%", ("70.0%", "90.0%", "30.0%", "10.0%")),
    ("5, 50%, 100, 2.5%", (5, "50.0%", 100, "2.5%")),
])
def test_region_parsing(text, expected):
    assert cli.region(text) == expected


@pytest.mark.parametrize("text", ["1,2,3", "a,b,c,d", "0,0,0,10", "0,0,10,-5%"])
def test_bad_regions(text):
    with pytest.raises(Exception):
        cli.region(text)


def test_percent_region_erases_watermark():
    img = np.zeros((400, 600, 3), np.uint8)
    img[:] = (90, 140, 60)
    img[360:390, 420:590] = 255  # bottom-right "watermark"
    cleaned = enhance.inpaint_regions(img, [("68%", "88%", "31%", "11%")])
    assert np.abs(cleaned[375, 500].astype(int) - (90, 140, 60)).max() < 10
    # The pipeline accepts the same percentage regions.
    page, _ = prepare(img, Options(remove=[("68%", "88%", "31%", "11%")], enhance=False))
    assert page.shape == (2480, 3508, 3)


def test_grid_only_writes_grid(tmp_path, save, portrait):
    out = tmp_path / "out"
    assert cli.main([str(save(portrait, "p.png")), "-o", str(out), "--grid"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["p_grid.jpg"]
    assert Image.open(out / "p_grid.jpg").size == (600, 900)


def test_ai_skipped_when_already_sharp(monkeypatch, tmp_path):
    """A 300 DPI image gains nothing from AI, so Real-ESRGAN must not run."""
    monkeypatch.setenv(upscale.ENV_VAR, str(tmp_path / "missing-binary"))
    monkeypatch.setenv("PATH", str(tmp_path))
    big = np.zeros((3508, 2480, 3), np.uint8)
    big[:] = (120, 90, 60)
    page, report = prepare(big[:3000, :2200], Options(ai=True, enhance=False))
    assert page.shape == (3508, 2480, 3)
    assert "AI skipped" in report.resize_method


def test_large_image_is_shrunk_before_cleanup():
    noisy_big = with_noise(gradient(3000, 4300), 6)
    page, report = prepare(noisy_big, Options())
    assert page.shape == (3508, 2480, 3)
    assert report.effective_dpi > 300


def test_pdf_asks_viewers_to_print_at_actual_size(tmp_path, save, portrait):
    import pikepdf

    cli.main([str(save(portrait, "p.png")), "-o", str(tmp_path / "o"), "--no-preview"])
    with pikepdf.open(tmp_path / "o" / "p_a4.pdf") as pdf:
        assert pdf.Root.ViewerPreferences.PrintScaling == pikepdf.Name.None_


@pytest.mark.parametrize("size", [(1, 1), (5, 3), (16, 16)])
def test_tiny_images_do_not_break(size):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:] = (200, 100, 50)
    page, report = prepare(img, Options())
    assert page.shape[:2] in ((3508, 2480), (2480, 3508))
    assert report.quality.startswith("low")


def test_heavy_crop_warns():
    panorama = gradient(3000, 400)
    _, report = prepare(panorama, Options(fill="crop"))
    assert any("cropping removes" in w for w in report.warnings)
    _, report = prepare(gradient(1600, 1000), Options(fill="crop"))
    assert not any("cropping removes" in w for w in report.warnings)


def test_thin_extend_warns():
    _, report = prepare(gradient(3000, 200), Options(fill="extend"))
    assert any("covers only" in w for w in report.warnings)


def test_large_original_tip(tmp_path, monkeypatch):
    from a4print import files
    monkeypatch.setattr(files, "LOAD_LIMIT_PIXELS", 50_000)
    path = tmp_path / "big.png"
    Image.fromarray(gradient(600, 400)).save(path)
    report, _ = process_file(path, tmp_path / "o", Options(pdf=False, preview=False))
    assert any("reduced" in t for t in report.tips)
