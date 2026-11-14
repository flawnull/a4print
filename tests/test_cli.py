import io

import pytest
from PIL import Image, ImageCms

from a4print import cli, upscale
from a4print.pipeline import Options, prepare


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
    assert any("low resolution" in w for w in report.warnings)


def test_bw_output_is_neutral(bw_photo):
    page, report = prepare(bw_photo, Options())
    assert report.grayscale
    assert page.ndim == 2  # saved later as R = G = B


def test_ai_without_binary_fails_clearly(tmp_path, save, portrait, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv(upscale.ENV_VAR, raising=False)
    code = cli.main([str(save(portrait, "p.png")), "-o", str(tmp_path / "o"), "--ai"])
    assert code == 3
    assert "realesrgan-ncnn-vulkan" in capsys.readouterr().err


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


def test_ai_calls_realesrgan(tmp_path, portrait, monkeypatch):
    """A stand-in binary with the same command line as realesrgan-ncnn-vulkan."""
    import sys

    fake = tmp_path / "realesrgan-ncnn-vulkan"
    fake.write_text(FAKE_REALESRGAN.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv(upscale.ENV_VAR, str(fake))

    page, report = prepare(portrait, Options(ai=True))
    assert page.shape == (3508, 2480, 3)
    assert report.resize_method == "Real-ESRGAN + Lanczos"
