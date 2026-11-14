"""Loading odd inputs: transparency, 16-bit, EXIF rotation, CMYK, profiles, HEIC, AVIF."""

import numpy as np
import pytest
from PIL import Image, ImageCms, features

from a4print import files


def test_transparency_goes_on_white(tmp_path):
    rgba = np.zeros((50, 80, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[:, 40:, 3] = 255  # left half transparent
    path = tmp_path / "logo.png"
    Image.fromarray(rgba, "RGBA").save(path)
    arr = files.load_rgb(path)
    assert (arr[:, :40] == 255).all()
    assert (arr[:, 60] == (200, 0, 0)).all()


def test_16_bit_png(tmp_path):
    data = (np.linspace(0, 65535, 100 * 60).reshape(60, 100)).astype(np.uint16)
    path = tmp_path / "scan16.png"
    Image.fromarray(data).save(path)
    arr = files.load_rgb(path)
    assert arr.dtype == np.uint8 and arr.shape == (60, 100, 3)
    assert arr[0, 0, 0] < 5 and arr[-1, -1, 0] > 250


def test_exif_rotation_is_applied(tmp_path):
    img = Image.new("RGB", (120, 60), "white")
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 CW to display", as phones write it
    path = tmp_path / "phone.jpg"
    img.save(path, exif=exif)
    assert files.load_rgb(path).shape[:2] == (120, 60)


def test_cmyk_jpeg(tmp_path):
    path = tmp_path / "print.jpg"
    Image.new("CMYK", (40, 30), (0, 255, 255, 0)).save(path)  # pure red in CMYK
    arr = files.load_rgb(path)
    assert arr.shape == (30, 40, 3)
    assert arr[15, 20, 0] > 200 and arr[15, 20, 1] < 60


def test_srgb_profile_passes_through(tmp_path):
    path = tmp_path / "p.png"
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGB", (10, 10), (180, 60, 60)).save(path, icc_profile=srgb)
    assert np.abs(files.load_rgb(path)[0, 0].astype(int) - (180, 60, 60)).max() <= 2


def test_broken_profile_does_not_crash(tmp_path):
    path = tmp_path / "bad.png"
    Image.new("RGB", (10, 10), (180, 60, 60)).save(path, icc_profile=b"not a profile" * 20)
    assert (files.load_rgb(path)[0, 0] == (180, 60, 60)).all()


def test_jpeg_quality_estimate(tmp_path):
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    for q in (40, 75, 92):
        path = tmp_path / f"q{q}.jpg"
        img.save(path, quality=q)
        assert abs(files.jpeg_quality(path) - q) <= 1
    png = tmp_path / "x.png"
    img.save(png)
    assert files.jpeg_quality(png) is None


def test_heic_input(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    path = tmp_path / "iphone.heic"
    pillow_heif.from_pillow(Image.new("RGB", (64, 48), (30, 140, 90))).save(path, quality=90)
    arr = files.load_rgb(path)
    assert arr.shape == (48, 64, 3)
    assert np.abs(arr[24, 32].astype(int) - (30, 140, 90)).max() < 12


@pytest.mark.skipif(not features.check("avif"), reason="Pillow built without AVIF")
def test_avif_input(tmp_path):
    path = tmp_path / "web.avif"
    Image.new("RGB", (64, 48), (200, 120, 40)).save(path, quality=90)
    assert files.load_rgb(path).shape == (48, 64, 3)


def test_not_an_image(tmp_path):
    path = tmp_path / "notes.jpg"
    path.write_text("hello")
    with pytest.raises(ValueError, match="not an image"):
        files.load_rgb(path)


def test_dark_16_bit_scan_stays_dark(tmp_path):
    """A dark 16-bit image must not be stretched to full brightness."""
    data = np.full((40, 40), 12850, np.uint16)  # 20% gray in 16-bit
    path = tmp_path / "dark16.png"
    Image.fromarray(data).save(path)
    assert abs(int(files.load_rgb(path)[0, 0, 0]) - 51) <= 1  # 20% of 255


def test_list_images_skips_own_outputs(tmp_path):
    for name in ("cat.jpg", "cat_a4.png", "cat_preview.jpg", "cat_grid.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in files.list_images(tmp_path)] == ["cat.jpg"]


def test_output_names_avoid_clashes(tmp_path):
    paths = [tmp_path / "photo.jpg", tmp_path / "photo.PNG", tmp_path / "other.webp"]
    assert list(files.output_names(paths).values()) == ["photo_jpg", "photo_png", "other"]


def test_truncated_download_gets_a_clear_error(tmp_path):
    full = tmp_path / "full.jpg"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(full)
    cut = tmp_path / "cut.jpg"
    cut.write_bytes(full.read_bytes()[: full.stat().st_size // 2])
    with pytest.raises(ValueError, match="incomplete"):
        files.load_rgb(cut)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_huge_images_are_reduced_while_loading(tmp_path, monkeypatch, fmt):
    monkeypatch.setattr(files, "LOAD_LIMIT_PIXELS", 20_000)
    path = tmp_path / f"big.{fmt.lower()}"
    Image.new("RGB", (400, 300), (120, 80, 40)).save(path, fmt)
    info = {}
    arr = files.load_rgb(path, info)
    assert arr.shape[0] * arr.shape[1] <= 20_000
    assert info["reduced_from"] == (400, 300)
    assert abs(arr.shape[1] / arr.shape[0] - 4 / 3) < 0.05


def test_float_tiff_in_0_to_1_range(tmp_path):
    path = tmp_path / "hdr.tif"
    Image.fromarray(np.full((20, 30), 0.5, np.float32)).save(path)
    assert abs(int(files.load_rgb(path)[0, 0, 0]) - 128) <= 1
