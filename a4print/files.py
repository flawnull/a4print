"""Loading images into sRGB and writing the PNG, PDF and preview."""

import io
from pathlib import Path

import img2pdf
import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps

from .layout import A4_MM, DPI

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

_SRGB = ImageCms.createProfile("sRGB")
SRGB_ICC = ImageCms.ImageCmsProfile(_SRGB).tobytes()


def list_images(path):
    """One file, or every image directly inside a folder (sorted)."""
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in EXTENSIONS)
    return [path]


def load_rgb(path):
    """Open an image as 8-bit sRGB: EXIF rotation applied, transparency on white,
    any embedded color profile (Adobe RGB, Display P3, CMYK...) converted to sRGB."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    icc = img.info.get("icc_profile")

    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        data = np.asarray(img, dtype=np.float64)
        img = Image.fromarray((data / max(data.max(), 1) * 255).astype(np.uint8))

    if icc and img.mode in ("RGB", "RGBA", "CMYK", "L"):
        try:
            source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            out_mode = "RGBA" if img.mode == "RGBA" else "RGB"
            img = ImageCms.profileToProfile(img, source, _SRGB, outputMode=out_mode)
        except (ImageCms.PyCMSError, OSError):
            pass  # Broken profile: fall back to treating the pixels as sRGB.

    if img.mode in ("P", "LA", "PA") or "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        page = Image.new("RGB", img.size, "white")
        page.paste(img, mask=img.getchannel("A"))
        img = page
    return np.asarray(img.convert("RGB"))


def to_rgb_image(arr):
    """Gray arrays become RGB with R = G = B, so prints stay neutral."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr, "RGB")


def save_png(img, path):
    img.save(path, dpi=(DPI, DPI), icc_profile=SRGB_ICC)


def save_pdf(img, path, quality=95):
    """One A4 page, image edge to edge, no margins.

    The page image is stored as a high-quality JPEG with the sRGB profile,
    which keeps the PDF at a few MB instead of 20+.
    """
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality, subsampling=0, dpi=(DPI, DPI), icc_profile=SRGB_ICC)
    page = (img2pdf.mm_to_pt(A4_MM[0]), img2pdf.mm_to_pt(A4_MM[1]))
    if img.width > img.height:
        page = page[::-1]
    layout = img2pdf.get_layout_fun(page)
    Path(path).write_bytes(img2pdf.convert(buffer.getvalue(), layout_fun=layout))


def save_preview(before, after, path, height=1200):
    """Side-by-side before/after JPG with labels."""
    gap, label_h = 40, 70

    def scaled(img):
        w = round(img.width * height / img.height)
        return img.resize((w, height), Image.LANCZOS)

    left, right = scaled(before), scaled(after)
    canvas = Image.new("RGB", (left.width + right.width + gap * 3, height + label_h + gap), (235, 235, 235))
    canvas.paste(left, (gap, label_h))
    canvas.paste(right, (gap * 2 + left.width, label_h))

    draw = ImageDraw.Draw(canvas)
    font = _font(36)
    draw.text((gap, 18), "Before", fill=(40, 40, 40), font=font)
    draw.text((gap * 2 + left.width, 18), "After (A4)", fill=(40, 40, 40), font=font)
    canvas.save(path, "JPEG", quality=90, icc_profile=SRGB_ICC)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font.
        return ImageFont.load_default()
