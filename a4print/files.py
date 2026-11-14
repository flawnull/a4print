"""Loading images into sRGB and writing the PNG, PDF and preview."""

import io
from pathlib import Path

import img2pdf
import numpy as np
import pikepdf
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from .layout import A4_MM, DPI

try:  # iPhone photos (HEIC) need the optional pillow-heif package.
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif", ".heic", ".heif"}

# Standard JPEG luminance quantization table (ITU T.81, Annex K).
_JPEG_LUMA = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]

_SRGB = ImageCms.createProfile("sRGB")
SRGB_ICC = ImageCms.ImageCmsProfile(_SRGB).tobytes()


OUTPUT_SUFFIXES = ("_a4", "_preview", "_grid")


def list_images(path):
    """One file, or every image directly inside a folder (sorted).

    Files a4print wrote itself (name_a4.png, name_preview.jpg, ...) are
    skipped, so running twice on the same folder doesn't process them."""
    path = Path(path)
    if path.is_dir():
        return sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in EXTENSIONS and not p.stem.endswith(OUTPUT_SUFFIXES)
        )
    return [path]


def output_names(paths):
    """Base name for each input's outputs. Usually the file name without
    extension; when two inputs share one (photo.jpg, photo.png), the
    extension is kept too (photo_jpg, photo_png) so nothing is overwritten."""
    stems = [p.stem for p in paths]
    return {
        p: f"{p.stem}_{p.suffix.lstrip('.').lower()}" if stems.count(p.stem) > 1 else p.stem
        for p in paths
    }


def load_rgb(path):
    """Open an image as 8-bit sRGB: EXIF rotation applied, transparency on white,
    any embedded color profile (Adobe RGB, Display P3, CMYK...) converted to sRGB."""
    try:
        img = Image.open(path)
    except UnidentifiedImageError:
        if Path(path).suffix.lower() in (".heic", ".heif") and not HEIF_SUPPORT:
            raise ValueError("HEIC/HEIF photos need pillow-heif: pip install pillow-heif") from None
        raise ValueError(f"{Path(path).name} is not an image Pillow can read") from None
    img = ImageOps.exif_transpose(img)
    icc = img.info.get("icc_profile")

    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        # 16-bit gray: scale the full 0-65535 range to 0-255. Don't stretch
        # to the brightest pixel, which would lighten dark scans.
        data = np.asarray(img, dtype=np.float64)
        full_range = 65535.0 if data.max() > 255 else 255.0
        img = Image.fromarray(np.clip(data / full_range * 255 + 0.5, 0, 255).astype(np.uint8))

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


def jpeg_quality(path):
    """Estimated JPEG quality (1-100) from the file's quantization tables,
    the way ImageMagick's `identify` does it. None for other formats."""
    with Image.open(path) as img:
        tables = getattr(img, "quantization", None) if img.format in ("JPEG", "MPO") else None
    if not tables or 0 not in tables:
        return None
    scale = sum(tables[0]) / sum(_JPEG_LUMA) * 100
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return int(round(min(max(quality, 1), 100)))


def to_rgb_image(arr):
    """Gray arrays become RGB with R = G = B, so prints stay neutral."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr, "RGB")


def save_png(img, path):
    img.save(path, dpi=(DPI, DPI), icc_profile=SRGB_ICC)


def pdf_page(img, quality=95):
    """Encode one page for the PDF: high-quality JPEG with the sRGB profile.
    This keeps a PDF page at a few MB instead of 20+; the PNG is the lossless copy."""
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality, subsampling=0, dpi=(DPI, DPI), icc_profile=SRGB_ICC)
    return buffer.getvalue()


def save_pdf(pages, path):
    """Write encoded pages (from pdf_page) as A4 pages, image edge to edge.
    Each page turns to match its image, so portrait and landscape can mix."""
    a4 = (img2pdf.mm_to_pt(A4_MM[0]), img2pdf.mm_to_pt(A4_MM[1]))
    layout = img2pdf.get_layout_fun(a4, auto_orient=True)
    raw = img2pdf.convert(list(pages), layout_fun=layout)

    # Ask PDF viewers to print at 100% instead of "fit to page", which would
    # shrink the page and add a border. Acrobat and most viewers honor it.
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        pdf.Root.ViewerPreferences = pikepdf.Dictionary(PrintScaling=pikepdf.Name.None_)
        pdf.save(path)


def save_preview(before, after, path, label="After (A4)", height=1200):
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
    draw.text((gap * 2 + left.width, 18), label, fill=(40, 40, 40), font=font)
    canvas.save(path, "JPEG", quality=90, icc_profile=SRGB_ICC)


def save_grid(arr, path, steps=10):
    """Save the image with grid lines every 10%, labeled in pixels and
    percent, to help pick --remove coordinates."""
    img = Image.fromarray(arr if arr.ndim == 3 else np.stack([arr] * 3, axis=-1)).convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(max(12, min(w, h) // 45))
    line = max(1, min(w, h) // 600)
    for i in range(1, steps):
        x, y = round(w * i / steps), round(h * i / steps)
        draw.line([(x, 0), (x, h)], fill=(255, 0, 180, 170), width=line)
        draw.line([(0, y), (w, y)], fill=(255, 0, 180, 170), width=line)
        pct = f"{i * 100 // steps}%"
        for text, pos in ((f"x={x} ({pct})", (x + 4, 4)), (f"y={y} ({pct})", (4, y + 4))):
            box = draw.textbbox(pos, text, font=font)
            draw.rectangle(box, fill=(0, 0, 0, 160))
            draw.text(pos, text, fill=(255, 255, 255, 255), font=font)
    Image.alpha_composite(img, overlay).convert("RGB").save(path, "JPEG", quality=90)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font.
        return ImageFont.load_default()
