"""The processing steps, in order.

`clean_source` erases regions, trims margins and detects B&W. `render`
scales and tunes the image for a printed area of a given size. `prepare`
combines them for one A4 page; `process_file` adds loading and saving, for
a single page or a poster split over several sheets.
"""

from dataclasses import dataclass, field
from pathlib import Path

from . import enhance, files, layout, poster, upscale

# Above this, the image is already sharp on paper and Real-ESRGAN adds
# minutes of work for no visible gain.
AI_USEFUL_BELOW_DPI = 240


@dataclass
class Options:
    orientation: str = "auto"      # auto | portrait | landscape
    fill: str = "crop"             # crop | extend | fit
    anchor: str = "center"         # crop only
    margin: float = None           # white margin in mm; None = 0 for A4, 6 for posters
    trim_border: bool = False
    trim_tolerance: int = 18
    remove: list = field(default_factory=list)   # [(x, y, w, h), ...]
    enhance: bool = True
    denoise: float = 1.0
    sharpen: float = 0.5
    sharpen_radius: float = 2.0
    shadows: float = 0.03
    midtones: float = 0.05
    ai: bool = False
    ai_model: str = "realesrgan-x4plus"
    ai_strength: float = 0.7       # mix of AI result vs plain resize
    min_dpi: float = 150
    poster: tuple = None           # (columns, rows) of A4 sheets, or None for one page
    overlap: float = 10.0          # mm shared by neighboring poster sheets
    pdf: bool = True
    preview: bool = True


@dataclass
class Report:
    source_size: tuple = (0, 0)
    orientation: str = ""
    effective_dpi: float = 0.0
    grayscale: bool = False
    trimmed: tuple = None
    quality: str = ""
    cleanup: str = ""              # why denoise ran, "" if it didn't
    resize_method: str = ""
    poster: object = None          # poster.Plan when making a poster
    warnings: list = field(default_factory=list)
    tips: list = field(default_factory=list)
    outputs: list = field(default_factory=list)


def clean_source(arr, opts, report):
    """Steps that work on the original pixels: erase regions, trim plain
    margins, detect black and white."""
    # Erase dates, credits, watermarks (coordinates refer to the original).
    if opts.remove:
        arr = enhance.inpaint_regions(arr, opts.remove)

    if opts.trim_border:
        arr, box = layout.trim_border(arr, opts.trim_tolerance)
        if arr.shape[1::-1] != tuple(report.source_size):
            report.trimmed = box
    elif layout.has_plain_border(arr, opts.trim_tolerance):
        report.tips.append("the image has a plain border; removing it (--trim-border) fills the page better")

    # Black and white: work in one channel so no color cast can creep in.
    report.grayscale = enhance.is_grayscale(arr)
    if report.grayscale:
        arr = enhance.to_gray(arr)
    return arr


def render(arr, opts, area, report, jpeg_quality=None, dpi=layout.DPI):
    """Scale, clean and tune `arr` for a printed area of `area` (w, h) pixels
    at `dpi`. Returns the area-sized array."""
    area_w, area_h = area
    h, w = arr.shape[:2]

    if opts.fill == "crop":
        x0, y0, x1, y1 = layout.crop_box(w, h, area_w, area_h, opts.anchor)
        arr = arr[y0:y1, x0:x1]
        out_size = (area_w, area_h)
        report.effective_dpi = layout.effective_dpi(x1 - x0, area_w, dpi)
    else:
        out_size = layout.fit_size(w, h, area_w, area_h)
        report.effective_dpi = layout.effective_dpi(w, out_size[0], dpi)
    report.quality = layout.print_quality(report.effective_dpi)

    if report.effective_dpi < opts.min_dpi:
        hint = "use a larger version of the image if you can" + ("" if opts.ai else ", or try AI upscaling (--ai)")
        report.warnings.append(f"below {opts.min_dpi:.0f} DPI: {hint}")

    # Clean noise and JPEG artifacts, and scale. When enlarging, clean first
    # so problems aren't enlarged. When shrinking, scale first: it's faster,
    # and shrinking already hides most noise. JPEG blocks shrunk below half
    # size are invisible, so they're left alone.
    scale = out_size[0] / arr.shape[1]
    use_ai = opts.ai and report.effective_dpi < AI_USEFUL_BELOW_DPI
    if scale >= 1:
        if opts.enhance:
            arr, report.cleanup = enhance.denoise(arr, opts.denoise, jpeg_quality)
        arr, report.resize_method = upscale.resize(arr, out_size, use_ai, opts.ai_model, opts.ai_strength)
    else:
        arr, report.resize_method = upscale.resize(arr, out_size)
        if opts.enhance:
            arr, report.cleanup = enhance.denoise(arr, opts.denoise, jpeg_quality if scale >= 0.5 else None)
    if opts.ai and not use_ai:
        report.resize_method += f" (AI skipped: already {report.effective_dpi:.0f} DPI)"

    # Fill the rest of the area.
    if opts.fill == "fit":
        arr = layout.place_on_white(arr, area_w, area_h)
    elif opts.fill == "extend":
        arr = layout.extend_background(arr, area_w, area_h)

    # Print tuning on luminance: gentle sharpening, shadow and midtone lift.
    # The sharpening radius is set for 300 DPI and scaled to the actual DPI.
    if opts.enhance:
        radius = opts.sharpen_radius * dpi / layout.DPI
        arr = enhance.finish_for_print(arr, opts.sharpen, radius, opts.shadows, opts.midtones)
    return arr


def prepare(arr, opts, jpeg_quality=None):
    """Turn an sRGB uint8 array into a finished A4 page. Returns (page, report)."""
    report = Report(source_size=(arr.shape[1], arr.shape[0]))
    arr = clean_source(arr, opts, report)

    h, w = arr.shape[:2]
    report.orientation = layout.choose_orientation(w, h, opts.orientation)
    page_w, page_h = layout.target_size(report.orientation)
    margin = layout.mm_to_px(opts.margin or 0)
    area = (page_w - 2 * margin, page_h - 2 * margin)
    if min(area) < 100:
        raise ValueError(f"--margin {opts.margin} mm leaves no room on the page")

    arr = render(arr, opts, area, report, jpeg_quality)
    if margin:
        arr = layout.place_on_white(arr, page_w, page_h)
    return arr, report


def prepare_poster(arr, opts, jpeg_quality=None):
    """Make a poster split over A4 sheets. Returns (sheets, overview, report)
    where overview is a small picture of the poster with the sheet grid."""
    report = Report(source_size=(arr.shape[1], arr.shape[0]))
    arr = clean_source(arr, opts, report)

    h, w = arr.shape[:2]
    cols, rows = opts.poster
    margin = 6.0 if opts.margin is None else opts.margin
    plan = poster.make_plan(w, h, cols, rows, opts.orientation, margin, opts.overlap)
    report.poster = plan
    report.orientation = plan.sheet_orientation

    art = render(arr, opts, plan.render_size, report, jpeg_quality, plan.render_dpi)
    sheets = poster.slice_sheets(art, plan)
    return sheets, poster.overview(art, plan), report


def process_file(path, out_dir, opts, name=None):
    """Load, prepare and save one image. `name` is the base for output file
    names (default: the input's name without extension).

    Returns (report, pdf_pages): the encoded PDF pages, so the caller can
    also build a combined PDF."""
    path, out_dir = Path(path), Path(out_dir)
    name = name or path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    original = files.load_rgb(path)
    quality = files.jpeg_quality(path)

    if opts.poster:
        sheets, overview, report = prepare_poster(original, opts, quality)
        after = files.to_rgb_image(overview)
        pages = [files.pdf_page(files.to_rgb_image(s)) for s in sheets]
        pdf = out_dir / f"{name}_poster.pdf"
        files.save_pdf(pages, pdf)
        report.outputs.append(pdf)
        preview_name = f"{name}_poster_preview.jpg"
    else:
        page, report = prepare(original, opts, quality)
        after = files.to_rgb_image(page)
        png = out_dir / f"{name}_a4.png"
        files.save_png(after, png)
        report.outputs.append(png)
        pages = [files.pdf_page(after)]
        if opts.pdf:
            pdf = out_dir / f"{name}_a4.pdf"
            files.save_pdf(pages, pdf)
            report.outputs.append(pdf)
        preview_name = f"{name}_preview.jpg"

    if opts.preview:
        preview = out_dir / preview_name
        files.save_preview(files.to_rgb_image(original), after, preview,
                           "After (poster)" if opts.poster else "After (A4)")
        report.outputs.append(preview)

    return report, pages
