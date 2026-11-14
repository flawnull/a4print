"""The processing steps, in order. `prepare` does the image work; `process_file`
adds loading and saving."""

from dataclasses import dataclass, field
from pathlib import Path

from . import enhance, files, layout, upscale


@dataclass
class Options:
    orientation: str = "auto"      # auto | portrait | landscape
    fill: str = "crop"             # crop | extend | fit
    anchor: str = "center"         # crop only
    margin: float = 0.0            # white margin in mm, for printers without borderless
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
    min_dpi: float = 150
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
    warnings: list = field(default_factory=list)
    outputs: list = field(default_factory=list)


def prepare(arr, opts, jpeg_quality=None):
    """Turn an sRGB uint8 array into the finished A4 page array. Returns (page, report).

    `jpeg_quality` (from files.jpeg_quality) lets heavily compressed JPEGs
    get artifact cleanup."""
    report = Report(source_size=(arr.shape[1], arr.shape[0]))

    # 1. Erase dates, credits, watermarks (coordinates refer to the original).
    if opts.remove:
        arr = enhance.inpaint_regions(arr, opts.remove)

    # 2. Plain outer margins.
    if opts.trim_border:
        arr, box = layout.trim_border(arr, opts.trim_tolerance)
        h, w = arr.shape[:2]
        if (w, h) != report.source_size:
            report.trimmed = box

    # 3. Black and white: work in one channel so no color cast can creep in.
    report.grayscale = enhance.is_grayscale(arr)
    if report.grayscale:
        arr = enhance.to_gray(arr)

    # 4. Page size and how the image maps onto it.
    h, w = arr.shape[:2]
    report.orientation = layout.choose_orientation(w, h, opts.orientation)
    page_w, page_h = layout.target_size(report.orientation)
    margin = layout.mm_to_px(opts.margin)
    area_w, area_h = page_w - 2 * margin, page_h - 2 * margin
    if area_w < 100 or area_h < 100:
        raise ValueError(f"--margin {opts.margin} mm leaves no room on the page")

    if opts.fill == "crop":
        x0, y0, x1, y1 = layout.crop_box(w, h, area_w, area_h, opts.anchor)
        arr = arr[y0:y1, x0:x1]
        out_size = (area_w, area_h)
        report.effective_dpi = layout.effective_dpi(x1 - x0, area_w)
    else:
        out_size = layout.fit_size(w, h, area_w, area_h)
        report.effective_dpi = layout.effective_dpi(w, out_size[0])
    report.quality = layout.print_quality(report.effective_dpi)

    if report.effective_dpi < opts.min_dpi:
        hint = "use a larger version of the image if you can" + ("" if opts.ai else ", or try --ai")
        report.warnings.append(f"below {opts.min_dpi:.0f} DPI: {hint}")

    # 5. Clean noise and JPEG artifacts at the source resolution, before they get enlarged.
    if opts.enhance:
        arr, report.cleanup = enhance.denoise(arr, opts.denoise, jpeg_quality)

    # 6. Scale.
    arr, report.resize_method = upscale.resize(arr, out_size, opts.ai, opts.ai_model)

    # 7. Fill the rest of the printable area, then add the margin.
    if opts.fill == "fit":
        arr = layout.place_on_white(arr, area_w, area_h)
    elif opts.fill == "extend":
        arr = layout.extend_background(arr, area_w, area_h)
    if margin:
        arr = layout.place_on_white(arr, page_w, page_h)

    # 8. Print tuning on luminance: gentle sharpening, shadow and midtone lift.
    if opts.enhance:
        arr = enhance.finish_for_print(arr, opts.sharpen, opts.sharpen_radius, opts.shadows, opts.midtones)

    return arr, report


def process_file(path, out_dir, opts):
    """Load, prepare and save one image.

    Returns (report, pdf_page_bytes) so the caller can also build a
    combined PDF."""
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original = files.load_rgb(path)
    page, report = prepare(original, opts, files.jpeg_quality(path))
    page_img = files.to_rgb_image(page)

    png = out_dir / f"{path.stem}_a4.png"
    files.save_png(page_img, png)
    report.outputs.append(png)

    pdf_bytes = files.pdf_page(page_img)
    if opts.pdf:
        pdf = out_dir / f"{path.stem}_a4.pdf"
        files.save_pdf([pdf_bytes], pdf)
        report.outputs.append(pdf)

    if opts.preview:
        preview = out_dir / f"{path.stem}_preview.jpg"
        files.save_preview(files.to_rgb_image(original), page_img, preview)
        report.outputs.append(preview)

    return report, pdf_bytes
