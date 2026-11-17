"""The processing steps, in order.

`clean_source` erases regions, trims margins and detects B&W. `render`
scales and tunes the image for a printed area of a given size. `prepare`
combines them for one A4 page; `process_file` adds loading and saving, for
a single page or a poster split over several sheets.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import enhance, files, layout, poster, upscale

# Above this, the image is already sharp on paper and Real-ESRGAN adds
# minutes of work for no visible gain.
AI_USEFUL_BELOW_DPI = 240


@dataclass
class Options:
    orientation: str = "auto"      # auto | portrait | landscape
    fill: str = "auto"             # auto | crop | extend | fit
    anchor: str = "center"         # crop only
    crop_at: tuple = None          # (fx, fy) in 0-1: free crop position; implies crop when fill is auto
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
    ai_model: str = None           # Real-ESRGAN (ncnn) model; None = picked per image
    ai_strength: float = None      # mix of AI result vs plain resize; None = 1.0
    max_quality: bool = False      # newer PyTorch models (maxq); implies AI
    grain: bool = True             # fine print grain on strongly enlarged photos
    min_dpi: float = 150
    paper: str = "a4"              # a4 | letter | a3
    poster: tuple = None           # (columns, rows) of sheets, or None for one page
    overlap: float = 10.0          # mm shared by neighboring poster sheets
    pdf: bool = True
    preview: bool = True
    detail_files: bool = False     # also save the detail crops as name_detail_plain/final.jpg (web page)


@dataclass
class Report:
    source_size: tuple = (0, 0)
    orientation: str = ""
    effective_dpi: float = 0.0
    grayscale: bool = False
    trimmed: tuple = None
    quality: str = ""
    fill: str = ""                 # the fill actually used (auto resolves to one)
    fill_reason: str = ""
    crop_note: str = ""            # set when the crop was moved for faces
    cleanup: str = ""              # why denoise ran, "" if it didn't
    grain: bool = False            # print grain was added
    resize_method: str = ""
    content: str = ""              # "artwork" or "photo", when AI ran
    detail: tuple = None           # (plain enlargement, result) of one area, for the preview
    detail_cm: float = 5.0         # side of that area on paper
    poster: object = None          # poster.Plan when making a poster
    warnings: list = field(default_factory=list)
    tips: list = field(default_factory=list)
    outputs: list = field(default_factory=list)


def check_options(opts):
    """Reject option values that make no sense, for every front end."""
    if (opts.margin is not None and opts.margin < 0) or opts.overlap < 0:
        raise ValueError("the margin and overlap can't be negative")


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
    elif opts.fill != "auto" and layout.has_plain_border(arr, opts.trim_tolerance):
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
    source = arr
    # Where the source's pixels land in the area: (source box, area box).
    mapping = ((0, 0, w, h), None)

    fill, pad = opts.fill, 255
    if opts.crop_at is not None and fill == "auto":
        fill = "crop"   # a chosen crop position is a request to crop
    if fill == "auto":
        fill, pad, report.fill_reason = layout.auto_fill(arr, area_w, area_h)
    report.fill = "fit" if fill == "pad" else fill

    if fill == "pad":
        # Extend the image's own border to the page shape, then treat it as
        # a page-filling image. Resolution is that of the original pixels.
        report.effective_dpi = layout.effective_dpi(w, layout.fit_size(w, h, area_w, area_h)[0], dpi)
        arr = layout.pad_to_shape(arr, area_w, area_h, *pad)
        s = area_w / arr.shape[1]
        ox, oy = (arr.shape[1] - w) // 2, (arr.shape[0] - h) // 2
        mapping = ((0, 0, w, h), (round(ox * s), round(oy * s), round(w * s), round(h * s)))
        h, w = arr.shape[:2]

    if fill == "pad":
        out_size = (area_w, area_h)
    elif fill == "crop":
        x0, y0, x1, y1 = layout.crop_box(w, h, area_w, area_h, opts.anchor, opts.crop_at)
        if opts.anchor == "center" and opts.crop_at is None and (x1 - x0 < w or y1 - y0 < h):
            # Nobody chose where to crop: don't cut through a face.
            moved = _keep_faces_in_frame(arr, (x0, y0, x1, y1))
            if moved != (x0, y0, x1, y1):
                (x0, y0, x1, y1), report.crop_note = moved, "crop moved to keep faces in the frame"
        arr = arr[y0:y1, x0:x1]
        out_size = (area_w, area_h)
        mapping = ((x0, y0, x1, y1), (0, 0, area_w, area_h))
        report.effective_dpi = layout.effective_dpi(x1 - x0, area_w, dpi)
        kept = (x1 - x0) * (y1 - y0) / (w * h)
        if kept < 0.6:
            report.warnings.append(f"cropping removes {100 - kept * 100:.0f}% of the image; "
                                   "fit or extend (--fill) keeps all of it")
    else:
        out_size = layout.fit_size(w, h, area_w, area_h)
        mapping = ((0, 0, w, h), ((area_w - out_size[0]) // 2, (area_h - out_size[1]) // 2) + out_size)
        report.effective_dpi = layout.effective_dpi(w, out_size[0], dpi)
        covered = out_size[0] * out_size[1] / (area_w * area_h)
        if fill == "extend" and covered < 0.5:
            report.warnings.append(f"the image covers only {covered * 100:.0f}% of the page and the rest "
                                   "is blurred, mirrored edges; fit (--fill fit) may look cleaner")
    report.quality = layout.print_quality(report.effective_dpi)

    if report.effective_dpi < opts.min_dpi:
        hint = "use a larger version of the image if you can" + (
            "" if (opts.ai or opts.max_quality) else ", or try AI upscaling (--ai)")
        report.warnings.append(f"below {opts.min_dpi:.0f} DPI: {hint}")

    # Clean noise and JPEG artifacts, and scale. When enlarging, clean first
    # so problems aren't enlarged. When shrinking, scale first: it's faster,
    # and shrinking already hides most noise. JPEG blocks shrunk below half
    # size are invisible, so they're left alone.
    scale = out_size[0] / arr.shape[1]
    wants_ai = opts.ai or opts.max_quality
    use_ai = wants_ai and report.effective_dpi < AI_USEFUL_BELOW_DPI
    if scale >= 1:
        strength, label, protect, model = opts.ai_strength, "", None, opts.ai_model
        if use_ai or (scale > 2 and opts.enhance and opts.grain):
            report.content = enhance.content_type(arr)
        if use_ai and not opts.max_quality and model is None:
            model = upscale.ai_model_for(report.content)
            if model is None:
                # GAN models invent faces and textures in photos; without
                # the photo model, a plain enlargement is the honest result.
                use_ai = False
                report.warnings.append("AI skipped for this photo: the photo model isn't set up yet; "
                                       "run 'a4print-setup-ai' once to add it")
        if opts.enhance:
            if use_ai and report.content == "photo":
                # The photo model removes noise and JPEG artifacts itself;
                # cleaning first only takes away real texture (measured:
                # 0.67 vs 0.72 SSIM against the original at 5x).
                report.cleanup = "left to the AI photo model"
            else:
                arr, report.cleanup = enhance.denoise(arr, opts.denoise, jpeg_quality)
        if use_ai:
            if strength is None:
                # Full strength for everything: merge_ai keeps the original's
                # tones and puts back texture (skin) the AI wiped out.
                strength = 1.0
                label = report.content
            if report.content == "artwork":
                protect = enhance.tiny_text_mask(arr)
                if protect.any():
                    label = ", ".join(filter(None, [label, "small text kept as is"]))
        arr, report.resize_method = upscale.resize(
            arr, out_size, use_ai, model, 1.0 if strength is None else strength, label, protect,
            max_quality=report.content if (use_ai and opts.max_quality) else None)
    else:
        arr, report.resize_method = upscale.resize(arr, out_size)
        if opts.enhance:
            arr, report.cleanup = enhance.denoise(arr, opts.denoise, jpeg_quality if scale >= 0.5 else None)
    if scale > 2 and opts.enhance and opts.grain:
        # Strong enlargements of photos print smooth and painted-looking;
        # fine grain reads as photographic texture (see enhance.grain_amount).
        if report.content == "photo":
            arr = enhance.add_grain(arr, enhance.grain_amount(scale), dpi)
            report.grain = True
    if wants_ai and not use_ai and report.effective_dpi >= AI_USEFUL_BELOW_DPI:
        report.resize_method += f" (AI skipped: already {report.effective_dpi:.0f} DPI)"

    # Fill the rest of the area.
    if fill == "fit":
        arr = layout.place_on_white(arr, area_w, area_h, pad)
    elif fill == "extend":
        arr = layout.extend_background(arr, area_w, area_h)

    # Print tuning on luminance: gentle sharpening, shadow and midtone lift.
    # The sharpening radius is set for 300 DPI and scaled to the actual DPI.
    if opts.enhance:
        radius = opts.sharpen_radius * dpi / layout.DPI
        arr = enhance.finish_for_print(arr, opts.sharpen, radius, opts.shadows, opts.midtones)
    report.detail = detail_pair(source, arr, mapping)
    if report.detail is not None:
        report.detail_cm = round(report.detail[1].shape[0] / dpi * 2.54, 1)
    return arr


def _keep_faces_in_frame(arr, box):
    """Shift a centered crop box just far enough that the faces in the image
    (with some room around them) stay inside it. The box keeps its size; if
    there are no faces, or they already fit, it comes back unchanged."""
    faces = enhance.find_faces(arr)
    if not faces:
        return box
    biggest = max(f[2] * f[3] for f in faces)
    faces = [f for f in faces if f[2] * f[3] >= 0.25 * biggest]     # ignore small background faces
    # The area to keep: all faces, with 35% of a face's size as headroom.
    left = min(x - 0.35 * fw for x, _, fw, _ in faces)
    right = max(x + 1.35 * fw for x, _, fw, _ in faces)
    top = min(y - 0.35 * fh for _, y, _, fh in faces)
    bottom = max(y + 1.35 * fh for _, y, _, fh in faces)
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box

    def shift(start, size, lo, hi, limit):
        if hi - lo >= size:                      # can't fit everything: center on it
            start = (lo + hi - size) / 2
        elif lo < start:
            start = lo
        elif hi > start + size:
            start = hi - size
        return int(min(max(round(start), 0), limit - size))

    nx, ny = shift(x0, x1 - x0, left, right, w), shift(y0, y1 - y0, top, bottom, h)
    return nx, ny, nx + (x1 - x0), ny + (y1 - y0)


def detail_pair(source, result, mapping, size=600):
    """One `size` x `size` area of the result at print size, plus the same
    area as a plain enlargement of the source pixels, to show what the
    processing did. The area is centered on the largest face if there is
    one (where AI quality shows most), otherwise it's the busiest area."""
    (sx0, sy0, sx1, sy1), (px, py, pw, ph) = mapping
    size = min(size, pw, ph)
    if size < 32:
        return None
    region = result[py:py + ph, px:px + pw]
    face = enhance.find_face(region)
    if face is not None:
        fx0, fy0, fw, fh = face
        cx, cy = fx0 + fw // 2, fy0 + int(fh * 0.45)   # just above the middle: the eyes
        x, y = px + cx - size // 2, py + cy - size // 2
    else:
        # Search on a 1/4 size copy for the window with the most edge energy.
        step = 4
        small = cv2.resize(enhance.to_gray(region), (max(1, pw // step), max(1, ph // step)),
                           interpolation=cv2.INTER_AREA)
        energy = np.abs(cv2.Laplacian(small.astype(np.float32), cv2.CV_32F))
        win = max(1, size // step)
        score = cv2.boxFilter(energy, -1, (win, win), normalize=False, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        score = score[: max(1, small.shape[0] - win + 1), : max(1, small.shape[1] - win + 1)]
        wy, wx = np.unravel_index(int(np.argmax(score)), score.shape)
        x, y = px + wx * step, py + wy * step
    x = min(max(x, px), px + pw - size)
    y = min(max(y, py), py + ph - size)

    final = result[y:y + size, x:x + size]
    fx, fy = (sx1 - sx0) / pw, (sy1 - sy0) / ph
    box = (sx0 + (x - px) * fx, sy0 + (y - py) * fy, sx0 + (x - px + size) * fx, sy0 + (y - py + size) * fy)
    plain = Image.fromarray(source).resize((size, size), Image.LANCZOS, box=box)
    return np.asarray(plain), final


def prepare(arr, opts, jpeg_quality=None):
    """Turn an sRGB uint8 array into a finished page. Returns (page, report)."""
    check_options(opts)
    report = Report(source_size=(arr.shape[1], arr.shape[0]))
    arr = clean_source(arr, opts, report)

    h, w = arr.shape[:2]
    report.orientation = layout.choose_orientation(w, h, opts.orientation)
    page_w, page_h = layout.target_size(report.orientation, opts.paper)
    margin = layout.mm_to_px(opts.margin or 0)
    area = (page_w - 2 * margin, page_h - 2 * margin)
    if min(area) < 100:
        raise ValueError(f"--margin {opts.margin} mm leaves no room on the page")

    arr = render(arr, opts, area, report, jpeg_quality)
    if margin:
        arr = layout.place_on_white(arr, page_w, page_h)
    return arr, report


def prepare_poster(arr, opts, jpeg_quality=None):
    """Make a poster split over sheets of paper. Returns (sheets, overview, report)
    where overview is a small picture of the poster with the sheet grid."""
    check_options(opts)
    report = Report(source_size=(arr.shape[1], arr.shape[0]))
    arr = clean_source(arr, opts, report)

    h, w = arr.shape[:2]
    cols, rows = opts.poster
    margin = 6.0 if opts.margin is None else opts.margin
    plan = poster.make_plan(w, h, cols, rows, opts.orientation, margin, opts.overlap,
                            layout.paper_mm(opts.paper))
    report.poster = plan
    report.orientation = plan.sheet_orientation

    art = render(arr, opts, plan.render_size, report, jpeg_quality, plan.render_dpi)
    report.fill_reason = report.fill_reason.replace("page", "poster")   # the wording is shared with pages
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

    load_info = {}
    original = files.load_rgb(path, load_info)
    quality = files.jpeg_quality(path)

    if opts.poster:
        sheets, overview, report = prepare_poster(original, opts, quality)
        after = files.to_rgb_image(overview)
        pages = [files.pdf_page(files.to_rgb_image(s)) for s in sheets]
        pdf = out_dir / f"{name}_poster.pdf"
        files.save_pdf(pages, pdf, layout.paper_mm(opts.paper))
        report.outputs.append(pdf)
        preview_name = f"{name}_poster_preview.jpg"
    else:
        page, report = prepare(original, opts, quality)
        after = files.to_rgb_image(page)
        png = out_dir / f"{name}_{opts.paper}.png"
        files.save_png(after, png)
        report.outputs.append(png)
        pages = [files.pdf_page(after)]
        if opts.pdf:
            pdf = out_dir / f"{name}_{opts.paper}.pdf"
            files.save_pdf(pages, pdf, layout.paper_mm(opts.paper))
            report.outputs.append(pdf)
        preview_name = f"{name}_preview.jpg"

    if "reduced_from" in load_info:
        w, h = load_info["reduced_from"]
        report.tips.append(f"very large original ({w} x {h} px) was reduced to "
                           f"{original.shape[1]} x {original.shape[0]} px while loading, "
                           "still more than the print needs")

    if opts.preview:
        preview = out_dir / preview_name
        files.save_preview(files.to_rgb_image(original), after, preview,
                           "After (poster)" if opts.poster else f"After ({layout.PAPER_NAMES[opts.paper]})",
                           detail=report.detail, detail_cm=report.detail_cm)
        report.outputs.append(preview)
    if opts.detail_files and report.detail is not None:
        for label, arr in zip(("plain", "final"), report.detail, strict=True):
            path = out_dir / f"{name}_detail_{label}.jpg"
            files.to_rgb_image(arr).save(path, "JPEG", quality=92, icc_profile=files.SRGB_ICC)
            report.outputs.append(path)

    return report, pages
