"""Command-line interface: a4print INPUT [options]."""

import argparse
import sys
from pathlib import Path

from . import __version__, calibrate, files, poster, upscale
from .layout import ANCHORS
from .pipeline import Options, process_file


def region(text):
    """x,y,w,h where each value is pixels (120) or a percentage (10%)."""
    parts = [v.strip() for v in text.split(",")]
    values = []
    try:
        if len(parts) != 4:
            raise ValueError
        for v in parts:
            values.append(f"{float(v[:-1])}%" if v.endswith("%") else int(v))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected x,y,w,h in pixels or percent (e.g. 70%,90%,30%,10%), got '{text}'")
    if any((float(v[:-1]) if isinstance(v, str) else v) <= 0 for v in values[2:]):
        raise argparse.ArgumentTypeError("width and height must be positive")
    return tuple(values)


def grid(text):
    try:
        return poster.parse_grid(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def build_parser():
    p = argparse.ArgumentParser(
        prog="a4print",
        description="Prepare images for borderless A4 printing at 300 DPI.",
    )
    p.add_argument("input", help="an image file or a folder of images")
    p.add_argument("-o", "--output-dir", default="a4print_out", help="where to save results (default: %(default)s)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    g = p.add_argument_group("layout")
    g.add_argument("--orientation", choices=("auto", "portrait", "landscape"), default="auto",
                   help="page orientation; auto follows the image shape (default)")
    g.add_argument("--fill", choices=("auto", "crop", "extend", "fit"), default="auto",
                   help="auto (default): keep artwork with a plain border whole, crop photos when "
                        "little is lost, else fit; crop: fill the page and cut overflow; extend: fit "
                        "and fill the rest with mirrored, blurred edges; fit: fit inside a white border")
    g.add_argument("--anchor", choices=ANCHORS, default="center", help="which part to keep when cropping")
    g.add_argument("--margin", type=float, default=None, metavar="MM",
                   help="white margin in mm, for printers that can't print borderless "
                        "(default: 0, or 6 for posters)")
    g.add_argument("--trim-border", action="store_true", help="remove plain outer margins first")
    g.add_argument("--trim-tolerance", type=int, default=18, metavar="N",
                   help="how far (0-255) a pixel may differ from the margin color (default: %(default)s)")

    g = p.add_argument_group("poster")
    g.add_argument("--poster", type=grid, metavar="COLSxROWS",
                   help="split into a poster of several A4 sheets, e.g. 2x2 (about A2) or 3x3 (about A1)")
    g.add_argument("--overlap", type=float, default=10.0, metavar="MM",
                   help="how much neighboring poster sheets overlap, for gluing (default: %(default)s)")

    g = p.add_argument_group("cleanup")
    g.add_argument("--remove", type=region, action="append", default=[], metavar="x,y,w,h",
                   help="erase a rectangle by inpainting, in pixels of the original image or "
                        "percentages (70%%,90%%,30%%,10%%); repeatable. --grid helps find them")
    g.add_argument("--grid", action="store_true",
                   help="only save name_grid.jpg, the image with labeled coordinates, for --remove")
    g.add_argument("--denoise", type=float, default=1.0, metavar="S",
                   help="denoise strength, 0 disables; skipped on clean images (default: %(default)s)")
    g.add_argument("--sharpen", type=float, default=0.5, metavar="S",
                   help="unsharp mask amount, 0 disables (default: %(default)s)")
    g.add_argument("--sharpen-radius", type=float, default=2.0, metavar="PX",
                   help="unsharp mask radius at 300 DPI (default: %(default)s)")
    saved = calibrate.load_settings()
    note = " from a4print-calibrate" if saved else ""
    g.add_argument("--shadows", type=float, default=saved.get("shadows", 0.03), metavar="S",
                   help=f"shadow lift, 0-0.2 (default: %(default)s{note})")
    g.add_argument("--midtones", type=float, default=saved.get("midtones", 0.05), metavar="S",
                   help=f"midtone lift, 0 disables (default: %(default)s{note})")
    g.add_argument("--no-enhance", action="store_true", help="skip denoise, sharpening and tone lift")

    g = p.add_argument_group("upscaling")
    g.add_argument("--ai", action="store_true", help="upscale with Real-ESRGAN (realesrgan-ncnn-vulkan)")
    g.add_argument("--ai-model", default="realesrgan-x4plus", help="Real-ESRGAN model (default: %(default)s)")
    g.add_argument("--ai-strength", type=float, default=None, metavar="S",
                   help="0-1 mix of the AI result with a plain resize (default: 1.0; tones and "
                        "skin texture are kept from the original either way)")
    g.add_argument("--min-dpi", type=float, default=150, help="warn below this effective DPI (default: %(default)s)")

    g = p.add_argument_group("output")
    g.add_argument("--no-pdf", action="store_true", help="don't write the PDF")
    g.add_argument("--no-preview", action="store_true", help="don't write the before/after preview")
    g.add_argument("--combine", action="store_true",
                   help="also write all pages into one PDF (a4print_all.pdf), handy for printing a folder")
    return p


def options_from_args(args):
    return Options(
        orientation=args.orientation, fill=args.fill, anchor=args.anchor, margin=args.margin,
        trim_border=args.trim_border, trim_tolerance=args.trim_tolerance,
        remove=args.remove, enhance=not args.no_enhance, denoise=args.denoise,
        sharpen=args.sharpen, sharpen_radius=args.sharpen_radius,
        shadows=args.shadows, midtones=args.midtones,
        ai=args.ai, ai_model=args.ai_model, ai_strength=args.ai_strength,
        poster=args.poster, overlap=args.overlap, min_dpi=args.min_dpi,
        pdf=not args.no_pdf, preview=not args.no_preview,
    )


def describe(r, opts):
    """The report for one image as (label, text) lines, shared by the CLI
    and the web page."""
    w, h = r.source_size
    lines = [("source", f"{w} x {h} px" + ("  (B&W, kept neutral)" if r.grayscale else ""))]
    if r.trimmed:
        x0, y0, x1, y1 = r.trimmed
        lines.append(("trimmed", f"to {x1 - x0} x {y1 - y0} px"))
    if r.poster:
        p = r.poster
        size = f"{p.poster_mm[0] / 10:.0f} x {p.poster_mm[1] / 10:.0f} cm"
        lines.append(("poster", f"{p.cols}x{p.rows} {p.sheet_orientation} A4 sheets, {size}"
                                + (f", {p.size_name}" if p.size_name else "")))
        lines.append(("page", f"{_fill_text(r)}, {r.resize_method}"))
        lines.append(("resolution", f"{r.effective_dpi:.0f} DPI on the poster, {r.quality}"))
    else:
        lines.append(("page", f"A4 {r.orientation}, {_fill_text(r)}, {r.resize_method}"))
        lines.append(("resolution", f"{r.effective_dpi:.0f} DPI at A4, {r.quality}"))
    if opts.enhance:
        lines.append(("cleanup", f"applied ({r.cleanup})" if r.cleanup else "not needed"))
    lines += [("warning", w) for w in r.warnings]
    lines += [("tip", t) for t in r.tips]
    return lines


def _fill_text(r):
    return f"auto: {r.fill_reason}" if r.fill_reason else r.fill


def main(argv=None):
    args = build_parser().parse_args(argv)
    opts = options_from_args(args)

    source = Path(args.input)
    if not source.exists():
        print(f"error: {source} does not exist", file=sys.stderr)
        return 2
    images = files.list_images(source)
    if not images:
        print(f"error: no images found in {source}", file=sys.stderr)
        return 2

    if opts.ai:
        try:
            upscale.find_realesrgan()
        except upscale.UpscalerMissing as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3

    if (args.margin is not None and args.margin < 0) or args.overlap < 0:
        print("error: --margin and --overlap can't be negative", file=sys.stderr)
        return 2

    names = files.output_names(images)
    if args.grid:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in images:
            grid = out_dir / f"{names[path]}_grid.jpg"
            files.save_grid(files.load_rgb(path), grid)
            print(f"{path.name}\n  saved       {grid}")
        return 0

    failures = 0
    pages = []
    for path in images:
        print(f"{path.name}", flush=True)
        if opts.ai:
            print("  upscaling   Real-ESRGAN running (seconds on a GPU, minutes on CPU)", flush=True)
        try:
            r, pdf_pages = process_file(path, args.output_dir, opts, names[path])
        except Exception as exc:  # Keep going through a folder.
            failures += 1
            print(f"  error: {exc}", file=sys.stderr)
            continue
        for label, text in describe(r, opts):
            print(f"  {label:<11} {text}")
        for out in r.outputs:
            print(f"  saved       {out}")
        pages.extend(pdf_pages)

    if args.combine and pages:
        combined = Path(args.output_dir) / "a4print_all.pdf"
        files.save_pdf(pages, combined)
        print(f"\n{len(pages)} pages saved to {combined}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
