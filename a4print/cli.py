"""Command-line interface: a4print INPUT [options]."""

import argparse
import sys
from pathlib import Path

from . import __version__, files, upscale
from .layout import ANCHORS
from .pipeline import Options, process_file


def region(text):
    try:
        x, y, w, h = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected x,y,w,h in pixels, got '{text}'")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return x, y, w, h


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
    g.add_argument("--fill", choices=("crop", "extend", "fit"), default="crop",
                   help="crop: fill the page and cut overflow (default); extend: fit and fill the "
                        "rest with mirrored, blurred edges; fit: fit inside a white border")
    g.add_argument("--anchor", choices=ANCHORS, default="center", help="which part to keep when cropping")
    g.add_argument("--margin", type=float, default=0.0, metavar="MM",
                   help="white margin in mm, for printers that can't print borderless (default: 0)")
    g.add_argument("--trim-border", action="store_true", help="remove plain outer margins first")
    g.add_argument("--trim-tolerance", type=int, default=18, metavar="N",
                   help="how far (0-255) a pixel may differ from the margin color (default: %(default)s)")

    g = p.add_argument_group("cleanup")
    g.add_argument("--remove", type=region, action="append", default=[], metavar="x,y,w,h",
                   help="erase a rectangle (original image pixels) by inpainting; repeatable")
    g.add_argument("--denoise", type=float, default=1.0, metavar="S",
                   help="denoise strength, 0 disables; skipped on clean images (default: %(default)s)")
    g.add_argument("--sharpen", type=float, default=0.5, metavar="S",
                   help="unsharp mask amount, 0 disables (default: %(default)s)")
    g.add_argument("--sharpen-radius", type=float, default=2.0, metavar="PX",
                   help="unsharp mask radius at 300 DPI (default: %(default)s)")
    g.add_argument("--shadows", type=float, default=0.03, metavar="S",
                   help="shadow lift, 0-0.2 (default: %(default)s)")
    g.add_argument("--midtones", type=float, default=0.05, metavar="S",
                   help="midtone lift, 0 disables (default: %(default)s)")
    g.add_argument("--no-enhance", action="store_true", help="skip denoise, sharpening and tone lift")

    g = p.add_argument_group("upscaling")
    g.add_argument("--ai", action="store_true", help="upscale with Real-ESRGAN (realesrgan-ncnn-vulkan)")
    g.add_argument("--ai-model", default="realesrgan-x4plus", help="Real-ESRGAN model (default: %(default)s)")
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
        ai=args.ai, ai_model=args.ai_model, min_dpi=args.min_dpi,
        pdf=not args.no_pdf, preview=not args.no_preview,
    )


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

    if args.margin < 0:
        print("error: --margin can't be negative", file=sys.stderr)
        return 2

    failures = 0
    pages = []
    for path in images:
        print(f"{path.name}")
        try:
            r, pdf_page = process_file(path, args.output_dir, opts)
        except Exception as exc:  # Keep going through a folder.
            failures += 1
            print(f"  error: {exc}", file=sys.stderr)
            continue
        w, h = r.source_size
        print(f"  source      {w} x {h} px{'  (B&W, kept neutral)' if r.grayscale else ''}")
        if r.trimmed:
            x0, y0, x1, y1 = r.trimmed
            print(f"  trimmed     to {x1 - x0} x {y1 - y0} px")
        print(f"  page        A4 {r.orientation}, {opts.fill}, {r.resize_method}")
        print(f"  resolution  {r.effective_dpi:.0f} DPI at A4, {r.quality}")
        if opts.enhance:
            print(f"  cleanup     {'applied (' + r.cleanup + ')' if r.cleanup else 'not needed'}")
        for warning in r.warnings:
            print(f"  warning     {warning}")
        for out in r.outputs:
            print(f"  saved       {out}")
        pages.append(pdf_page)

    if args.combine and pages:
        combined = Path(args.output_dir) / "a4print_all.pdf"
        files.save_pdf(pages, combined)
        print(f"\n{len(pages)} pages saved to {combined}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
