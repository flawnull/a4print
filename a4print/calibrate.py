"""Printer test page and saved print settings.

    a4print-calibrate                      # writes a4print_test_page.pdf
    a4print-calibrate --black 4 --gray C   # saves your answers for this printer

Home printers differ: some lose detail in dark shadows, many print
midtones darker than the screen because ink spreads into the paper. The
test page measures both, and the answers become the defaults for
--shadows and --midtones.

- Shadows: a row of numbered squares just above black, on a black band.
  The first square you can tell apart from the band is the darkest level
  your printer shows; a4print lifts black to that level so dark detail
  isn't lost.
- Midtones: gray patches A-E, each next to fine black-and-white stripes.
  Stripes reflect half the light whatever the printer does, so from arm's
  length the patch that matches its stripes is the one this printer prints
  at the right lightness.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import enhance, files
from .layout import PAPER_NAMES, PAPERS, target_size

BLACK_LEVELS = [3, 6, 9, 12, 16, 20, 25, 30, 36]        # squares 1-9, out of 255
MIDTONE_CHOICES = {"A": 0.0, "B": 0.05, "C": 0.1, "D": 0.2, "E": 0.3}
STRIPES_GRAY = 188   # sRGB value that reflects 50% of the light, like the stripes


def settings_path():
    return Path(os.environ.get("A4PRINT_HOME") or Path.home() / ".a4print") / "settings.json"


def load_settings():
    """Saved print settings, e.g. {"shadows": 0.05, "midtones": 0.1, "paper": "letter"}, or {}."""
    try:
        data = json.loads(settings_path().read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    limits = {"shadows": (0.0, 0.2), "midtones": (0.0, 0.5)}
    settings = {}
    for key, (low, high) in limits.items():
        try:
            value = float(data[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):   # a damaged or edited file can't feed in NaN or huge values
            settings[key] = min(max(value, low), high)
    if data.get("paper") in PAPERS:
        settings["paper"] = data["paper"]
    return settings


def answers_to_settings(black, gray):
    """Turn test-page answers into --shadows and --midtones values."""
    if not 1 <= black <= len(BLACK_LEVELS):
        raise ValueError(f"the black square number must be 1 to {len(BLACK_LEVELS)}")
    gray = gray.strip().upper()
    if gray not in MIDTONE_CHOICES:
        raise ValueError("the gray patch must be one of A, B, C, D, E")
    return {"shadows": round(BLACK_LEVELS[black - 1] / 255, 3), "midtones": MIDTONE_CHOICES[gray]}


def save_settings(settings):
    """Save settings, keeping saved values that `settings` doesn't replace."""
    settings = {**load_settings(), **settings}
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2))
    return path


def test_page(paper="a4"):
    """The test page as an RGB array, portrait at 300 DPI, on `paper`."""
    w, h = target_size("portrait", paper)
    page = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(page)
    title, body, small = files._font(80), files._font(44), files._font(36)
    margin = 180

    d.text((margin, 160), "a4print printer test page", fill=(0, 0, 0), font=title)
    d.text((margin, 290), "Print at 100% / Actual size, on the paper you'll use. Judge it on paper:",
           fill=(60, 60, 60), font=body)
    d.text((margin, 350), "on a screen the stripes look darker than they print.", fill=(60, 60, 60), font=body)

    # 1. Shadows: squares just above black, on a black band.
    y = 470
    d.text((margin, y), "1. Which is the first square you can tell apart from the black band?",
           fill=(0, 0, 0), font=body)
    band_top, square = y + 90, 190
    d.rectangle([margin, band_top, w - margin, band_top + square + 140], fill=(0, 0, 0))
    gap = (w - 2 * margin - len(BLACK_LEVELS) * square) // (len(BLACK_LEVELS) + 1)
    for i, level in enumerate(BLACK_LEVELS):
        x = margin + gap + i * (square + gap)
        d.rectangle([x, band_top + 30, x + square, band_top + 30 + square], fill=(level,) * 3)
        d.text((x + square // 2 - 12, band_top + square + 50), str(i + 1), fill=(150, 150, 150), font=small)

    # 2. Midtones: gray patches next to 50% stripes.
    y = band_top + square + 260
    d.text((margin, y), "2. From arm's length, which gray patch matches the stripes next to it best?",
           fill=(0, 0, 0), font=body)
    patch_w, patch_h, row_gap = (w - 2 * margin - 160) // 2, 260, 60
    stripes = np.zeros((patch_h, patch_w), np.uint8)
    stripes[::4] = stripes[1::4] = 255   # 2 px black, 2 px white: half the light
    stripe_img = Image.fromarray(stripes).convert("RGB")
    top = y + 100
    for i, (letter, midtones) in enumerate(MIDTONE_CHOICES.items()):
        row_y = top + i * (patch_h + row_gap)
        value = int(enhance.tone_curve(0.0, midtones)[STRIPES_GRAY])
        d.text((margin, row_y + patch_h // 2 - 30), letter, fill=(0, 0, 0), font=title)
        d.rectangle([margin + 120, row_y, margin + 120 + patch_w, row_y + patch_h], fill=(value,) * 3)
        page.paste(stripe_img, (margin + 160 + patch_w, row_y))

    y = top + len(MIDTONE_CHOICES) * (patch_h + row_gap) + 20
    d.text((margin, y), "Then save the answers, for example square 4 and patch C:", fill=(60, 60, 60), font=body)
    d.text((margin, y + 70), "a4print-calibrate --black 4 --gray C", fill=(0, 0, 0), font=body)
    d.text((margin, y + 140), "or use Printer calibration under More options on the a4print-web page.",
           fill=(60, 60, 60), font=body)
    return np.asarray(page)


def write_test_page(path, paper="a4"):
    img = files.to_rgb_image(test_page(paper))
    files.save_pdf([files.pdf_page(img, quality=100)], path, PAPERS[paper])
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="a4print-calibrate",
        description="Print a test page, then save your printer's answers as a4print's defaults.")
    parser.add_argument("--black", type=int, metavar="N",
                        help=f"the first square (1-{len(BLACK_LEVELS)}) you can tell apart from black")
    parser.add_argument("--gray", metavar="LETTER", help="the gray patch (A-E) that best matches its stripes")
    parser.add_argument("--paper", choices=tuple(PAPERS),
                        help="the paper you print on (a4, letter, a3); saved as a4print's default")
    parser.add_argument("--reset", action="store_true", help="forget the saved settings")
    parser.add_argument("-o", "--output-dir", default=".", help="where to write the test page (default: here)")
    args = parser.parse_args(argv)

    if args.reset:
        settings_path().unlink(missing_ok=True)
        print("Saved print settings removed; a4print uses its standard values again.")
        return 0
    if args.paper:
        save_settings({"paper": args.paper})
        print(f"Paper set to {PAPER_NAMES[args.paper]} for every print (--paper still overrides it).")
    paper = args.paper or load_settings().get("paper", "a4")
    if args.black is None and args.gray is None:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = write_test_page(out / "a4print_test_page.pdf", paper)
        print(f"Test page ({PAPER_NAMES[paper]}) saved to {path}")
        print("Print it at 100% / Actual size, then run for example:")
        print("  a4print-calibrate --black 4 --gray C")
        return 0
    if args.black is None or args.gray is None:
        parser.error("give both --black and --gray")
    try:
        settings = answers_to_settings(args.black, args.gray)
    except ValueError as exc:
        parser.error(str(exc))
    path = save_settings(settings)
    print(f"Saved to {path}: --shadows {settings['shadows']} --midtones {settings['midtones']}")
    print("a4print now uses these for every print. Flags on the command line still override them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
