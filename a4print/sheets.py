"""Photo sheets: several images on one page, with cutting guides.

    a4print a.jpg b.jpg c.jpg d.jpg --per-page 4

Slots are equal, in a grid on a portrait page, separated by small gaps
with a dashed guide down the middle of each gap. Each image goes through
the same steps as a full page (cleanup, fit, AI, print tuning), just into
a smaller area, and is turned 90 degrees when that crops less.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from . import layout
from .pipeline import Report, check_options, clean_source, render

GRIDS = {2: (1, 2), 4: (2, 2), 6: (2, 3), 8: (2, 4), 9: (3, 3), 12: (3, 4)}   # columns x rows
GUIDE_GRAY = (200, 200, 200)


@dataclass
class Slot:
    x: int
    y: int
    w: int
    h: int


def plan(per_page, paper="a4", margin_mm=5.0, gap_mm=4.0):
    """Page size in pixels and the slots on it, row by row."""
    if per_page not in GRIDS:
        raise ValueError(f"--per-page must be one of {', '.join(map(str, GRIDS))}")
    cols, rows = GRIDS[per_page]
    page_w, page_h = layout.target_size("portrait", paper)
    margin, gap = layout.mm_to_px(margin_mm), layout.mm_to_px(gap_mm)
    slot_w = (page_w - 2 * margin - (cols - 1) * gap) // cols
    slot_h = (page_h - 2 * margin - (rows - 1) * gap) // rows
    if min(slot_w, slot_h) < 100:
        raise ValueError("the margin and gaps leave no room for the photos")
    # Center the grid (rounding can leave a pixel or two).
    used_w, used_h = cols * slot_w + (cols - 1) * gap, rows * slot_h + (rows - 1) * gap
    x0, y0 = (page_w - used_w) // 2, (page_h - used_h) // 2
    slots = [Slot(x0 + c * (slot_w + gap), y0 + r * (slot_h + gap), slot_w, slot_h)
             for r in range(rows) for c in range(cols)]
    return (page_w, page_h), slots, gap


def _crop_loss(w, h, slot):
    x0, y0, x1, y1 = layout.crop_box(w, h, slot.w, slot.h)
    return 1 - (x1 - x0) * (y1 - y0) / (w * h)


def fit_into_slot(arr, opts, slot, jpeg_quality=None):
    """Prepare one image for one slot. Returns (slot_array, report)."""
    report = Report(source_size=(arr.shape[1], arr.shape[0]))
    arr = clean_source(arr, opts, report)
    h, w = arr.shape[:2]
    # Turn the photo when its shape matches the slot better sideways.
    if (w > h) != (slot.w > slot.h) and _crop_loss(h, w, slot) < _crop_loss(w, h, slot) - 0.05:
        arr = np.ascontiguousarray(np.rot90(arr, k=-1))
        report.orientation = "turned 90 degrees"
    out = render(arr, opts, (slot.w, slot.h), report, jpeg_quality)
    return out, report


def make_sheets(images, opts, per_page):
    """Lay out `images` ((name, array, jpeg_quality) tuples) on sheets.

    Returns (sheets, entries, failed): RGB page arrays, (sheet number,
    name, report) for each image placed, and (name, reason) for each image
    that couldn't be prepared. The others still get their sheets."""
    check_options(opts)
    margin = 5.0 if opts.margin is None else opts.margin
    (page_w, page_h), slots, gap = plan(per_page, opts.paper, margin)
    ready, failed = [], []
    for name, arr, quality in images:      # all slots are the same size
        try:
            out, report = fit_into_slot(arr, opts, slots[0], quality)
        except Exception as exc:           # one bad image must not lose the batch
            failed.append((name, str(exc)))
            continue
        ready.append((name, np.stack([out] * 3, axis=-1) if out.ndim == 2 else out, report))
    sheets, entries = [], []
    for start in range(0, len(ready), per_page):
        page = np.full((page_h, page_w, 3), 255, np.uint8)
        group = ready[start:start + per_page]
        for slot, (name, out, report) in zip(slots, group, strict=False):
            page[slot.y:slot.y + slot.h, slot.x:slot.x + slot.w] = out
            entries.append((len(sheets) + 1, name, report))
        sheets.append(_draw_guides(page, slots[:len(group)], gap))
    return sheets, entries, failed


def _draw_guides(page, slots, gap):
    """Dashed cutting guides down the middle of the gaps between photos."""
    img = Image.fromarray(page)
    draw = ImageDraw.Draw(img)
    xs = sorted({s.x + s.w + gap // 2 for s in slots} - {max(s.x + s.w for s in slots) + gap // 2})
    ys = sorted({s.y + s.h + gap // 2 for s in slots} - {max(s.y + s.h for s in slots) + gap // 2})
    top, bottom = min(s.y for s in slots), max(s.y + s.h for s in slots)
    left, right = min(s.x for s in slots), max(s.x + s.w for s in slots)
    dash = 18
    for x in xs:
        for y in range(top, bottom, dash * 2):
            draw.line([(x, y), (x, min(y + dash, bottom))], fill=GUIDE_GRAY, width=2)
    for y in ys:
        for x in range(left, right, dash * 2):
            draw.line([(x, y), (min(x + dash, right), y)], fill=GUIDE_GRAY, width=2)
    return np.asarray(img)
