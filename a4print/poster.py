"""Posters split over several A4 sheets, for printing on a home printer.

Each sheet keeps a white margin (home printers can't print to the edge).
Neighboring sheets share an overlap strip: you trim the left and top
margins of a sheet and glue it over its neighbor's overlap, lining its
edge up with the dashed guide line. The guide ends up hidden under the
sheet on top.

Sheets are named like a spreadsheet: rows A, B, C from the top, columns
1, 2, 3 from the left. Assemble row by row, starting at A1.
"""

import math
import string
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .layout import A4_MM, DPI

# Rendering a 4x4 poster at 300 DPI would need 120+ megapixels. Above this
# budget the poster is rendered at a lower DPI and each sheet is scaled up
# when it's cut out. Posters are seen from further away, so this is fine.
MAX_POSTER_PIXELS = 75_000_000
MAX_SHEETS_PER_SIDE = 6

ISO_A = {"A3": (297, 420), "A2": (420, 594), "A1": (594, 841), "A0": (841, 1189)}

GUIDE_GRAY = (150, 150, 150)
TEXT_GRAY = (110, 110, 110)


def parse_grid(text):
    """'2x3' -> (2, 3): columns by rows."""
    try:
        cols, rows = (int(v) for v in text.lower().replace("×", "x").split("x"))
    except ValueError:
        raise ValueError(f"expected COLUMNSxROWS like 2x2 or 3x2, got '{text}'") from None
    if not (1 <= cols <= MAX_SHEETS_PER_SIDE and 1 <= rows <= MAX_SHEETS_PER_SIDE) or cols * rows < 2:
        raise ValueError(f"a poster needs 2 to {MAX_SHEETS_PER_SIDE}x{MAX_SHEETS_PER_SIDE} sheets, got '{text}'")
    return cols, rows


@dataclass
class Plan:
    cols: int
    rows: int
    sheet_orientation: str      # portrait | landscape
    sheet_mm: tuple             # (w, h)
    margin_mm: float
    overlap_mm: float
    poster_mm: tuple            # finished size (w, h) after assembly
    render_dpi: int
    render_size: tuple          # poster pixels (w, h) at render_dpi

    @property
    def area_mm(self):
        """Printed area of one sheet (w, h)."""
        return self.sheet_mm[0] - 2 * self.margin_mm, self.sheet_mm[1] - 2 * self.margin_mm

    @property
    def size_name(self):
        """Closest ISO A size, like 'about A2', or '' if none is close."""
        w, h = sorted(self.poster_mm)
        for name, (aw, ah) in ISO_A.items():
            if abs(w * h / (aw * ah) - 1) < 0.15:
                return f"about {name}"
        return ""


def make_plan(img_w, img_h, cols, rows, orientation="auto", margin_mm=6.0, overlap_mm=10.0, paper_mm=A4_MM):
    """Choose the sheet direction whose finished poster shape is closest to
    the image's, so the least gets cropped. `orientation` (portrait or
    landscape) refers to the whole poster."""
    candidates = []
    for sheet_orientation, sheet in (("portrait", paper_mm), ("landscape", paper_mm[::-1])):
        area = (sheet[0] - 2 * margin_mm, sheet[1] - 2 * margin_mm)
        if min(area) <= overlap_mm * 2:
            raise ValueError(f"margin {margin_mm} mm and overlap {overlap_mm} mm leave no room on the sheet")
        poster = (cols * area[0] - (cols - 1) * overlap_mm, rows * area[1] - (rows - 1) * overlap_mm)
        candidates.append((sheet_orientation, sheet, poster))

    if orientation in ("portrait", "landscape"):
        wanted = [c for c in candidates if (c[2][1] >= c[2][0]) == (orientation == "portrait")]
        if not wanted:
            raise ValueError(f"a {cols}x{rows} poster can't be {orientation}; try {rows}x{cols}")
        candidates = wanted

    image_shape = math.log(img_w / img_h)
    sheet_orientation, sheet, poster = min(candidates, key=lambda c: abs(math.log(c[2][0] / c[2][1]) - image_shape))

    full_px = (poster[0] / 25.4 * DPI) * (poster[1] / 25.4 * DPI)
    dpi = DPI if full_px <= MAX_POSTER_PIXELS else int(DPI * math.sqrt(MAX_POSTER_PIXELS / full_px))
    render_size = (round(poster[0] / 25.4 * dpi), round(poster[1] / 25.4 * dpi))
    return Plan(cols, rows, sheet_orientation, sheet, margin_mm, overlap_mm, poster, dpi, render_size)


def sheet_name(row, col):
    return f"{string.ascii_uppercase[row]}{col + 1}"


def _mm(mm, dpi=DPI):
    return round(mm / 25.4 * dpi)


def _area_box(plan, row, col, dpi):
    """Where a sheet's printed area sits on the poster, in pixels at `dpi`."""
    area_w, area_h = plan.area_mm
    x0 = col * (area_w - plan.overlap_mm)
    y0 = row * (area_h - plan.overlap_mm)
    return _mm(x0, dpi), _mm(y0, dpi), _mm(x0 + area_w, dpi), _mm(y0 + area_h, dpi)


def slice_sheets(art, plan):
    """Cut the rendered poster into A4 sheets at 300 DPI, with margins,
    overlap guides, cut marks and labels. Returns RGB arrays, row by row."""
    if art.ndim == 2:
        art = np.stack([art] * 3, axis=-1)
    sheet_px = (_mm(plan.sheet_mm[0]), _mm(plan.sheet_mm[1]))
    area_px = (_mm(plan.area_mm[0]), _mm(plan.area_mm[1]))
    margin = _mm(plan.margin_mm)
    sheets = []
    for row in range(plan.rows):
        for col in range(plan.cols):
            x0, y0, x1, y1 = _area_box(plan, row, col, plan.render_dpi)
            if plan.render_dpi == DPI:
                # Exact 1:1 cut, so the assembled poster matches pixel for pixel.
                x1, y1 = x0 + area_px[0], y0 + area_px[1]
            tile = art[y0:y1, x0:x1]
            short = (y1 - y0 - tile.shape[0], x1 - x0 - tile.shape[1])
            if any(short):  # last row or column can be a rounding pixel short
                tile = np.pad(tile, [(0, short[0]), (0, short[1]), (0, 0)], mode="edge")
            tile = Image.fromarray(tile)
            if tile.size != area_px:
                tile = tile.resize(area_px, Image.LANCZOS)

            sheet = Image.new("RGB", sheet_px, "white")
            sheet.paste(tile, (margin, margin))
            draw = ImageDraw.Draw(sheet)
            _draw_guides(draw, plan, row, col, margin, area_px)
            if plan.margin_mm >= 4:
                _draw_label(draw, plan, row, col, margin, sheet_px)
            sheets.append(np.asarray(sheet))
    return sheets


def _dashed(draw, start, end, width, dash=24):
    (x0, y0), (x1, y1) = start, end
    length = max(abs(x1 - x0), abs(y1 - y0))
    for s in range(0, length, dash * 2):
        e = min(s + dash, length)
        if x0 == x1:
            draw.line([(x0, y0 + s), (x0, y0 + e)], fill=GUIDE_GRAY, width=width)
        else:
            draw.line([(x0 + s, y0), (x0 + e, y0)], fill=GUIDE_GRAY, width=width)


def _draw_guides(draw, plan, row, col, margin, area_px):
    area_w, area_h = area_px
    width = 3
    # Dashed guides where the next sheet's edge goes, measured from where
    # that sheet really starts. The line is drawn just inside the overlap
    # strip (its center 1 px in), so it ends up fully hidden under that sheet.
    x0, y0, _, _ = _area_box(plan, row, col, DPI)
    if col < plan.cols - 1:
        x = margin + _area_box(plan, row, col + 1, DPI)[0] - x0 + width // 2
        _dashed(draw, (x, margin), (x, margin + area_h), width)
    if row < plan.rows - 1:
        y = margin + _area_box(plan, row + 1, col, DPI)[1] - y0 + width // 2
        _dashed(draw, (margin, y), (margin + area_w, y), width)

    # Cut marks in the margin, at the image edges that get trimmed.
    tick = max(margin - 8, 0)
    if tick and col > 0:
        for y in ((0, tick), (margin + area_h + 8, margin * 2 + area_h)):
            draw.line([(margin, y[0]), (margin, y[1])], fill=(0, 0, 0), width=2)
    if tick and row > 0:
        for x in ((0, tick), (margin + area_w + 8, margin * 2 + area_w)):
            draw.line([(x[0], margin), (x[1], margin)], fill=(0, 0, 0), width=2)


def _draw_label(draw, plan, row, col, margin, sheet_px):
    from .files import _font  # avoid a circular import at module load

    steps = []
    if col > 0 or row > 0:
        sides = " and ".join(s for s, on in (("left", col > 0), ("top", row > 0)) if on)
        steps.append(f"cut off the {sides} margin, glue onto the dashed line")
    text = f"{sheet_name(row, col)}  (row {row + 1}, column {col + 1} of {plan.cols}x{plan.rows})"
    if steps:
        text += "  -  " + "; ".join(steps)
    font = _font(max(14, int(margin * 0.42)))
    y = sheet_px[1] - margin + (margin - font.size) // 2
    draw.text((margin + 16, y), text, fill=TEXT_GRAY, font=font)  # clear of the cut mark


def overview(art, plan, max_side=1600):
    """A small picture of the whole poster with each sheet's area outlined
    and labeled, for the preview."""
    img = Image.fromarray(art if art.ndim == 3 else np.stack([art] * 3, axis=-1))
    scale = min(1.0, max_side / max(img.size))
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS).convert("RGBA")

    from .files import _font

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _font(max(16, img.width // (plan.cols * 6)))
    dpi = plan.render_dpi * scale
    for row in range(plan.rows):
        for col in range(plan.cols):
            x0, y0, x1, y1 = _area_box(plan, row, col, dpi)
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 0, 180, 220), width=3)
            label = sheet_name(row, col)
            box = draw.textbbox((0, 0), label, font=font)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            tw, th = box[2] - box[0], box[3] - box[1]
            draw.rectangle([cx - tw // 2 - 8, cy - th // 2 - 8, cx + tw // 2 + 8, cy + th // 2 + 12],
                           fill=(0, 0, 0, 150))
            draw.text((cx - tw // 2, cy - th // 2), label, fill=(255, 255, 255, 255), font=font)
    return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))
