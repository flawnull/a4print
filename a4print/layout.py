"""A4 geometry: orientation, crop/fit/extend placement and border trimming.

All functions work on NumPy uint8 arrays, either H x W x 3 (RGB) or H x W (gray).

The fit-on-white placement follows the approach in SECRET-GUEST/PDF_Printer
(MIT License, https://github.com/SECRET-GUEST/PDF_Printer): compute A4 pixels
from the DPI, scale to the printable area keeping the aspect ratio, center on
a white page.
"""

import cv2
import numpy as np

DPI = 300
A4_MM = (210, 297)
A4_PORTRAIT = (round(A4_MM[0] * DPI / 25.4), round(A4_MM[1] * DPI / 25.4))  # (2480, 3508)
A4_LANDSCAPE = (A4_PORTRAIT[1], A4_PORTRAIT[0])

ANCHORS = ("center", "top", "bottom", "left", "right")


def choose_orientation(width, height, override="auto"):
    """Pick portrait or landscape. Square images go portrait."""
    if override in ("portrait", "landscape"):
        return override
    return "landscape" if width > height else "portrait"


def target_size(orientation):
    """(width, height) in pixels at 300 DPI."""
    return A4_LANDSCAPE if orientation == "landscape" else A4_PORTRAIT


def crop_box(src_w, src_h, tgt_w, tgt_h, anchor="center"):
    """Largest box in the source with the target's aspect ratio.

    Returns (x0, y0, x1, y1) in source pixels. The anchor decides which part
    is kept when the overflow is cropped.
    """
    target_ratio = tgt_w / tgt_h
    if src_w / src_h > target_ratio:
        # Too wide: crop left and right.
        crop_w, crop_h = round(src_h * target_ratio), src_h
    else:
        # Too tall: crop top and bottom.
        crop_w, crop_h = src_w, round(src_w / target_ratio)
    crop_w, crop_h = min(crop_w, src_w), min(crop_h, src_h)

    spare_x, spare_y = src_w - crop_w, src_h - crop_h
    x0 = {"left": 0, "right": spare_x}.get(anchor, spare_x // 2)
    y0 = {"top": 0, "bottom": spare_y}.get(anchor, spare_y // 2)
    return x0, y0, x0 + crop_w, y0 + crop_h


def fit_size(src_w, src_h, tgt_w, tgt_h):
    """Size that fits inside the target while keeping the aspect ratio."""
    scale = min(tgt_w / src_w, tgt_h / src_h)
    return min(tgt_w, max(1, round(src_w * scale))), min(tgt_h, max(1, round(src_h * scale)))


def effective_dpi(source_px, printed_px):
    """DPI the source pixels end up at when printed_px pixels cover the page at 300 DPI."""
    return source_px / (printed_px / DPI)


def _offsets(inner, outer):
    (iw, ih), (ow, oh) = inner, outer
    return (ow - iw) // 2, (oh - ih) // 2


def place_on_white(fitted, tgt_w, tgt_h):
    """Center a fitted image on a white page."""
    h, w = fitted.shape[:2]
    page = np.full((tgt_h, tgt_w) + fitted.shape[2:], 255, dtype=np.uint8)
    x, y = _offsets((w, h), (tgt_w, tgt_h))
    page[y:y + h, x:x + w] = fitted
    return page


def extend_background(fitted, tgt_w, tgt_h, blur=0.03, feather=0.06):
    """Fill the space around a fitted image with mirrored, blurred edge pixels.

    The image itself is untouched. Outside it, the edges are mirrored out to
    the page size, blurred (sigma = `blur` x the page's long side), and
    blended in over `feather` x the long side so there is no hard seam.
    """
    h, w = fitted.shape[:2]
    x, y = _offsets((w, h), (tgt_w, tgt_h))
    if (w, h) == (tgt_w, tgt_h):
        return fitted.copy()

    pad = [(y, tgt_h - h - y), (x, tgt_w - w - x)] + [(0, 0)] * (fitted.ndim - 2)
    mirrored = np.pad(fitted, pad, mode="symmetric")
    blurred = _strong_blur(mirrored, blur * max(tgt_w, tgt_h))

    # Distance from the image rectangle: 0 inside, growing outward.
    inside = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
    inside[y:y + h, x:x + w] = 1
    distance = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
    ramp = max(1.0, feather * max(tgt_w, tgt_h))
    weight = np.clip(1.0 - distance / ramp, 0.0, 1.0) ** 2
    if fitted.ndim == 3:
        weight = weight[..., None]

    out = mirrored * weight + blurred * (1.0 - weight)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _strong_blur(arr, sigma, factor=8):
    """Large Gaussian blur, done on a 1/8 size copy for speed."""
    h, w = arr.shape[:2]
    small = cv2.resize(arr, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / factor)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def trim_border(arr, tolerance=18, min_match=0.98):
    """Remove plain outer margins (white, beige, any flat color).

    The margin color is the median of the outermost pixel ring. Rows and
    columns are trimmed from each side while at least `min_match` of their
    pixels are within `tolerance` of that color.

    Returns (trimmed_array, (x0, y0, x1, y1)). If the outer ring is not a
    plain color, or trimming would leave almost nothing, the image is
    returned unchanged.
    """
    h, w = arr.shape[:2]
    pixels = arr.reshape(h, w, -1).astype(np.int16)
    ring = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    color = np.median(ring, axis=0)

    matches = np.abs(pixels - color).max(axis=2) <= tolerance
    if matches[[0, -1]].mean() < min_match and matches[:, [0, -1]].mean() < min_match:
        return arr, (0, 0, w, h)  # Outer edge is busy: no plain margin.

    rows = matches.mean(axis=1) >= min_match
    cols = matches.mean(axis=0) >= min_match

    def first_false(flags):
        idx = np.flatnonzero(~flags)
        return int(idx[0]) if idx.size else len(flags)

    y0 = first_false(rows)
    y1 = h - first_false(rows[::-1])
    x0 = first_false(cols)
    x1 = w - first_false(cols[::-1])

    if x1 - x0 < w * 0.1 or y1 - y0 < h * 0.1:
        return arr, (0, 0, w, h)  # Would remove nearly everything.
    return arr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def print_quality(dpi):
    """Plain-language verdict for an effective DPI on paper."""
    if dpi >= 240:
        return "excellent"
    if dpi >= 150:
        return "good"
    if dpi >= 100:
        return "fine from a distance, soft up close"
    return "low: will look soft or blocky up close"


def mm_to_px(mm):
    return round(mm * DPI / 25.4)
