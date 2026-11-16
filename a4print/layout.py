"""Page geometry: paper sizes, orientation, crop/fit/extend placement and
border trimming.

All functions work on NumPy uint8 arrays, either H x W x 3 (RGB) or H x W (gray).

The fit-on-white placement follows the approach in SECRET-GUEST/PDF_Printer
(MIT License, https://github.com/SECRET-GUEST/PDF_Printer): compute A4 pixels
from the DPI, scale to the printable area keeping the aspect ratio, center on
a white page.
"""

import cv2
import numpy as np

DPI = 300

# Paper sizes in mm (portrait). Letter is the home-printer size in the US,
# Canada, Mexico and the Philippines; A4 nearly everywhere else.
PAPERS = {"a4": (210.0, 297.0), "letter": (215.9, 279.4), "a3": (297.0, 420.0)}
PAPER_NAMES = {"a4": "A4", "letter": "Letter", "a3": "A3"}

A4_MM = PAPERS["a4"]
A4_PORTRAIT = (round(A4_MM[0] * DPI / 25.4), round(A4_MM[1] * DPI / 25.4))  # (2480, 3508)
A4_LANDSCAPE = (A4_PORTRAIT[1], A4_PORTRAIT[0])

ANCHORS = ("center", "top", "bottom", "left", "right")


def choose_orientation(width, height, override="auto"):
    """Pick portrait or landscape. Square images go portrait."""
    if override in ("portrait", "landscape"):
        return override
    return "landscape" if width > height else "portrait"


def paper_mm(paper="a4"):
    """(width, height) of a paper size in mm, portrait."""
    try:
        return PAPERS[paper]
    except KeyError:
        raise ValueError(f"unknown paper size '{paper}'; choose from {', '.join(PAPERS)}") from None


def target_size(orientation, paper="a4"):
    """(width, height) of the page in pixels at 300 DPI."""
    w, h = (round(v * DPI / 25.4) for v in paper_mm(paper))
    return (h, w) if orientation == "landscape" else (w, h)


def crop_box(src_w, src_h, tgt_w, tgt_h, anchor="center", position=None):
    """Largest box in the source with the target's aspect ratio.

    Returns (x0, y0, x1, y1) in source pixels. The anchor decides which part
    is kept when the overflow is cropped. `position` (fx, fy), each 0 to 1,
    places the box freely instead: 0 is the left or top edge, 0.5 the
    center, 1 the right or bottom edge.
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
    if position is not None:
        fx, fy = (min(max(float(v), 0.0), 1.0) for v in position)
        x0, y0 = round(spare_x * fx), round(spare_y * fy)
    else:
        x0 = {"left": 0, "right": spare_x}.get(anchor, spare_x // 2)
        y0 = {"top": 0, "bottom": spare_y}.get(anchor, spare_y // 2)
    return x0, y0, x0 + crop_w, y0 + crop_h


def fit_size(src_w, src_h, tgt_w, tgt_h):
    """Size that fits inside the target while keeping the aspect ratio."""
    scale = min(tgt_w / src_w, tgt_h / src_h)
    return min(tgt_w, max(1, round(src_w * scale))), min(tgt_h, max(1, round(src_h * scale)))


def effective_dpi(source_px, printed_px, dpi=DPI):
    """DPI the source pixels end up at when printed_px pixels are printed at `dpi`."""
    return source_px / (printed_px / dpi)


def _offsets(inner, outer):
    (iw, ih), (ow, oh) = inner, outer
    return (ow - iw) // 2, (oh - ih) // 2


def place_on_white(fitted, tgt_w, tgt_h, color=255):
    """Center a fitted image on a page of `color` (white by default)."""
    h, w = fitted.shape[:2]
    page = np.empty((tgt_h, tgt_w) + fitted.shape[2:], dtype=np.uint8)
    page[:] = color
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
    # Blend the 8-bit images with a float weight map; posters are large
    # enough that floating-point copies of the images would need gigabytes.
    return cv2.blendLinear(mirrored, blurred, weight, 1.0 - weight)


def border_color(arr, tolerance=18, min_match=0.97):
    """(color, grain) of a plain outer border (paper, white, flat background),
    or None if the image's edges are busy. Looks at an outer band 1% wide, so
    paper grain and JPEG noise still count as plain. `grain` is the border's
    texture strength (standard deviation, in 0-255 levels)."""
    h, w = arr.shape[:2]
    band = max(2, min(h, w) // 100)
    pixels = arr.reshape(h, w, -1)
    ring = np.concatenate([
        pixels[:band].reshape(-1, pixels.shape[2]), pixels[-band:].reshape(-1, pixels.shape[2]),
        pixels[:, :band].reshape(-1, pixels.shape[2]), pixels[:, -band:].reshape(-1, pixels.shape[2]),
    ]).astype(np.int16)
    color = np.median(ring, axis=0)
    if (np.abs(ring - color).max(axis=1) <= tolerance).mean() < min_match:
        return None
    luma = ring.mean(axis=1)
    grain = float(1.4826 * np.median(np.abs(luma - np.median(luma))))  # robust std
    color = tuple(int(round(c)) for c in color)
    return (color[0] if arr.ndim == 2 else color), grain


def pad_to_shape(arr, area_w, area_h, color, grain=0.0, seed=0):
    """Pad the image with its border color to the page's shape, at the
    image's own resolution, adding grain that matches the border's. Scaled
    up together, the padding then has the same texture as the paper."""
    h, w = arr.shape[:2]
    if w / h > area_w / area_h:
        new_w, new_h = w, round(w * area_h / area_w)
    else:
        new_w, new_h = round(h * area_w / area_h), h
    canvas = np.empty((new_h, new_w) + arr.shape[2:], np.float32)
    canvas[:] = color
    if grain > 0.3:
        noise = np.random.default_rng(seed).normal(0, grain, (new_h, new_w)).astype(np.float32)
        canvas += noise if arr.ndim == 2 else noise[..., None]
    canvas = np.clip(canvas + 0.5, 0, 255).astype(np.uint8)
    x, y = (new_w - w) // 2, (new_h - h) // 2
    canvas[y:y + h, x:x + w] = arr
    return canvas


def auto_fill(arr, area_w, area_h, max_crop=0.2):
    """Pick how the image fills the page. Returns (fill, pad, reason) where
    pad is (color, grain) for "pad" and a color for "fit".

    - A plain border (designed posters, product shots): keep the whole image
      and pad the page with that border color, so nothing is cut.
    - Otherwise crop, if that removes at most `max_crop` of the image.
    - Otherwise keep the whole image on white.
    """
    h, w = arr.shape[:2]
    border = border_color(arr)
    if border is not None:
        return "pad", border, "kept whole, page extended in the image's own border color"
    x0, y0, x1, y1 = crop_box(w, h, area_w, area_h)
    loss = 1 - (x1 - x0) * (y1 - y0) / (w * h)
    if loss <= max_crop:
        return "crop", 255, "fills the page" + (f", {loss:.0%} cropped off the edges" if loss >= 0.005 else "")
    return "fit", 255, f"kept whole on white (filling the page would crop {loss:.0%})"


def has_plain_border(arr, tolerance=18, min_share=0.02):
    """True if trim_border would remove at least `min_share` of the width or
    height (checked on a small copy, so it's fast on big photos)."""
    h, w = arr.shape[:2]
    scale = min(1.0, 600 / max(h, w))
    small = cv2.resize(arr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    trimmed, _ = trim_border(small, tolerance)
    sh, sw = small.shape[:2]
    return trimmed.shape[0] <= sh * (1 - min_share) or trimmed.shape[1] <= sw * (1 - min_share)


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
