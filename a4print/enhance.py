"""Cleanup steps: inpainting, denoise, sharpening, tone lift, B&W detection.

Everything here works on uint8 arrays, H x W x 3 RGB or H x W gray.
Sharpening and tone changes touch luminance only, so hues stay put.
"""

import math

import cv2
import numpy as np


def is_grayscale(arr, tolerance=6.0, max_colored=0.0005):
    """True if the image has no real color (a B&W photo or scan).

    Most pixels must be near-neutral (99th percentile of Lab chroma below
    `tolerance`), and almost none (`max_colored`, 0.05%) may be clearly
    colored. The second check keeps a small colored logo or stamp from
    being turned gray.
    """
    if arr.ndim == 2:
        return True
    small = _shrink(arr, 512)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    if float(np.percentile(chroma, 99)) >= tolerance:
        return False
    return float((chroma > 3 * tolerance).mean()) <= max_colored


def content_type(arr):
    """"artwork" (posters, illustrations, logos, graphics) or "photo".

    Two measurements, taken at the image's own resolution on a center crop:
    how much of it is flat, and how abrupt its strong edges are. Drawn
    graphics have hard, step-like edges (score near 1.0); photos change
    gradually across edges (0.85 or less), even when heavily compressed.
    Mistaking a photo for artwork would over-sharpen it, so artwork needs
    both signs."""
    h, w = arr.shape[:2]
    y, x = max(0, (h - 768) // 2), max(0, (w - 768) // 2)
    crop = arr[y:y + 768, x:x + 768]
    gray = cv2.medianBlur(to_gray(crop), 3).astype(np.float32)
    if min(gray.shape) < 16:
        return "photo"
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    flat = float((cv2.blur((lap < 4).astype(np.float32), (5, 5)) > 0.95).mean())
    mag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    box = np.ones((5, 5), np.uint8)
    contrast = cv2.dilate(gray, box) - cv2.erode(gray, box)
    edges = (contrast > 40) & (mag >= cv2.dilate(mag, np.ones((3, 3), np.uint8)))
    if edges.sum() < 50:
        return "photo"
    sharpness = float(np.median(mag[edges] / (4 * contrast[edges])))
    return "artwork" if sharpness >= 0.92 and flat >= 0.4 else "photo"


def tiny_text_mask(arr, min_h=4, max_h=14):
    """Mask (uint8, 255 = text) of lines of very small text: characters 4-14
    px tall, like credit lines and dates on a downloaded poster.

    AI upscalers invent letter shapes at that size ("2025" can come out as
    "2023"), so these lines are better left to a plain enlargement:
    blurrier, but they still say what they said. Text is found the way OCR
    tools find it: character-sized shapes with similar neighbors in a row.
    Each hit is widened to cover its whole line."""
    gray = to_gray(arr)
    hits = np.zeros(gray.shape, np.uint8)
    for mode in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):  # dark text, then light text
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, 15, 15)
        n, _, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
        keep = [i for i in range(1, n)
                if min_h <= stats[i, 3] <= max_h and stats[i, 2] <= 2.5 * stats[i, 3] and stats[i, 4] >= 6]
        if not keep:
            continue
        c = centers[keep]
        for j, i in enumerate(keep):
            x, y, w, h = stats[i, :4]
            same_row = np.abs(c[:, 1] - c[j, 1]) < 0.6 * h
            close = np.abs(c[:, 0] - c[j, 0]) < 2.0 * h
            if np.sum(same_row & close) - 1 >= 2:
                hits[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2] = 255
    if not hits.any():
        return hits
    # Join the letters of a line into one block, then fill each block's box.
    joined = cv2.dilate(hits, np.ones((1, 4 * max_h), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    mask = np.zeros_like(hits)
    for x, y, w, h, _ in stats[1:]:
        # The widening added 2 * max_h at each end; keep about one character
        # of it, so letters at the ends of the line that weren't hit are covered.
        mask[y:y + h, max(0, x + max_h):x + w - max_h] = 255
    return mask


def to_gray(arr):
    return arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def resolve_region(region, width, height):
    """(x, y, w, h) in pixels. Values may be ints (pixels) or strings like
    "10%" (of the image width for x and w, of the height for y and h)."""
    totals = (width, height, width, height)
    return tuple(
        round(float(v[:-1]) / 100 * total) if isinstance(v, str) else int(v)
        for v, total in zip(region, totals)
    )


def inpaint_regions(arr, regions, radius=5):
    """Erase rectangles (x, y, w, h) by filling them from the surroundings.
    Values are pixels or percentages, see resolve_region."""
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        x, y, rw, rh = resolve_region(region, w, h)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + rw), min(h, y + rh)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"--remove {x},{y},{rw},{rh} is outside the {w}x{h} image")
        mask[y0:y1, x0:x1] = 255
    # Grow the mask a little so anti-aliased text edges go too.
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.inpaint(arr, mask, radius, cv2.INPAINT_TELEA)


def estimate_noise(arr):
    """Noise level (sigma, in 0-255 levels) measured in the flat half of the image.

    Immerkaer's (1996) operator, restricted to pixels with below-median
    gradient as in Tai & Yang (2008), so edges and texture (hair, leaves,
    fabric) aren't mistaken for noise. Clean photos score below 2.
    """
    # Crop rather than shrink: shrinking would average the noise away.
    h, w = arr.shape[:2]
    if h < 8 or w < 8:
        return 0.0  # too small to measure
    y, x = max(0, (h - 1600) // 2), max(0, (w - 1600) // 2)
    gray = to_gray(arr[y:y + 1600, x:x + 1600]).astype(np.float64)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    response = np.abs(cv2.filter2D(gray, -1, kernel))[1:-1, 1:-1]
    gradient = np.hypot(cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1))[1:-1, 1:-1]
    flat = gradient <= np.median(gradient)
    return float(response[flat].mean() * math.sqrt(math.pi / 2) / 6)


def denoise(arr, strength=1.0, jpeg_quality=None, skip_below=2.5, keep_grain=0.3):
    """Light non-local-means cleanup, run only when the image needs it.

    Two reasons can trigger it, and they share one pass so the image is
    never smoothed twice:
      - sensor noise above `skip_below` (phone photos in low light)
      - heavy JPEG compression, below quality 80 (Pinterest, social media):
        the 8x8 blocks and color smears would be enlarged on paper.

    For noise, `keep_grain` of the original is blended back in so photos
    keep a natural texture instead of a plastic, over-smoothed look.

    Returns (array, reason) where reason is "" when nothing was done.
    """
    if strength <= 0:
        return arr, ""
    sigma = estimate_noise(arr)
    h_noise = 0.4 * min(sigma, 12.0) if sigma >= skip_below else 0.0
    h_jpeg = (90 - jpeg_quality) / 13 if jpeg_quality is not None and jpeg_quality < 80 else 0.0
    h = min(max(h_noise, h_jpeg), 7.0) * strength
    if h < 0.5:
        return arr, ""

    if arr.ndim == 2:
        out = cv2.fastNlMeansDenoising(arr, None, h, 7, 21)
    else:
        # Color smears and chroma noise get a bit more cleanup than detail.
        out = cv2.fastNlMeansDenoisingColored(arr, None, h, h * 1.5, 7, 21)

    if h_noise and not h_jpeg and keep_grain > 0:
        out = cv2.addWeighted(out, 1 - keep_grain, arr, keep_grain, 0)

    reasons = []
    if h_noise:
        reasons.append(f"noise level {sigma:.1f}")
    if h_jpeg:
        reasons.append(f"JPEG quality {jpeg_quality}")
    return out, ", ".join(reasons)


def tone_curve(shadows=0.03, midtones=0.05):
    """256-entry lookup table that lifts shadows and midtones.

    Home inkjets print dark areas darker than the screen shows. This curve
    raises black slightly (by `shadows`, 0.03 = about 8 levels) and brightens
    midtones with a gentle gamma. White stays white.
    """
    shadows = float(np.clip(shadows, 0.0, 0.2))
    x = np.linspace(0.0, 1.0, 256)
    y = x ** (1.0 / (1.0 + max(midtones, -0.5)))
    y = y + shadows * (1.0 - y) ** 4
    return np.clip(np.round(y * 255), 0, 255).astype(np.uint8)


def unsharp(channel, amount=0.5, radius=2.0, threshold=2):
    """Unsharp mask on one uint8 channel. `threshold` protects flat areas."""
    if amount <= 0:
        return channel
    src = channel.astype(np.float32)
    detail = src - cv2.GaussianBlur(src, (0, 0), radius)
    detail[np.abs(detail) < threshold] = 0
    return np.clip(src + amount * detail + 0.5, 0, 255).astype(np.uint8)


def finish_for_print(arr, sharpen=0.5, sharpen_radius=2.0, shadows=0.03, midtones=0.05):
    """Sharpen and tone-lift the luminance only (Lab L for color images)."""
    lut = tone_curve(shadows, midtones)

    def adjust(luma):
        return cv2.LUT(unsharp(luma, sharpen, sharpen_radius), lut)

    if arr.ndim == 2:
        return adjust(arr)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    lab[..., 0] = adjust(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _shrink(arr, max_side):
    h, w = arr.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return arr
    return cv2.resize(arr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
