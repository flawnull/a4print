"""Cleanup steps: inpainting, denoise, sharpening, tone lift, B&W detection.

Everything here works on uint8 arrays, H x W x 3 RGB or H x W gray.
Sharpening and tone changes touch luminance only, so hues stay put.
"""

import math
from pathlib import Path

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


_FACE_MODEL = Path(__file__).parent / "data" / "face_detection_yunet_2023mar.onnx"
_face_detector = None


def find_face(arr, max_side=1024):
    """(x, y, w, h) of the largest face in the image, or None."""
    faces = find_faces(arr, max_side)
    return max(faces, key=lambda f: f[2] * f[3]) if faces else None


def find_faces(arr, max_side=1024):
    """(x, y, w, h) of every face found in the image.

    Uses YuNet (MIT license, from the OpenCV model zoo), a small face
    detector that ships with a4print. Runs on a reduced copy for speed."""
    global _face_detector
    if not hasattr(cv2, "FaceDetectorYN") or not _FACE_MODEL.exists():
        return []
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(arr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR if small.ndim == 2 else cv2.COLOR_RGB2BGR)
    if _face_detector is None:
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        _face_detector = cv2.FaceDetectorYN.create(str(_FACE_MODEL), "", (320, 320), 0.7)
    _face_detector.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = _face_detector.detect(bgr)
    if faces is None:
        return []
    return [tuple(int(round(v / scale)) for v in face[:4]) for face in faces]


def to_gray(arr):
    return arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def resolve_region(region, width, height):
    """(x, y, w, h) in pixels. Values may be ints (pixels) or strings like
    "10%" (of the image width for x and w, of the height for y and h)."""
    totals = (width, height, width, height)
    x, y, w, h = (
        round(float(v[:-1]) / 100 * total) if isinstance(v, str) else int(v)
        for v, total in zip(region, totals, strict=True)
    )
    if isinstance(region[2], str):
        w = max(1, w)   # a small percentage of a small image must not round to nothing
    if isinstance(region[3], str):
        h = max(1, h)
    return x, y, w, h


def inpaint_regions(arr, regions, radius=5):
    """Erase what's inside rectangles (x, y, w, h), filling from the
    surroundings. Values are pixels or percentages, see resolve_region.

    Inside each box, only the marks themselves (text strokes, a small logo)
    are filled, found by _marks_in_box; the background between them is left
    untouched. Filling a whole box smears in whatever borders it, such as a
    nearby frame line. The whole box is filled when no separate marks can
    be told apart (a solid patch, or a faint watermark)."""
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        x, y, rw, rh = resolve_region(region, w, h)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + rw), min(h, y + rh)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"--remove {x},{y},{rw},{rh} is outside the {w}x{h} image")
        marks = _marks_in_box(arr, x0, y0, x1, y1)
        if marks is None:
            mask[y0:y1, x0:x1] = 255
        else:
            mask[y0:y1, x0:x1] |= marks
    # Grow the mask a little so anti-aliased edges and JPEG halos go too.
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.inpaint(arr, mask, radius, cv2.INPAINT_TELEA)


def _marks_in_box(arr, x0, y0, x1, y1, threshold=18):
    """Mask (uint8, box-sized) of the pixels in a box that stand out from its
    background, or None when that can't be decided.

    The background is estimated with a median filter about half as wide as
    the box's short side: thin things (text strokes, digits, line art)
    vanish from it while the surroundings keep their shape. Pixels that
    differ from it are the marks. A solid patch survives the filter, so
    nothing stands out and the caller fills the whole box."""
    h, w = arr.shape[:2]
    short = min(x1 - x0, y1 - y0)
    pad = short
    wx0, wy0, wx1, wy1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    window = arr[wy0:wy1, wx0:wx1]
    # Work on a reduced copy where the short side is about 24 px: fast, and
    # the kernel stays within what medianBlur handles well.
    scale = min(1.0, 24 / short)
    small = cv2.resize(window, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else window
    kernel = int(0.5 * short * scale) | 1
    if kernel < 3 or min(small.shape[:2]) < 3:
        return None
    background = cv2.medianBlur(small, min(kernel, 255))
    background = cv2.resize(background, (window.shape[1], window.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = np.abs(window.astype(np.int16) - background.astype(np.int16))
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    box = (slice(y0 - wy0, y1 - wy0), slice(x0 - wx0, x1 - wx0))
    # On a textured background (wood, fabric, foliage) the texture itself
    # differs from the smoothed background. Measure that outside the box and
    # only count pixels that stand out well beyond it.
    outside = np.ones(diff.shape, bool)
    outside[box] = False
    if outside.any():
        threshold = max(threshold, 3.0 * float(np.percentile(diff[outside], 75)))
    # A solid patch (a sticker, a filled logo) survives the median filter, so
    # only its outline would stand out. It shows as a jump in the background
    # across the box's border: compare a rim just inside with one just outside.
    rim = max(2, short // 6)
    inner = np.zeros(diff.shape, bool)
    inner[box] = True
    inner[box[0].start + rim:box[0].stop - rim, box[1].start + rim:box[1].stop - rim] = False
    outer = np.zeros(diff.shape, bool)
    outer[max(0, box[0].start - rim):box[0].stop + rim, max(0, box[1].start - rim):box[1].stop + rim] = True
    outer[box] = False
    if outer.any():
        flat_bg = background.reshape(*background.shape[:2], -1).astype(np.float32)
        jump = np.abs(np.median(flat_bg[inner], axis=0) - np.median(flat_bg[outer], axis=0)).max()
        if jump > 25:
            return None
    marks = (diff[box] > threshold).astype(np.uint8)
    # Whatever the marks enclose belongs to them: the inside of a solid
    # shape whose outline was found, or the counters of letters. Flood the
    # background in from the box's border; what it can't reach is enclosed.
    padded = np.pad(marks, 1)
    reached = padded.copy()
    cv2.floodFill(reached, None, (0, 0), 2)
    marks = (reached[1:-1, 1:-1] != 2)
    share = marks.mean()
    if share < 0.005 or share > 0.6:
        return None   # nothing stands out, or the box is one solid patch: fill all of it
    return marks.astype(np.uint8) * 255


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


def unsharp(channel, amount=0.5, radius=2.0, threshold=2, rows=1024):
    """Unsharp mask on one uint8 channel. `threshold` protects flat areas.

    Works in strips of `rows` lines (with an overlap wider than the blur),
    so a poster-sized channel never needs full-size float copies."""
    if amount <= 0:
        return channel
    out = np.empty_like(channel)
    margin = int(radius * 4) + 2
    for top in range(0, channel.shape[0], rows):
        a, b = max(0, top - margin), min(channel.shape[0], top + rows + margin)
        src = channel[a:b].astype(np.float32)
        detail = src - cv2.GaussianBlur(src, (0, 0), radius)
        detail[np.abs(detail) < threshold] = 0
        strip = np.clip(src + amount * detail + 0.5, 0, 255).astype(np.uint8)
        out[top:top + rows] = strip[top - a: top - a + rows]
    return out


def grain_amount(scale):
    """Print grain (standard deviation in levels) for a photo enlarged
    `scale` times: none up to 2x, then rising to 3 levels at 5x.

    A strong enlargement has no fine texture left: even the most faithful
    AI result prints as smooth, painted-looking areas. Fine grain at print
    scale reads as photographic texture instead. It is random, so it
    doesn't draw any shapes. 3 levels looked natural on paper-sized
    crops; 6 looked noisy."""
    return float(np.clip((scale - 2.0), 0.0, 3.0))


def add_grain(arr, amount, dpi=300, seed=0, rows=1024):
    """Add fine luminance grain of `amount` levels, strongest in midtones
    and fading out in highlights and shadows. The grain size is fixed on
    paper (about 0.07 mm), whatever the DPI. Works in strips."""
    if amount <= 0:
        return arr
    sigma = 0.8 * dpi / 300
    rng = np.random.default_rng(seed)
    out = np.empty_like(arr)
    pad = int(3 * sigma) + 2
    for top in range(0, arr.shape[0], rows):
        a0, b0 = max(0, top - pad), min(arr.shape[0], top + rows + pad)
        noise = rng.standard_normal((b0 - a0, arr.shape[1]), dtype=np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigma)
        noise *= amount / max(float(noise.std()), 1e-6)
        strip = arr[a0:b0].astype(np.float32)
        luma = (strip.mean(axis=2) if strip.ndim == 3 else strip) / 255
        noise *= 4 * luma * (1 - luma)
        strip = strip + (noise[..., None] if strip.ndim == 3 else noise)
        out[top:top + rows] = np.clip(strip + 0.5, 0, 255).astype(np.uint8)[top - a0: top - a0 + rows]
    return out


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
    return cv2.resize(arr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
