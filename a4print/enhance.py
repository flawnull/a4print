"""Cleanup steps: inpainting, denoise, sharpening, tone lift, B&W detection.

Everything here works on uint8 arrays, H x W x 3 RGB or H x W gray.
Sharpening and tone changes touch luminance only, so hues stay put.
"""

import math

import cv2
import numpy as np


def is_grayscale(arr, tolerance=6.0):
    """True if the image has no real color (B&W photo, sepia-free scan)."""
    if arr.ndim == 2:
        return True
    small = _shrink(arr, 512)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    return float(np.percentile(chroma, 99)) < tolerance


def to_gray(arr):
    return arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def inpaint_regions(arr, regions, radius=5):
    """Erase rectangles (x, y, w, h) by filling them from the surroundings."""
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, rw, rh in regions:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + rw), min(h, y + rh)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"--remove {x},{y},{rw},{rh} is outside the {w}x{h} image")
        mask[y0:y1, x0:x1] = 255
    # Grow the mask a little so anti-aliased text edges go too.
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.inpaint(arr, mask, radius, cv2.INPAINT_TELEA)


def estimate_noise(arr):
    """Gaussian noise sigma, Immerkaer's method (1996). Clean photos score ~1-2."""
    # Crop rather than shrink: shrinking would average the noise away.
    h, w = arr.shape[:2]
    y, x = max(0, (h - 1600) // 2), max(0, (w - 1600) // 2)
    gray = to_gray(arr[y:y + 1600, x:x + 1600]).astype(np.float64)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    response = cv2.filter2D(gray, -1, kernel)[1:-1, 1:-1]
    h, w = gray.shape
    return float(np.abs(response).sum() * math.sqrt(math.pi / 2) / (6 * (w - 2) * (h - 2)))


def denoise(arr, strength=1.0, skip_below=2.5):
    """Light non-local-means denoise. Skipped when the image is already clean.

    Returns (array, sigma, applied).
    """
    sigma = estimate_noise(arr)
    if strength <= 0 or sigma < skip_below:
        return arr, sigma, False
    h = min(sigma, 10.0) * 0.8 * strength
    if arr.ndim == 2:
        out = cv2.fastNlMeansDenoising(arr, None, h, 7, 21)
    else:
        out = cv2.fastNlMeansDenoisingColored(arr, None, h, h, 7, 21)
    return out, sigma, True


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
