"""Resizing: Lanczos by default, Real-ESRGAN (realesrgan-ncnn-vulkan) with --ai."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BINARY = "realesrgan-ncnn-vulkan"
ENV_VAR = "A4PRINT_REALESRGAN"


class UpscalerMissing(RuntimeError):
    pass


def find_realesrgan():
    """Path to the Real-ESRGAN binary: $A4PRINT_REALESRGAN, then PATH, then
    the copy a4print-setup-ai installs."""
    from . import setup_ai

    candidate = os.environ.get(ENV_VAR)
    if candidate and Path(candidate).is_file():
        return candidate
    found = shutil.which(BINARY) or setup_ai.installed_binary()
    if found:
        return str(found)
    raise UpscalerMissing(
        "--ai needs Real-ESRGAN, which isn't installed yet. Install it with:\n"
        "  a4print-setup-ai\n"
        "(a one-time 50 MB download), or run without --ai to use Lanczos resizing."
    )


def lanczos(arr, size):
    return np.asarray(Image.fromarray(arr).resize(size, Image.LANCZOS))


def realesrgan_x4(arr, model="realesrgan-x4plus"):
    """Upscale 4x with the Real-ESRGAN command-line tool."""
    binary = find_realesrgan()
    rgb = np.stack([arr] * 3, axis=-1) if arr.ndim == 2 else arr
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.png", Path(tmp) / "out.png"
        Image.fromarray(rgb).save(src)
        cmd = [binary, "-i", str(src), "-o", str(dst), "-n", model, "-s", "4", "-f", "png"]
        models_dir = Path(binary).resolve().parent / "models"
        if models_dir.is_dir():
            cmd += ["-m", str(models_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not dst.exists():
            raise RuntimeError(f"Real-ESRGAN failed:\n{result.stderr.strip()}")
        return np.asarray(Image.open(dst).convert("L" if arr.ndim == 2 else "RGB"))


def _protect_areas(result, plain, protect, scale):
    """Put the plain enlargement back where `protect` (small text) is set,
    with a soft edge so there's no seam. Works only inside the protected
    areas' bounding box, so memory stays small."""
    ys, xs = np.nonzero(protect)
    pad = 3
    y0, y1 = max(0, ys.min() - pad), min(protect.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(protect.shape[1], xs.max() + 1 + pad)
    oy0, oy1 = round(y0 * scale), min(result.shape[0], round(y1 * scale))
    ox0, ox1 = round(x0 * scale), min(result.shape[1], round(x1 * scale))
    mask = cv2.resize(protect[y0:y1, x0:x1], (ox1 - ox0, oy1 - oy0), interpolation=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 1.5 * scale)
    keep_plain = np.clip(mask * 1.5, 0, 1)
    if result.ndim == 3:
        keep_plain = keep_plain[..., None]
    a = result[oy0:oy1, ox0:ox1].astype(np.float32)
    b = plain[oy0:oy1, ox0:ox1].astype(np.float32)
    result[oy0:oy1, ox0:ox1] = np.clip(a + keep_plain * (b - a) + 0.5, 0, 255).astype(np.uint8)


def resize(arr, size, ai=False, ai_model="realesrgan-x4plus", ai_strength=0.7, label="", protect=None):
    """Resize to `size` (w, h). With ai=True, enlargements go through Real-ESRGAN
    first and Lanczos takes the result to the exact size.

    `ai_strength` mixes the AI result with a plain Lanczos resize. Full
    strength gives skin a waxy, airbrushed look; 0.7 keeps the sharp edges
    and brings back natural texture. Use 1.0 for illustrations.

    Returns (array, method_used).
    """
    h, w = arr.shape[:2]
    enlarging = size[0] > w or size[1] > h
    if ai and enlarging:
        strength = min(max(ai_strength, 0.0), 1.0)
        result = lanczos(realesrgan_x4(arr, ai_model), size)
        plain = lanczos(arr, size)
        if strength < 1:
            # Blend in 8 bits: a poster can be 75 megapixels, and
            # floating-point copies of that would need gigabytes.
            result = cv2.addWeighted(result, strength, plain, 1 - strength, 0)
        if protect is not None and protect.any():
            result = np.array(result)  # writable
            _protect_areas(result, plain, protect, size[0] / w)
        return result, f"Real-ESRGAN ({strength:.0%}{', ' + label if label else ''}) + Lanczos"
    return lanczos(arr, size), "Lanczos"
