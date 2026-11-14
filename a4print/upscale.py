"""Resizing: Lanczos by default, Real-ESRGAN (realesrgan-ncnn-vulkan) with --ai."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

BINARY = "realesrgan-ncnn-vulkan"
ENV_VAR = "A4PRINT_REALESRGAN"


class UpscalerMissing(RuntimeError):
    pass


def find_realesrgan():
    """Path to the Real-ESRGAN binary, from $A4PRINT_REALESRGAN or PATH."""
    candidate = os.environ.get(ENV_VAR)
    if candidate and Path(candidate).is_file():
        return candidate
    found = shutil.which(BINARY)
    if found:
        return found
    raise UpscalerMissing(
        f"--ai needs Real-ESRGAN, but '{BINARY}' was not found.\n"
        "  1. Download the build for your system from\n"
        "     https://github.com/xinntao/Real-ESRGAN/releases (realesrgan-ncnn-vulkan-*.zip)\n"
        "  2. Unzip it and add that folder to your PATH,\n"
        f"     or set {ENV_VAR}=/full/path/to/{BINARY}\n"
        "Or run without --ai to use Lanczos resizing."
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


def resize(arr, size, ai=False, ai_model="realesrgan-x4plus"):
    """Resize to `size` (w, h). With ai=True, enlargements go through Real-ESRGAN
    first and Lanczos takes the result to the exact size.

    Returns (array, method_used).
    """
    h, w = arr.shape[:2]
    enlarging = size[0] > w or size[1] > h
    if ai and enlarging:
        return lanczos(realesrgan_x4(arr, ai_model), size), "Real-ESRGAN + Lanczos"
    return lanczos(arr, size), "Lanczos"
