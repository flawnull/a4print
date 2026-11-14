"""Maximum quality mode: newer upscaling models run with PyTorch.

    pipx inject a4print 'torch>=2.6' spandrel   # once, with pipx
    a4print-setup-ai --max                 # downloads the two models (~100 MB)
    a4print photo.jpg --max-quality

The model is picked per image, from a side-by-side test on photos and
graphics after a4print's merge (edge sharpness of the strongest edges):

- Photos: 4xNomosWebPhoto_esrgan by Philip Hofmann (CC BY 4.0), trained on
  web photos with JPEG/WebP compression, blur and noise. On a cat photo it
  gave 27% crisper fur than Real-ESRGAN, and more natural skin on a face.
  https://github.com/Phhofm/models
- Artwork and graphics: Real-ESRGAN x4plus by Xintao Wang et al.
  (BSD-3-Clause), still the crispest on hard edges and lettering.
  https://github.com/xinntao/Real-ESRGAN

Also tried and left out: SwinIR-L, DAT2, ATD and RealPLKSR were softer.

Models are loaded with spandrel (MIT), which only unpickles model weights,
and each file's SHA-256 is checked after download.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import setup_ai


@dataclass(frozen=True)
class Model:
    file: str
    url: str
    sha256: str
    size: int
    credit: str


MODELS = {
    "photo": Model(
        "4xNomosWebPhoto_esrgan.safetensors",
        "https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_esrgan/4xNomosWebPhoto_esrgan.safetensors",
        "2f49c3065db8aaeb64adb08e8879001fe2f1460df5655499707efedad8a3d556", 33_467_822,
        "4xNomosWebPhoto_esrgan by Philip Hofmann, CC BY 4.0",
    ),
    "artwork": Model(
        "RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1", 67_040_989,
        "Real-ESRGAN x4plus by Xintao Wang et al., BSD-3-Clause",
    ),
}

INSTALL_HINT = ("maximum quality needs PyTorch and spandrel. Install them with "
                "'pipx inject a4print torch spandrel' (or 'pip install torch spandrel'; PyTorch 2.6 or newer), "
                "then run 'a4print-setup-ai --max'.")

_loaded = {}


class MaxQualityUnavailable(RuntimeError):
    pass


def models_dir():
    return Path(os.environ.get("A4PRINT_HOME") or Path.home() / ".a4print") / "models"


def libraries_available():
    """True if PyTorch and spandrel are installed. Checked without importing
    them: the first PyTorch import can take many seconds, and the web page
    asks this every time it opens."""
    import importlib.util

    return all(importlib.util.find_spec(name) is not None for name in ("torch", "spandrel"))


def models_installed():
    return all((models_dir() / m.file).is_file() for m in MODELS.values())


def available():
    return libraries_available() and models_installed()


def install_models(progress=None):
    """Download both models, checking their checksums."""
    target = models_dir()
    target.mkdir(parents=True, exist_ok=True)
    for model in MODELS.values():
        path = target / model.file
        if path.is_file():
            continue
        partial = path.with_suffix(path.suffix + ".part")
        try:
            setup_ai.download(model.url, partial, model.sha256, progress)
        except OSError as exc:
            partial.unlink(missing_ok=True)
            raise setup_ai.SetupError(f"couldn't download {model.file}: {exc}") from None
        except setup_ai.SetupError:
            partial.unlink(missing_ok=True)
            raise
        partial.replace(path)


def device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon graphics
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _model(kind):
    if kind not in _loaded:
        path = models_dir() / MODELS[kind].file
        if not path.is_file():
            raise MaxQualityUnavailable("the maximum quality models aren't downloaded yet; "
                                        "run 'a4print-setup-ai --max'")
        # Check the file is still exactly the one that was tested, in case it
        # was damaged or replaced since it was downloaded.
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != MODELS[kind].sha256:
            raise MaxQualityUnavailable(f"{path.name} doesn't match its expected checksum; "
                                        "delete it and run 'a4print-setup-ai --max' again")

        import spandrel

        descriptor = spandrel.ModelLoader().load_from_file(str(path))
        _loaded[kind] = (descriptor.model.to(device()).eval(), descriptor.scale)
    return _loaded[kind]


def upscale_x4(arr, kind="photo", tile=None, overlap=64):
    """Upscale 4x with the model for `kind` ("photo" or "artwork").

    Runs on overlapping tiles so memory stays bounded; each output pixel is
    taken from the tile where it sits at least overlap/2 pixels from an edge,
    so there are no seams. (Measured against one whole-image pass: with a
    64 px overlap, 99.9% of pixels match within 2 levels out of 255.)"""
    if not libraries_available():
        raise MaxQualityUnavailable(INSTALL_HINT)
    import torch

    net, scale = _model(kind)
    dev = next(net.parameters()).device
    tile = tile or (256 if dev.type == "cpu" else 384)
    gray = arr.ndim == 2
    rgb = np.stack([arr] * 3, axis=-1) if gray else arr
    h, w = rgb.shape[:2]
    x = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    out = np.empty((h * scale, w * scale, 3), np.uint8)

    step = max(1, tile - overlap)
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ya, xa = max(0, y1 - tile), max(0, x1 - tile)   # full-size tiles at the edges
            with torch.inference_mode():
                r = net(x[:, :, ya:y1, xa:x1].to(dev)).clamp_(0, 1).mul_(255).add_(0.5)
            r = r[0].permute(1, 2, 0).byte().cpu().numpy()
            # Keep this tile's inner part: skip half the overlap on sides
            # that another tile covers.
            ky = 0 if y0 == 0 else overlap // 2
            kx = 0 if x0 == 0 else overlap // 2
            oy, ox = y0 + ky, x0 + kx
            out[oy * scale:y1 * scale, ox * scale:x1 * scale] = r[(oy - ya) * scale:, (ox - xa) * scale:]
            if x1 == w:
                break
        if y1 == h:
            break
    return out[..., 0].copy() if gray else out
