"""One-step install of Real-ESRGAN (realesrgan-ncnn-vulkan) for --ai.

    a4print-setup-ai

Downloads the official build for this system from the Real-ESRGAN GitHub
releases (about 50 MB), checks its SHA-256 checksum, and unpacks it into
~/.a4print/realesrgan (or $A4PRINT_HOME/realesrgan). a4print finds it
there automatically. Real-ESRGAN is BSD-3-Clause licensed, by Xintao Wang
and contributors: https://github.com/xinntao/Real-ESRGAN

It also sets up the photo model, RealESRNet x4plus (another 67 MB). That
network is Real-ESRGAN x4plus trained without the GAN loss: it doesn't
invent detail, so faces and textures in photos stay true to the original.
The Real-ESRGAN releases ship its PyTorch weights only, so they are
downloaded, checked, and written into the ncnn layout of realesrgan-x4plus,
which has exactly the same network. The result is checked against a second
pinned checksum, so every computer ends up with the same file.
"""

import hashlib
import os
import pickle  # nosec B403: only a restricted unpickler is used, see _read_pth
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

RELEASE = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"

# platform -> (file, SHA-256, size in bytes), checked when downloading.
BUILDS = {
    "darwin": ("realesrgan-ncnn-vulkan-20220424-macos.zip",
               "e0ad05580abfeb25f8d8fb55aaf7bedf552c375b5b4d9bd3c8d59764d2cc333a", 51_817_124),
    "win32": ("realesrgan-ncnn-vulkan-20220424-windows.zip",
              "abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d", 45_474_481),
    "linux": ("realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
              "e5aa6eb131234b87c0c51f82b89390f5e3e642b7b70f2b9bbe95b6a285a40c96", 46_931_474),
}

# The photo model: official PyTorch weights, and the ncnn file made from them.
PHOTO_MODEL = "realesrnet-x4plus"
PHOTO_WEIGHTS = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
                 "a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df", 67_040_989)
PHOTO_BIN_SHA256 = "98d7e2a27aaabc31e6bd6fc82ce122c77c5b4e1f065c998df8d881a6e8e4b369"
GRAPH = "realesrgan-x4plus.param"   # same network, so the same ncnn graph
GRAPH_SHA256 = "35330ececcea33b6c397a72548e788d5d53becee4734c50b7fada36e89f10a86"

# Demo files in the zips that a4print doesn't need.
SKIP_SUFFIXES = (".jpg", ".png", ".mp4", ".md")


class SetupError(RuntimeError):
    pass


def install_dir():
    return Path(os.environ.get("A4PRINT_HOME") or Path.home() / ".a4print") / "realesrgan"


def binary_name():
    return "realesrgan-ncnn-vulkan.exe" if sys.platform == "win32" else "realesrgan-ncnn-vulkan"


def installed_binary():
    """Path of an a4print-installed Real-ESRGAN, or None."""
    path = install_dir() / binary_name()
    return path if path.is_file() else None


def models_dir():
    """The installed ncnn models (the folder name must contain "models" for
    realesrgan-ncnn-vulkan to accept it)."""
    return install_dir() / "models"


def photo_model_installed():
    d = models_dir()
    return (d / f"{PHOTO_MODEL}.param").is_file() and (d / f"{PHOTO_MODEL}.bin").is_file()


def supported():
    return sys.platform in BUILDS


def download(url, dest, expected_sha256, progress=None, max_bytes=300_000_000):
    """Download `url` to `dest` and check its SHA-256. Only https (and
    file: URLs, which the tests use) are allowed, and a download that grows
    past `max_bytes` is stopped, so a misbehaving server can't fill the disk."""
    if urllib.parse.urlsplit(url).scheme not in ("https", "file"):
        raise SetupError(f"refusing to download from a non-https address: {url}")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as out:  # nosec B310: scheme checked
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            done += len(chunk)
            if done > max_bytes:
                raise SetupError("the download is much larger than expected, so it was stopped")
            out.write(chunk)
            digest.update(chunk)
            if progress:
                progress(done, total)
    if digest.hexdigest() != expected_sha256:
        raise SetupError("the download doesn't match the expected checksum, so it wasn't installed. "
                         "Try again later.")


def install(progress=None, url_base=RELEASE):
    """Download, verify and unpack Real-ESRGAN. Returns the binary's path."""
    if not supported():
        raise SetupError(f"no Real-ESRGAN build for this system ({sys.platform})")
    name, sha256, _ = BUILDS[sys.platform]
    target = install_dir()

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / name
        try:
            download(url_base + name, archive, sha256, progress)
        except OSError as exc:
            raise SetupError(f"couldn't download Real-ESRGAN: {exc}") from None

        staging = Path(tmp) / "unpacked"
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                if member.is_dir() or member.filename.lower().endswith(SKIP_SUFFIXES):
                    continue
                zf.extract(member, staging)  # zipfile strips absolute and ".." paths

        binary = next(staging.rglob(binary_name()), None)
        if binary is None:
            raise SetupError("the download didn't contain the Real-ESRGAN program")
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(binary.parent), str(target))

    binary = target / binary_name()
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if sys.platform == "darwin":
        # Files fetched by Python aren't quarantined, but clear the flag in
        # case something else added it, so macOS lets the program run.
        xattr = "/usr/bin/xattr" if Path("/usr/bin/xattr").exists() else "xattr"
        try:
            subprocess.run([xattr, "-dr", "com.apple.quarantine", str(target)], capture_output=True)
        except OSError:
            pass  # not fatal: the flag is usually absent anyway
    return binary


class _WeightsUnpickler(pickle.Unpickler):
    """Reads a PyTorch checkpoint without PyTorch, and without running any
    code from the file: only plain containers and tensors are allowed."""

    def __init__(self, fh, archive, prefix):
        super().__init__(fh)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            import collections
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch" and name in _DTYPES:
            return name
        raise SetupError(f"unexpected content in the model file ({module}.{name})")

    def persistent_load(self, pid):
        kind, dtype, key, _location, _count = pid
        if kind != "storage" or dtype not in _DTYPES:
            raise SetupError("unexpected storage in the model file")
        raw = self.archive.read(f"{self.prefix}/data/{key}")
        return np.frombuffer(raw, _DTYPES[dtype])


_DTYPES = {"FloatStorage": np.dtype("<f4"), "HalfStorage": np.dtype("<f2")}


def _rebuild_tensor(storage, offset, size, stride, *_):
    if not size:
        return storage[offset:offset + 1].reshape(())
    item = storage.itemsize
    if (offset < 0 or any(n < 0 for n in size) or any(st < 0 for st in stride)
            or offset + sum((n - 1) * st for n, st in zip(size, stride, strict=True)) >= len(storage)):
        raise SetupError("damaged tensor in the model file")
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                           strides=tuple(s * item for s in stride)).copy()


def _read_pth(path):
    """name -> numpy array from a PyTorch .pth checkpoint (zip format)."""
    with zipfile.ZipFile(path) as archive:
        pkl = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        with archive.open(pkl) as fh:
            state = _WeightsUnpickler(fh, archive, pkl.rsplit("/", 1)[0]).load()
    for key in ("params_ema", "params"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    return state


def _write_ncnn_bin(state, graph, out):
    """Write the weights in the order and format of the ncnn graph's
    Convolution layers: fp16 weights after the fp16 tag, then fp32 bias."""
    convs = [k[:-len(".weight")] for k in state if k.endswith(".weight")]
    layers = [line.split() for line in graph.read_text().splitlines() if line.startswith("Convolution")]
    if len(convs) != len(layers):
        raise SetupError("the photo model doesn't fit the Real-ESRGAN graph")
    with open(out, "wb") as fh:
        for name, layer in zip(convs, layers, strict=True):
            w, b = state[name + ".weight"], state[name + ".bias"]
            params = dict(p.split("=", 1) for p in layer[6:] if "=" in p)
            if int(params.get("0", -1)) != w.shape[0] or int(params.get("6", -1)) != w.size:
                raise SetupError("the photo model doesn't fit the Real-ESRGAN graph")
            fh.write(struct.pack("<I", 0x01306B47))
            data = w.astype("<f2").tobytes()
            fh.write(data + b"\0" * (-len(data) % 4))
            fh.write(b.astype("<f4").tobytes())


def _find_graph():
    """The realesrgan-x4plus graph: from a4print's own install, or next to a
    Real-ESRGAN found on PATH or through $A4PRINT_REALESRGAN."""
    from . import upscale

    candidates = [models_dir() / GRAPH]
    try:
        candidates.append(Path(upscale.find_realesrgan()).resolve().parent / "models" / GRAPH)
    except upscale.UpscalerMissing:
        pass
    return next((c for c in candidates if c.is_file()), None)


def install_photo_model(progress=None, url=None):
    """Download RealESRNet x4plus and write it as an ncnn model next to the
    Real-ESRGAN models. Needs Real-ESRGAN installed (for the graph)."""
    graph = _find_graph()
    if graph is None:
        raise SetupError("install Real-ESRGAN first (a4print-setup-ai)")
    models_dir().mkdir(parents=True, exist_ok=True)
    src, sha256, _ = PHOTO_WEIGHTS
    with tempfile.TemporaryDirectory() as tmp:
        weights, made = Path(tmp) / "weights.pth", Path(tmp) / "model.bin"
        try:
            download(url or src, weights, sha256, progress)
        except OSError as exc:
            raise SetupError(f"couldn't download the photo model: {exc}") from None
        _write_ncnn_bin(_read_pth(weights), graph, made)
        digest = hashlib.sha256(made.read_bytes()).hexdigest()
        if digest != PHOTO_BIN_SHA256:
            raise SetupError("the converted photo model doesn't match its expected checksum")
        if hashlib.sha256(graph.read_bytes()).hexdigest() != GRAPH_SHA256:
            raise SetupError(f"{graph} isn't the Real-ESRGAN graph the photo model was made for")
        shutil.copyfile(graph, models_dir() / f"{PHOTO_MODEL}.param")
        shutil.move(str(made), str(models_dir() / f"{PHOTO_MODEL}.bin"))


def self_test():
    """Upscale a tiny image to check Real-ESRGAN runs on this computer.
    Returns "" on success, otherwise the reason it failed."""
    from . import upscale

    try:
        out = upscale.realesrgan_x4(np.full((16, 16, 3), 128, np.uint8))
        if photo_model_installed():
            out = upscale.realesrgan_x4(np.full((16, 16, 3), 128, np.uint8), PHOTO_MODEL)
    except Exception as exc:  # report any failure in plain words
        return str(exc).strip() or exc.__class__.__name__
    return "" if out.shape[:2] == (64, 64) else "unexpected output size"


def setup_max_quality():
    """a4print-setup-ai --max: download the PyTorch models and test them."""
    from . import maxq

    if not maxq.libraries_available():
        print(f"error: {maxq.INSTALL_HINT}", file=sys.stderr)
        return 1
    total = sum(m.size for m in maxq.MODELS.values()) / 1e6
    print(f"Downloading the maximum quality models ({total:.0f} MB)...")

    def progress(done, size):
        if size:
            print(f"\r  {done * 100 // size:3d}%", end="\n" if done >= size else "", flush=True)

    try:
        maxq.install_models(progress)
    except SetupError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    print(f"\nInstalled in {maxq.models_dir()}. Checking that they run...", flush=True)
    problem = max_self_test()
    if problem:
        print(f"error: the models are installed but didn't run: {problem}", file=sys.stderr)
        return 1
    print(f"Maximum quality is ready (running on: {maxq.device().type}). "
          "Use --max-quality, or tick it on the a4print-web page.")
    print("Credits: " + "; ".join(m.credit for m in maxq.MODELS.values()))
    return 0


def max_self_test():
    import numpy as np

    from . import maxq

    try:
        for kind in maxq.MODELS:
            out = maxq.upscale_x4(np.full((16, 16, 3), 128, np.uint8), kind)
            if out.shape[:2] != (64, 64):
                return "unexpected output size"
    except Exception as exc:  # report any failure in plain words
        return str(exc).strip() or exc.__class__.__name__
    return ""


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    if "--max" in args:
        return setup_max_quality()
    if installed_binary() and "--force" not in args:
        print(f"Real-ESRGAN is already installed in {install_dir()}")
    else:
        if not supported():
            print(f"error: no Real-ESRGAN build for this system ({sys.platform})", file=sys.stderr)
            return 1
        name, _, size = BUILDS[sys.platform]
        print(f"Downloading {name} ({size / 1e6:.0f} MB) from the Real-ESRGAN releases...")

        def progress(done, total):
            if total:
                print(f"\r  {done * 100 // total:3d}%", end="", flush=True)

        try:
            install(progress)
        except SetupError as exc:
            print(f"\nerror: {exc}", file=sys.stderr)
            return 1
        print(f"\nInstalled in {install_dir()}")

    if not photo_model_installed() or "--force" in args:
        size = PHOTO_WEIGHTS[2]
        print(f"Downloading the photo model, RealESRNet x4plus ({size / 1e6:.0f} MB)...")

        def progress(done, total):
            if total:
                print(f"\r  {done * 100 // total:3d}%", end="", flush=True)

        try:
            install_photo_model(progress)
        except SetupError as exc:
            print(f"\nerror: {exc}", file=sys.stderr)
            return 1
        print()

    print("Checking that it runs...", flush=True)
    problem = self_test()
    if problem:
        print(f"error: Real-ESRGAN is installed but didn't run: {problem}", file=sys.stderr)
        return 1
    print("AI upscaling is ready. Use --ai, or tick AI sharpening on the a4print-web page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
