"""One-step install of Real-ESRGAN (realesrgan-ncnn-vulkan) for --ai.

    a4print-setup-ai

Downloads the official build for this system from the Real-ESRGAN GitHub
releases (about 50 MB), checks its SHA-256 checksum, and unpacks it into
~/.a4print/realesrgan (or $A4PRINT_HOME/realesrgan). a4print finds it
there automatically. Real-ESRGAN is BSD-3-Clause licensed, by Xintao Wang
and contributors: https://github.com/xinntao/Real-ESRGAN
"""

import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

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


def self_test():
    """Upscale a tiny image to check Real-ESRGAN runs on this computer.
    Returns "" on success, otherwise the reason it failed."""
    import numpy as np

    from . import upscale

    try:
        out = upscale.realesrgan_x4(np.full((16, 16, 3), 128, np.uint8))
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

    print("Checking that it runs...", flush=True)
    problem = self_test()
    if problem:
        print(f"error: Real-ESRGAN is installed but didn't run: {problem}", file=sys.stderr)
        return 1
    print("AI upscaling is ready. Use --ai, or tick AI sharpening on the a4print-web page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
