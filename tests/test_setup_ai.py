"""The Real-ESRGAN installer, using a local stand-in zip instead of GitHub."""

import hashlib
import sys
import zipfile

import pytest

from a4print import setup_ai, upscale


def fake_release(tmp_path, binary_body=b"#!/bin/sh\nexit 0\n"):
    name, _, _ = setup_ai.BUILDS[sys.platform]
    path = tmp_path / "release" / name
    path.parent.mkdir()
    with zipfile.ZipFile(path, "w") as zf:
        folder = "realesrgan-ncnn-vulkan-20220424-test/"
        zf.writestr(folder + setup_ai.binary_name(), binary_body)
        zf.writestr(folder + "models/realesrgan-x4plus.param", "param")
        zf.writestr(folder + "models/realesrgan-x4plus.bin", "bin")
        zf.writestr(folder + "onepiece_demo.mp4", b"x" * 1000)
        zf.writestr("../escape.txt", "should not leave the install folder")
    return path


@pytest.fixture
def local_release(tmp_path, monkeypatch):
    if not setup_ai.supported():
        pytest.skip("no build for this platform")
    archive = fake_release(tmp_path)
    name, _, size = setup_ai.BUILDS[sys.platform]
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setitem(setup_ai.BUILDS, sys.platform, (name, digest, size))
    return archive.parent.as_uri() + "/"


def test_install_unpacks_and_is_found(local_release, monkeypatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv(upscale.ENV_VAR, raising=False)
    binary = setup_ai.install(url_base=local_release)
    target = setup_ai.install_dir()
    assert binary == target / setup_ai.binary_name()
    assert (target / "models" / "realesrgan-x4plus.param").exists()
    assert not (target / "onepiece_demo.mp4").exists()       # demo files skipped
    assert not (target.parent / "escape.txt").exists()       # no path escapes
    assert upscale.find_realesrgan() == str(binary)
    if sys.platform != "win32":
        assert binary.stat().st_mode & 0o111


def test_checksum_mismatch_installs_nothing(local_release, monkeypatch):
    name, _, size = setup_ai.BUILDS[sys.platform]
    monkeypatch.setitem(setup_ai.BUILDS, sys.platform, (name, "0" * 64, size))
    with pytest.raises(setup_ai.SetupError, match="checksum"):
        setup_ai.install(url_base=local_release)
    assert setup_ai.installed_binary() is None


def test_reinstall_replaces_the_old_copy(local_release):
    setup_ai.install(url_base=local_release)
    (setup_ai.install_dir() / "stale.txt").write_text("old")
    setup_ai.install(url_base=local_release)
    assert not (setup_ai.install_dir() / "stale.txt").exists()


def test_pinned_builds_cover_the_three_systems():
    assert set(setup_ai.BUILDS) == {"darwin", "win32", "linux"}
    for name, digest, size in setup_ai.BUILDS.values():
        assert name.endswith(".zip") and len(digest) == 64 and size > 40_000_000
