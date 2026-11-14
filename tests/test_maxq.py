"""Maximum quality mode. Real models need PyTorch; these tests use a tiny
stand-in network when PyTorch is installed and check the plumbing otherwise."""

import hashlib

import pytest

from a4print import cli, maxq, setup_ai


def test_model_list_is_pinned_and_credited():
    assert set(maxq.MODELS) == {"photo", "artwork"}
    for m in maxq.MODELS.values():
        assert m.url.startswith("https://github.com/") and len(m.sha256) == 64 and m.size > 1_000_000
        assert "CC BY 4.0" in m.credit or "BSD-3-Clause" in m.credit


def test_cli_explains_missing_pytorch(monkeypatch, tmp_path, save, portrait, capsys):
    monkeypatch.setattr(maxq, "libraries_available", lambda: False)
    assert cli.main([str(save(portrait, "p.png")), "-o", str(tmp_path / "o"), "--max-quality"]) == 3
    assert "pipx inject a4print torch spandrel" in capsys.readouterr().err


def test_cli_explains_missing_models(monkeypatch, tmp_path, save, portrait, capsys):
    monkeypatch.setattr(maxq, "libraries_available", lambda: True)
    assert cli.main([str(save(portrait, "p.png")), "-o", str(tmp_path / "o"), "--max-quality"]) == 3
    assert "a4print-setup-ai --max" in capsys.readouterr().err


def test_install_models_checks_checksums(tmp_path, monkeypatch):
    blobs = {}
    for kind in maxq.MODELS:
        data = f"weights for {kind}".encode()
        (tmp_path / kind).write_bytes(data)
        blobs[kind] = data
        m = maxq.MODELS[kind]
        monkeypatch.setitem(maxq.MODELS, kind, maxq.Model(m.file, (tmp_path / kind).as_uri(),
                                                         hashlib.sha256(data).hexdigest(), len(data), m.credit))
    maxq.install_models()
    assert maxq.models_installed()

    # A file that doesn't match its checksum is rejected and leaves nothing behind.
    m = maxq.MODELS["photo"]
    (maxq.models_dir() / m.file).unlink()
    monkeypatch.setitem(maxq.MODELS, "photo", maxq.Model(m.file, m.url, "0" * 64, m.size, m.credit))
    with pytest.raises(setup_ai.SetupError, match="checksum"):
        maxq.install_models()
    assert not (maxq.models_dir() / m.file).exists()
    assert not list(maxq.models_dir().glob("*.part"))
