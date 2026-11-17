"""The photo model set-up: reading PyTorch weights without PyTorch, and
writing them in the ncnn layout of the Real-ESRGAN graph."""

import io
import pickle
import struct
import sys
import types
import zipfile

import numpy as np
import pytest

from a4print import setup_ai


class _Storage:
    def __init__(self, key, array):
        self.key, self.array = key, array


def _fake_torch_modules(monkeypatch):
    """Stand-ins so pickle can write references to torch globals."""
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch._utils")

    def _rebuild_tensor_v2(*args):
        raise AssertionError("only used for pickling")
    _rebuild_tensor_v2.__module__, _rebuild_tensor_v2.__qualname__ = "torch._utils", "_rebuild_tensor_v2"
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2

    class FloatStorage:
        pass
    FloatStorage.__module__, FloatStorage.__qualname__ = "torch", "FloatStorage"
    torch.FloatStorage, torch._utils = FloatStorage, utils
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch._utils", utils)
    return torch


def write_checkpoint(path, state, monkeypatch, extra=None):
    """A .pth file in PyTorch's zip format, written without PyTorch."""
    torch = _fake_torch_modules(monkeypatch)
    storages = []

    class Tensor:
        def __init__(self, array):
            self.storage = _Storage(str(len(storages)), np.ascontiguousarray(array, "<f4"))
            storages.append(self.storage)
            self.shape = array.shape

        def __reduce__(self):
            stride = tuple(int(s) // 4 for s in self.storage.array.strides)
            return torch._utils._rebuild_tensor_v2, (self.storage, 0, self.shape, stride, False, None)

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, _Storage):
                return ("storage", torch.FloatStorage, obj.key, "cpu", obj.array.size)
            return None

    buf = io.BytesIO()
    Pickler(buf, protocol=2).dump({"params_ema": {k: Tensor(v) for k, v in state.items()}, **(extra or {})})
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
        for s in storages:
            zf.writestr(f"archive/data/{s.key}", s.array.tobytes())


GRAPH = """7767517
4 4
Input            input.1   0 1 data
Convolution      Conv_0    1 1 data a 0=2 1=3 4=1 5=1 6=54
Convolution      Conv_1    1 1 a b 0=3 1=3 4=1 5=1 6=54 9=2 -23310=1,2.000000e-01
"""


def test_weights_are_written_in_ncnn_order_and_format(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    state = {"conv_first.weight": rng.normal(size=(2, 3, 3, 3)), "conv_first.bias": rng.normal(size=2),
             "conv_last.weight": rng.normal(size=(3, 2, 3, 3)), "conv_last.bias": rng.normal(size=3)}
    write_checkpoint(tmp_path / "w.pth", state, monkeypatch)
    read = setup_ai._read_pth(tmp_path / "w.pth")
    assert list(read) == list(state)
    for k in state:
        assert np.allclose(read[k], state[k].astype(np.float32))

    (tmp_path / "g.param").write_bytes(GRAPH.encode())
    setup_ai._write_ncnn_bin(read, tmp_path / "g.param", tmp_path / "m.bin")
    data, pos = (tmp_path / "m.bin").read_bytes(), 0
    for name in ("conv_first", "conv_last"):
        w, b = state[name + ".weight"], state[name + ".bias"]
        assert struct.unpack_from("<I", data, pos)[0] == 0x01306B47
        pos += 4
        assert np.allclose(np.frombuffer(data, "<f2", w.size, pos), w.ravel(), atol=2e-3)
        pos += w.size * 2 + (w.size * 2) % 4
        assert np.allclose(np.frombuffer(data, "<f4", b.size, pos), b)
        pos += b.size * 4
    assert pos == len(data)


def test_weights_that_dont_fit_the_graph_are_refused(tmp_path, monkeypatch):
    state = {"conv_first.weight": np.zeros((4, 3, 3, 3)), "conv_first.bias": np.zeros(4),
             "conv_last.weight": np.zeros((3, 4, 3, 3)), "conv_last.bias": np.zeros(3)}
    write_checkpoint(tmp_path / "w.pth", state, monkeypatch)
    (tmp_path / "g.param").write_bytes(GRAPH.encode())
    with pytest.raises(setup_ai.SetupError):
        setup_ai._write_ncnn_bin(setup_ai._read_pth(tmp_path / "w.pth"), tmp_path / "g.param", tmp_path / "m.bin")


class _Evil:
    def __reduce__(self):
        import os
        return os.system, ("echo pwned > /dev/null",)


def test_a_model_file_cant_run_code(tmp_path, monkeypatch):
    path = tmp_path / "evil.pth"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps({"params_ema": _Evil()}, protocol=2))
    with pytest.raises(setup_ai.SetupError, match="unexpected content"):
        setup_ai._read_pth(path)


def test_install_photo_model_checks_the_result(tmp_path, monkeypatch):
    state = {"conv_first.weight": np.ones((2, 3, 3, 3)), "conv_first.bias": np.ones(2),
             "conv_last.weight": np.ones((3, 2, 3, 3)), "conv_last.bias": np.ones(3)}
    write_checkpoint(tmp_path / "w.pth", state, monkeypatch)
    setup_ai.models_dir().mkdir(parents=True)
    (setup_ai.models_dir() / setup_ai.GRAPH).write_bytes(GRAPH.encode())
    import hashlib
    monkeypatch.setattr(setup_ai, "PHOTO_WEIGHTS",
                        ("", hashlib.sha256((tmp_path / "w.pth").read_bytes()).hexdigest(), 0))
    url = (tmp_path / "w.pth").as_uri()
    # Right download, but the converted file isn't the pinned one: refused.
    with pytest.raises(setup_ai.SetupError, match="converted"):
        setup_ai.install_photo_model(url=url)
    assert not setup_ai.photo_model_installed()

    (tmp_path / "g.param").write_bytes(GRAPH.encode())
    setup_ai._write_ncnn_bin(setup_ai._read_pth(tmp_path / "w.pth"), tmp_path / "g.param", tmp_path / "m.bin")
    monkeypatch.setattr(setup_ai, "PHOTO_BIN_SHA256", hashlib.sha256((tmp_path / "m.bin").read_bytes()).hexdigest())
    # Right files, but a graph other than the official one: refused.
    with pytest.raises(setup_ai.SetupError, match="graph"):
        setup_ai.install_photo_model(url=url)
    monkeypatch.setattr(setup_ai, "GRAPH_SHA256", hashlib.sha256(GRAPH.encode()).hexdigest())
    setup_ai.install_photo_model(url=url)
    assert setup_ai.photo_model_installed()


def test_photo_model_needs_real_esrgan_first(tmp_path, monkeypatch):
    monkeypatch.delenv("A4PRINT_REALESRGAN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(setup_ai.SetupError, match="install Real-ESRGAN first"):
        setup_ai.install_photo_model(url="file:///nonexistent")


def test_a_tensor_reaching_past_its_data_is_refused():
    storage = np.zeros(10, "<f4")
    with pytest.raises(setup_ai.SetupError, match="damaged"):
        setup_ai._rebuild_tensor(storage, 0, (4, 4), (4, 1))
    assert setup_ai._rebuild_tensor(storage, 2, (2, 4), (4, 1)).shape == (2, 4)
