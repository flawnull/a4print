"""Maximum quality with a tiny stand-in network (needs PyTorch)."""

import numpy as np
import pytest

from a4print import maxq
from a4print.pipeline import Options, prepare

from .conftest import gradient, with_noise

torch = pytest.importorskip("torch")


class FakeNet(torch.nn.Module):
    """4x nearest-neighbour upscale plus a small offset, so output is
    recognizable; stands in for a real model."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest") + self.w


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(maxq, "libraries_available", lambda: True)
    monkeypatch.setattr(maxq, "models_installed", lambda: True)
    monkeypatch.setattr(maxq, "_model", lambda kind: (FakeNet(), 4))


def test_tiles_cover_the_image_exactly(fake_models):
    img = (np.arange(130 * 170 * 3) % 251).astype(np.uint8).reshape(130, 170, 3)
    out = maxq.upscale_x4(img, tile=48, overlap=16)
    expected = img.repeat(4, axis=0).repeat(4, axis=1)
    assert out.shape == expected.shape and np.array_equal(out, expected)


def test_gray_images_stay_gray(fake_models):
    out = maxq.upscale_x4(np.full((20, 30), 90, np.uint8))
    assert out.shape == (80, 120) and (out == 90).all()


def test_pipeline_picks_model_by_content(fake_models, monkeypatch):
    used = []
    real = maxq.upscale_x4
    monkeypatch.setattr(maxq, "upscale_x4", lambda arr, kind="photo", **k: used.append(kind) or real(arr, kind))
    _, report = prepare(with_noise(gradient(300, 200), 6), Options(max_quality=True, enhance=False))
    assert used[-1] == "photo" and "maximum quality: RealESRNet" in report.resize_method

    from .test_ai_quality import flat_graphic
    _, report = prepare(flat_graphic(), Options(max_quality=True, enhance=False))
    assert used[-1] == "artwork" and "maximum quality: Real-ESRGAN" in report.resize_method
