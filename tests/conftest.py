"""Generated sample images. No real photos or posters are used."""

import cv2
import numpy as np
import pytest
from PIL import Image


def gradient(w, h):
    y, x = np.mgrid[0:h, 0:w]
    return np.dstack([x / w * 255, y / h * 255, np.full_like(x, 120)]).astype(np.uint8)


def with_noise(arr, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """Keep tests away from a real ~/.a4print (an installed Real-ESRGAN)."""
    monkeypatch.setenv("A4PRINT_HOME", str(tmp_path_factory.mktemp("a4print_home")))


@pytest.fixture
def portrait():
    """600 x 900 gradient with a red disc in the middle."""
    img = gradient(600, 900)
    cv2.circle(img, (300, 450), 150, (220, 40, 40), -1)
    return img


@pytest.fixture
def landscape():
    """1500 x 1000 gradient."""
    return gradient(1500, 1000)


@pytest.fixture
def bw_photo():
    """Gray gradient with a tiny blue tint, like a scanned B&W print."""
    g = np.tile(np.linspace(20, 235, 700).astype(np.uint8), (1000, 1))
    img = np.dstack([g, g, g]).astype(np.int16)
    img[..., 2] += 2
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture
def save(tmp_path):
    def _save(arr, name):
        path = tmp_path / name
        Image.fromarray(arr).save(path)
        return path

    return _save
