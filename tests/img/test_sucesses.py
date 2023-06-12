import pytest
from io import BytesIO

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.filters import gaussian

from serde_numpy import decode_jpeg, decode_png

from .fixtures import img_array



@pytest.fixture
def jpeg_bytes() -> bytes:
    img = Image.fromarray(img_array)
    img_bytes = BytesIO()
    img.save(img_bytes, format='jpeg')
    return img_bytes.getvalue()


@pytest.fixture
def grayscale_jpeg_bytes() -> bytes:
    img = Image.fromarray(img_array)
    img = img.convert('L')
    img_bytes = BytesIO()
    img.save(img_bytes, format='jpeg')
    return img_bytes.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    img = Image.fromarray(img_array)
    img_bytes = BytesIO()
    img.save(img_bytes, format='png')
    return img_bytes.getvalue()


@pytest.fixture
def grayscale_png_bytes() -> bytes:
    img = Image.fromarray(img_array)
    img = img.convert('L')
    img_bytes = BytesIO()
    img.save(img_bytes, format='png')
    return img_bytes.getvalue()


@pytest.fixture
def rgba_png_bytes() -> bytes:
    img = Image.fromarray(img_array)
    img = img.convert('RGBA')
    img_bytes = BytesIO()
    img.save(img_bytes, format='png')
    return img_bytes.getvalue()


def lab_difference(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute the L*a*b* difference between two RGB images of shape [height, width, 3].]"""
    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    return np.sqrt(np.sum((lab1 - lab2)**2, axis=-1)).mean()


def test_decode_jpeg(jpeg_bytes: bytes):
    assert decode_jpeg(jpeg_bytes) is not None
    assert lab_difference(gaussian(img_array, 5, channel_axis=-1), gaussian(decode_jpeg(jpeg_bytes), 5, channel_axis=-1)) < 2.3
    # 2.3 is considered just percentible distance, jpeg adds percetible artifacts hence the blurring


def test_decode_jpeg_grayscale(grayscale_jpeg_bytes: bytes):
    assert decode_jpeg(grayscale_jpeg_bytes) is not None


def test_decode_png(png_bytes: bytes):
    assert decode_png(png_bytes) is not None
    assert np.allclose(img_array, decode_png(png_bytes), atol=1)


def test_decode_png_grayscale(grayscale_png_bytes: bytes):
    assert decode_png(grayscale_png_bytes) is not None


def test_decode_png_rgba(rgba_png_bytes: bytes):
    assert decode_png(rgba_png_bytes) is not None