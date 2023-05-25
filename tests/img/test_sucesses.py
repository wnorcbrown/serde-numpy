import pytest
from io import BytesIO

from PIL import Image

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


def test_decode_jpeg(jpeg_bytes: bytes):
    assert decode_jpeg(jpeg_bytes) is not None


def test_decode_jpeg_grayscale(grayscale_jpeg_bytes: bytes):
    assert decode_jpeg(grayscale_jpeg_bytes) is not None


def test_decode_png(png_bytes: bytes):
    assert decode_png(png_bytes) is not None


def test_decode_png_grayscale(grayscale_png_bytes: bytes):
    assert decode_png(grayscale_png_bytes) is not None


def test_decode_png_rgba(rgba_png_bytes: bytes):
    assert decode_png(rgba_png_bytes) is not None