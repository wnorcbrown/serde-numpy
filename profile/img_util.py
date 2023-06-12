from io import BytesIO

import numpy as np
from PIL import Image


def get_random_img(size: int):
    return (np.random.rand(size, size, 3) * 256).astype(np.uint8)


def get_jpeg_bytes(img: np.ndarray) -> bytes:
    img = Image.fromarray(img)
    img_bytes = BytesIO()
    img.save(img_bytes, format='jpeg')
    return img_bytes.getvalue()


def get_png_bytes(img: np.ndarray) -> bytes:
    img = Image.fromarray(img)
    img_bytes = BytesIO()
    img.save(img_bytes, format='png')
    return img_bytes.getvalue()


def decode_pillow(byte_array: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(byte_array)))