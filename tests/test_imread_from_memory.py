import numpy as np
import pytest

from ojph._imwrite import imwrite_to_memory
from ojph._imread import imread_from_memory


def test_imread_from_memory():
    test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    compressed_data = imwrite_to_memory(test_image)
    decoded_image = imread_from_memory(compressed_data)

    assert np.array_equal(test_image, decoded_image)
    assert decoded_image.shape == test_image.shape
    assert decoded_image.dtype == test_image.dtype


def test_imread_from_memory_with_bytes():
    test_image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    compressed_data = imwrite_to_memory(test_image)
    compressed_bytes = bytes(compressed_data)

    decoded_image = imread_from_memory(compressed_bytes)

    assert np.array_equal(test_image, decoded_image)
    assert decoded_image.shape == test_image.shape
    assert decoded_image.dtype == test_image.dtype


def test_imread_from_memory_rgb():
    test_image = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)

    compressed_data = imwrite_to_memory(test_image, channel_order='HWC')
    decoded_image = imread_from_memory(compressed_data)

    assert np.array_equal(test_image, decoded_image)
    assert decoded_image.shape == test_image.shape
    assert decoded_image.dtype == test_image.dtype


@pytest.mark.parametrize('size', [(16, 16), (64, 64), (128, 128), (256, 256)])
def test_imread_from_memory_different_sizes(size):
    test_image = np.random.randint(0, 256, size, dtype=np.uint8)
    compressed_data = imwrite_to_memory(test_image)
    decoded_image = imread_from_memory(compressed_data)

    assert np.array_equal(test_image, decoded_image)
    assert decoded_image.shape == test_image.shape


@pytest.mark.parametrize('dtype', [
    np.uint8, np.uint16, np.uint32,
    np.int8, np.int16, np.int32,
])
def test_imread_from_memory_different_dtypes(dtype):
    test_image = np.random.randint(
        np.iinfo(dtype).min, np.iinfo(dtype).max, (64, 64), dtype=dtype)
    compressed_data = imwrite_to_memory(test_image)
    decoded_image = imread_from_memory(compressed_data)

    assert np.array_equal(test_image, decoded_image)
    assert decoded_image.dtype == test_image.dtype


def test_imread_from_memory_invalid_input():
    with pytest.raises(ValueError):
        imread_from_memory(np.array([[1, 2], [3, 4]]))  # 2D array instead of 1D

    with pytest.raises(ValueError):
        imread_from_memory("not valid data")  # Invalid data type
