import numpy as np
import pytest

from ojph._imwrite import imwrite_to_memory
from ojph._imread import imread_from_memory, OJPHImageFile


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


def test_read_image_out_preallocated_exact_shape():
    test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    compressed_data = imwrite_to_memory(test_image)

    reader = OJPHImageFile.from_memory(compressed_data)
    out = np.empty(reader.shape, dtype=reader.dtype)

    decoded = reader.read_image(out=out)

    assert decoded.shape == test_image.shape
    assert decoded.dtype == test_image.dtype
    np.testing.assert_array_equal(decoded, test_image)
    assert np.shares_memory(decoded, out)


def test_read_image_out_flat_array_view():
    test_image = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)
    compressed_data = imwrite_to_memory(test_image, channel_order="HWC")

    reader = OJPHImageFile.from_memory(compressed_data, channel_order="HWC")
    flat_out = np.empty(np.prod(reader.shape), dtype=reader.dtype)

    decoded = reader.read_image(out=flat_out)

    assert decoded.shape == test_image.shape
    np.testing.assert_array_equal(decoded, test_image)
    assert np.shares_memory(decoded, flat_out)


def test_read_image_out_wrong_dtype_raises():
    test_image = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
    compressed_data = imwrite_to_memory(test_image)

    reader = OJPHImageFile.from_memory(compressed_data)
    out = np.empty(reader.shape, dtype=np.float32)

    with pytest.raises(ValueError, match="dtype mismatch"):
        reader.read_image(out=out)


def test_imread_from_memory_non_reversible_no_negative_values():
    # This test will trigger overflow during the decoding process
    # since the reconstruction isn't exact
    base_value = 240
    noise_range = 15
    size = 128
    test_image = np.full((size, size), base_value, dtype=np.uint8)
    noise = np.random.randint(-noise_range, noise_range + 1, size=(size, size), dtype=np.int16)
    test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    compressed_data = imwrite_to_memory(
        test_image,
        reversible=False,
        qstep=0.005
    )
    decoded_image = imread_from_memory(compressed_data)

    assert decoded_image.shape == test_image.shape
    assert decoded_image.dtype == test_image.dtype
    assert np.all(decoded_image >= 50), (
        f"Found values below 50 in decoded image. "
        f"Min value: {decoded_image.min()}, "
        f"Values below 50: {np.sum(decoded_image < 50)}"
    )
