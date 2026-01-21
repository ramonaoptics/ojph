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


@pytest.mark.parametrize('size', [(96, 96), (97, 97), (100, 100), (127, 127), (128, 128)])
def test_get_level_shape(size):
    """Test that get_level_shape returns correct dimensions at each level."""
    test_image = np.random.randint(0, 256, size, dtype=np.uint8)
    compressed_data = imwrite_to_memory(test_image, num_decompositions=5)

    reader = OJPHImageFile.from_memory(compressed_data)

    # Level 0 should match the original shape
    assert reader.get_level_shape(0) == size

    # Test each level and verify by actually reading at that level
    for level in range(1, reader.levels + 1):
        expected_shape = reader.get_level_shape(level)

        # Create a new reader to actually read at this level
        reader2 = OJPHImageFile.from_memory(compressed_data)
        actual_image = reader2.read_image(level=level)

        assert actual_image.shape == expected_shape, (
            f"Level {level}: get_level_shape returned {expected_shape} "
            f"but read_image returned shape {actual_image.shape}"
        )


def test_get_level_shape_ceiling_division():
    """Test that get_level_shape uses ceiling division for odd dimensions."""
    # 97 is chosen because it doesn't divide evenly by powers of 2
    test_image = np.random.randint(0, 256, (97, 97), dtype=np.uint8)
    compressed_data = imwrite_to_memory(test_image, num_decompositions=5)

    reader = OJPHImageFile.from_memory(compressed_data)

    # Expected shapes using ceiling division: ceil(97 / 2^level)
    expected_shapes = {
        0: (97, 97),
        1: (49, 49),  # ceil(97/2) = 49, not 48
        2: (25, 25),  # ceil(97/4) = 25, not 24
        3: (13, 13),  # ceil(97/8) = 13, not 12
        4: (7, 7),    # ceil(97/16) = 7, not 6
        5: (4, 4),    # ceil(97/32) = 4, not 3
    }

    for level, expected in expected_shapes.items():
        assert reader.get_level_shape(level) == expected, (
            f"Level {level}: expected {expected}, got {reader.get_level_shape(level)}"
        )


def test_get_level_shape_invalid_level():
    """Test that get_level_shape raises errors for invalid levels."""
    test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    compressed_data = imwrite_to_memory(test_image, num_decompositions=3)

    reader = OJPHImageFile.from_memory(compressed_data)

    with pytest.raises(ValueError, match="level must be >= 0"):
        reader.get_level_shape(-1)

    with pytest.raises(ValueError, match="cannot be greater than"):
        reader.get_level_shape(reader.levels + 1)


def test_get_level_shape_rgb():
    """Test get_level_shape with multi-channel images."""
    test_image = np.random.randint(0, 256, (97, 97, 3), dtype=np.uint8)
    compressed_data = imwrite_to_memory(test_image, channel_order='HWC', num_decompositions=3)

    reader = OJPHImageFile.from_memory(compressed_data, channel_order='HWC')

    # Level 0 should include channel dimension
    assert reader.get_level_shape(0) == (97, 97, 3)

    # Other levels should also include channel dimension
    assert reader.get_level_shape(1) == (49, 49, 3)
    assert reader.get_level_shape(2) == (25, 25, 3)
