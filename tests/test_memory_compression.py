import numpy as np
import tempfile
import pytest
import os
from ojph import imwrite_to_memory, imread


def test_imwrite_to_memory_grayscale(tmp_path):
    test_image = np.random.randint(0, 256, (100, 150), dtype=np.uint8)

    compressed_data = imwrite_to_memory(test_image)

    # Check that it's a CompressedData object
    array = np.asarray(compressed_data)
    assert not array.flags.writeable
    assert len(compressed_data) > 0

    # Convert to bytes for file writing
    compressed_bytes = compressed_data.tobytes()

    filename = tmp_path / 'test.j2c'
    with open(filename, 'wb') as f:
        f.write(compressed_bytes)

    loaded_image = imread(filename)
    np.testing.assert_array_equal(loaded_image, test_image)


def test_imwrite_to_memory_color(tmp_path):
    test_image = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)

    compressed_data = imwrite_to_memory(test_image)

    # Check that it's a CompressedData object
    array = np.asarray(compressed_data)
    assert not array.flags.writeable
    assert len(compressed_data) > 0

    # Convert to bytes for file writing
    compressed_bytes = compressed_data.tobytes()

    filename = tmp_path / 'test.j2c'
    with open(filename, 'wb') as f:
        f.write(compressed_bytes)

    loaded_image = imread(filename)
    np.testing.assert_array_equal(loaded_image, test_image)


def test_imwrite_to_memory_channel_order(tmp_path):
    # Test HWC format
    test_image_hwc = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
    compressed_data_hwc = imwrite_to_memory(test_image_hwc, channel_order='HWC')

    # Test CHW format
    test_image_chw = np.random.randint(0, 256, (3, 100, 150), dtype=np.uint8)
    compressed_data_chw = imwrite_to_memory(test_image_chw, channel_order='CHW')

    # Both should produce valid compressed data
    assert len(compressed_data_hwc) > 0
    assert len(compressed_data_chw) > 0

    # Convert to bytes and verify they can be read back
    for compressed_data, test_image, is_chw in [
        (compressed_data_hwc, test_image_hwc, False),
        (compressed_data_chw, test_image_chw, True)
    ]:
        compressed_bytes = compressed_data.tobytes()

        filename = tmp_path / 'test.j2c'
        with open(filename, 'wb') as f:
            f.write(compressed_bytes)

        loaded_image = imread(filename)
        np.testing.assert_array_equal(loaded_image, test_image)


def test_imwrite_to_memory_16bit(tmp_path):
    test_image = np.random.randint(0, 65536, (100, 150), dtype=np.uint16)

    compressed_data = imwrite_to_memory(test_image)

    # Check that it's a CompressedData object
    array = np.asarray(compressed_data)
    assert not array.flags.writeable
    assert len(compressed_data) > 0

    # Convert to bytes for file writing
    compressed_bytes = compressed_data.tobytes()

    filename = tmp_path / 'test.j2c'
    with open(filename, 'wb') as f:
        f.write(compressed_bytes)

    loaded_image = imread(filename)
    np.testing.assert_array_equal(loaded_image, test_image)


def test_imwrite_to_memory_consistency():
    test_image = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)

    # Compress the same image multiple times
    compressed_data_1 = imwrite_to_memory(test_image)
    compressed_data_2 = imwrite_to_memory(test_image)

    # The compressed data should be identical (deterministic compression)
    np.testing.assert_array_equal(
        np.asarray(compressed_data_1),
        np.asarray(compressed_data_2)
    )
