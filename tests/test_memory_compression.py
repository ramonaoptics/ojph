import numpy as np
import tempfile
import pytest
import os
from ojph import imwrite_to_memory, imread


def _mse(a, b):
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


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
    for compressed_data, test_image, channel_order in [
        (compressed_data_hwc, test_image_hwc, "HWC"),
        (compressed_data_chw, test_image_chw, "CHW")
    ]:
        compressed_bytes = compressed_data.tobytes()

        filename = tmp_path / 'test.j2c'
        with open(filename, 'wb') as f:
            f.write(compressed_bytes)

        loaded_image = imread(filename, channel_order=channel_order)
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


def test_imwrite_3d_single_channel_as_monochrome(tmp_path):
    test_image = np.random.randint(0, 256, (100, 150, 1), dtype=np.uint8)

    compressed_data = imwrite_to_memory(test_image)

    filename = tmp_path / 'test.j2c'
    with open(filename, 'wb') as f:
        f.write(compressed_data.tobytes())

    loaded_image = imread(filename)
    expected_image = test_image.squeeze()
    np.testing.assert_array_equal(loaded_image, expected_image)


def test_lossless_vs_lossy_memory_error_increase(tmp_path):
    rng = np.random.default_rng(321)
    test_image = rng.integers(0, 256, (256, 256), dtype=np.uint8)

    lossless_data = imwrite_to_memory(test_image, reversible=True)
    lossless_bytes = lossless_data.tobytes()
    lossless_filename = tmp_path / 'lossless.j2c'
    with open(lossless_filename, 'wb') as f:
        f.write(lossless_bytes)
    lossless = imread(lossless_filename)
    np.testing.assert_array_equal(lossless, test_image)

    low_qstep_data = imwrite_to_memory(test_image, reversible=False, qstep=0.002)
    low_qstep_bytes = low_qstep_data.tobytes()
    low_qstep_filename = tmp_path / 'lossy_low.j2c'
    with open(low_qstep_filename, 'wb') as f:
        f.write(low_qstep_bytes)
    low_qstep = imread(low_qstep_filename)

    high_qstep_data = imwrite_to_memory(test_image, reversible=False, qstep=0.01)
    high_qstep_bytes = high_qstep_data.tobytes()
    high_qstep_filename = tmp_path / 'lossy_high.j2c'
    with open(high_qstep_filename, 'wb') as f:
        f.write(high_qstep_bytes)
    high_qstep = imread(high_qstep_filename)

    mse_low = _mse(test_image, low_qstep)
    mse_high = _mse(test_image, high_qstep)

    assert mse_low > 0.0
    assert mse_high > mse_low
