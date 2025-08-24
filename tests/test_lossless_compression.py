from ojph import imwrite, imread
import numpy as np
import pytest


@pytest.mark.parametrize(
    'shape', [
        (200, 100),
        (1024, 256),
        (256, 1024),
        (1024, 1024),
        (1024, 2048),
        (4096, 2048),
        (4096, 4096),
        (8192, 1024),
        (8192, 8192),
    ]
)
def test_write_lossless_grayscale(shape, tmp_path):
    filename = tmp_path / 'test.jp2'
    data = np.random.randint(0, 256, shape, dtype=np.uint8)
    imwrite(filename, data)
    image_read = imread(filename)

    np.testing.assert_array_equal(data, image_read)


@pytest.mark.parametrize(
    'dtype', [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
    ]
)
def test_write_lossless_dtypes(dtype, tmp_path):
    filename = tmp_path / 'test.jp2'

    if dtype in [np.uint8, np.int8]:
        max_val = 127 if dtype == np.int8 else 255
        data = np.random.randint(0, max_val + 1, (100, 150), dtype=dtype)
    elif dtype in [np.uint16, np.int16]:
        max_val = 32767 if dtype == np.int16 else 65535
        data = np.random.randint(0, max_val + 1, (100, 150), dtype=dtype)
    else:  # uint32, int32
        max_val = 2147483647 if dtype == np.int32 else 4294967295
        data = np.random.randint(0, max_val + 1, (100, 150), dtype=dtype)

    imwrite(filename, data)
    image_read = imread(filename)

    np.testing.assert_array_equal(data, image_read)


@pytest.mark.parametrize(
    'shape', [
        (100, 150, 3),   # RGB
        (200, 300, 3),   # RGB larger
        (512, 512, 3),   # RGB square
        (100, 150, 4),   # RGBA
        (200, 300, 4),   # RGBA larger
        (512, 512, 4),   # RGBA square
    ]
)
def test_write_lossless_color(shape, tmp_path):
    filename = tmp_path / 'test.jp2'
    data = np.random.randint(0, 256, shape, dtype=np.uint8)
    imwrite(filename, data)
    image_read = imread(filename)

    np.testing.assert_array_equal(data, image_read)


@pytest.mark.parametrize(
    'dtype', [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
    ]
)
def test_write_lossless_color_dtypes(dtype, tmp_path):
    filename = tmp_path / 'test.jp2'

    if dtype in [np.uint8, np.int8]:
        max_val = 127 if dtype == np.int8 else 255
        data = np.random.randint(0, max_val + 1, (100, 150, 3), dtype=dtype)
    elif dtype in [np.uint16, np.int16]:
        max_val = 32767 if dtype == np.int16 else 65535
        data = np.random.randint(0, max_val + 1, (100, 150, 3), dtype=dtype)
    else:  # uint32, int32
        max_val = 2147483647 if dtype == np.int32 else 4294967295
        data = np.random.randint(0, max_val + 1, (100, 150, 3), dtype=dtype)

    imwrite(filename, data)
    image_read = imread(filename)

    np.testing.assert_array_equal(data, image_read)


@pytest.mark.parametrize(
    'shape', [
        (100, 150),
        (200, 300),
        (512, 512),
        (100, 150, 3),
        (200, 300, 3),
        (512, 512, 3),
        (100, 150, 4),
        (200, 300, 4),
        (512, 512, 4),
    ]
)
def test_write_lossless_edge_cases(shape, tmp_path):
    filename = tmp_path / 'test.jp2'

    # Test with zeros
    data_zeros = np.zeros(shape, dtype=np.uint8)
    imwrite(filename, data_zeros)
    image_read = imread(filename)
    np.testing.assert_array_equal(data_zeros, image_read)

    # Test with ones
    data_ones = np.ones(shape, dtype=np.uint8)
    imwrite(filename, data_ones)
    image_read = imread(filename)
    np.testing.assert_array_equal(data_ones, image_read)

    # Test with max values
    data_max = np.full(shape, 255, dtype=np.uint8)
    imwrite(filename, data_max)
    image_read = imread(filename)
    np.testing.assert_array_equal(data_max, image_read)


def test_write_lossless_large_images(tmp_path):
    filename = tmp_path / 'test.jp2'

    # Test large grayscale image
    data_gray = np.random.randint(0, 256, (2048, 2048), dtype=np.uint8)
    imwrite(filename, data_gray)
    image_read = imread(filename)
    np.testing.assert_array_equal(data_gray, image_read)

    # Test large RGB image
    data_rgb = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    imwrite(filename, data_rgb)
    image_read = imread(filename)
    np.testing.assert_array_equal(data_rgb, image_read)


def test_write_lossless_small_images(tmp_path):
    filename = tmp_path / 'test.jp2'

    # Test very small images
    data_tiny = np.random.randint(0, 256, (1, 1), dtype=np.uint8)
    imwrite(filename, data_tiny)
    image_read = imread(filename)
    np.testing.assert_array_equal(data_tiny, image_read)

    # Test small RGB
    data_tiny_rgb = np.random.randint(0, 256, (1, 1, 3), dtype=np.uint8)
    imwrite(filename, data_tiny_rgb)
    image_read = imread(filename)
    np.testing.assert_array_equal(data_tiny_rgb, image_read)

    # Test small RGBA
    data_tiny_rgba = np.random.randint(0, 256, (1, 1, 4), dtype=np.uint8)
    imwrite(filename, data_tiny_rgba)
    image_read = imread(filename)
    np.testing.assert_array_equal(data_tiny_rgba, image_read)
