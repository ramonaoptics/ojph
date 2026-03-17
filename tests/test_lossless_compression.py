from ojph import imwrite, imread
import numpy as np
import pytest


def _mse(a, b):
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


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
    'shape', [
        (200, 100),
        (1024, 256),
        (256, 1024),
    ]
)
@pytest.mark.parametrize(
    'use_wavelet_oneXone', [False, True],
)
def test_write_lossless_grayscale_wavelet_oneXone(shape, use_wavelet_oneXone, tmp_path):
    filename = tmp_path / 'test_wavelet_oneXone.jp2'
    data = np.random.randint(0, 256, shape, dtype=np.uint8)
    imwrite(filename, data, wavelet_oneXone=use_wavelet_oneXone)
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
    image_read = imread(filename, channel_order='HWC')

    np.testing.assert_array_equal(data, image_read)


@pytest.mark.parametrize(
    'channel_order', ['HWC', 'CHW'],
)
@pytest.mark.parametrize(
    'chroma', ['rgb', 'rgba'],
)
def test_write_lossless_channel_order(channel_order, chroma, tmp_path):
    filename = tmp_path / 'test.jp2'

    if channel_order == 'HWC':
        # Create HWC data: (height, width, channels)
        data = np.random.randint(0, 256, (100, 150, 4), dtype=np.uint8)
        if chroma == 'rgb':
            data = data[:, :, :3]
        else:
            data = data[:, :, :4]
    else:  # CHW
        # Create CHW data: (channels, height, width)
        data = np.random.randint(0, 256, (4, 100, 150), dtype=np.uint8)
        if chroma == 'rgb':
            data = data[:3, :, :]
        else:
            data = data[:4, :, :]

    imwrite(filename, data, channel_order=channel_order)
    image_read = imread(filename, channel_order=channel_order).reshape(data.shape)

    # Only HWC RGB/RGBA images get color transform and return HWC format
    # CHW images preserve their original format
    if channel_order == 'HWC':
        # HWC input -> HWC output (with color transform optimization)
        np.testing.assert_array_equal(data, image_read)
    else:  # CHW
        # CHW input -> CHW output (preserves format)
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


def test_wavelet_oneXone_irreversible_lossy(tmp_path):
    filename = tmp_path / 'test_wavelet_oneXone_lossy.jp2'
    data = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    imwrite(filename, data, reversible=False, wavelet_oneXone=True, qstep=0.1)
    image_read = imread(filename)

    assert image_read.shape == data.shape
    mse = _mse(data, image_read)
    assert mse >= 0.0

    imwrite(filename, data, reversible=True, wavelet_oneXone=True)
    image_read = imread(filename)

    assert image_read.shape == data.shape
    np.testing.assert_array_equal(data, image_read)


def test_wavelet_oneXone_irreversible_mse_monotonic(tmp_path):
    filename = tmp_path / 'test_wavelet_oneXone_irrev_mse.jp2'
    data = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    prev_mse = None
    for q in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
        imwrite(filename, data, reversible=False, wavelet_oneXone=True, qstep=q)
        image_read = imread(filename)
        assert image_read.shape == data.shape
        mse = _mse(data, image_read)
        assert mse >= 0.0
        if prev_mse is not None:
            assert mse >= prev_mse
        prev_mse = mse


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


def test_ll_levels_standard_seed0(tmp_path):
    filename = tmp_path / "seed0_levels.jp2"
    filename_r1x1 = tmp_path / "seed0_levels_r1x1.jp2"

    h, w = 1024, 1024
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    data = np.zeros((h, w), dtype=np.uint8)
    data[(x - cx) ** 2 + (y - cy) ** 2 <= r * r] = 255

    imwrite(filename, data)
    ll = imread(filename, level=0)
    assert ll.shape == data.shape
    assert np.array_equal(ll, data)

    for level in (1, 2, 3, 4):
        ll = imread(filename, level=level)
        expected = data[:: 2 ** level, :: 2 ** level]
        assert ll.shape == expected.shape
        assert not np.array_equal(ll, expected)

    imwrite(filename_r1x1, data, wavelet_oneXone=True)
    for level in (1, 2, 3, 4):
        ll = imread(filename_r1x1, skipped_res_for_data=level, skipped_res_for_recon=level)
        expected = data[:: 2 ** level, :: 2 ** level]
        assert ll.shape == expected.shape
        np.testing.assert_array_equal(ll, expected)


def test_channel_order_validation(tmp_path):
    filename = tmp_path / 'test.jp2'

    # Test invalid channel order
    data = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Invalid channel_order"):
        imwrite(filename, data, channel_order='WHC')

    with pytest.raises(ValueError, match="Invalid channel_order"):
        imwrite(filename, data, channel_order='CWH')

    # Test channel order mismatch with dimensions
    with pytest.raises(ValueError, match="must be consistent"):
        imwrite(filename, data, channel_order='HW')

    # Test 2D image with 3D channel order
    data_2d = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
    with pytest.raises(ValueError, match="must be consistent"):
        imwrite(filename, data_2d, channel_order='HWC')


@pytest.mark.parametrize(
    'num_decompositions', [None, 3, 5, 7]
)
def test_num_decompositions(num_decompositions, tmp_path):
    from ojph.ojph_bindings import J2CInfile, Codestream

    filename = tmp_path / 'test.jp2'
    data = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

    imwrite(filename, data, num_decompositions=num_decompositions)

    if num_decompositions is None:
        num_decompositions = 5
    for level in range(num_decompositions):
        imread(filename, level=level)

    from ojph._imread import OJPHImageFile
    f = OJPHImageFile(filename)
    assert f.levels == num_decompositions


def test_lossless_vs_lossy_error_increase(tmp_path):
    rng = np.random.default_rng(123)
    data = rng.integers(0, 256, (256, 256), dtype=np.uint8)

    lossless_filename = tmp_path / 'lossless.jp2'
    imwrite(lossless_filename, data, reversible=True)
    lossless = imread(lossless_filename)
    np.testing.assert_array_equal(data, lossless)

    low_qstep_filename = tmp_path / 'lossy_low.jp2'
    imwrite(low_qstep_filename, data, reversible=False, qstep=0.002)
    low_qstep = imread(low_qstep_filename)

    high_qstep_filename = tmp_path / 'lossy_high.jp2'
    imwrite(high_qstep_filename, data, reversible=False, qstep=0.01)
    high_qstep = imread(high_qstep_filename)

    mse_low = _mse(data, low_qstep)
    mse_high = _mse(data, high_qstep)

    assert mse_low > 0.0
    assert mse_high > mse_low
