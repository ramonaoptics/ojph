"""Tests for the rev12 (previous-sample predict-only, 1/2) wavelet.

Like ``wavelet='rev13'``, every resolution level decodes to exactly
``image[::2**r, ::2**r]`` and full resolution is lossless.  The high-pass
subbands hold exact previous-sample differences, which compress better for
mask-like images.  The kernel is signaled as a JPEG 2000 Part 2 arbitrary
(ARB) filter, so decoding requires an OpenJPH build with ARB support.
"""
from ojph import imwrite, imread, imwrite_to_memory, imread_from_memory
from ojph._imread import OJPHImageFile
import numpy as np
import pytest


def _mask_image(shape, rng):
    img = np.zeros(shape, dtype=np.uint8)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for _ in range(5):
        value = rng.integers(1, 201)
        r = rng.integers(min(shape) // 16, min(shape) // 4)
        cy = rng.integers(0, shape[0])
        cx = rng.integers(0, shape[1])
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2] = value
    return img


@pytest.mark.parametrize(
    'shape', [
        (256, 256),
        (255, 193),
        (200, 100),
        (1024, 256),
    ]
)
def test_rev12_lossless_full_resolution(shape, tmp_path):
    filename = tmp_path / 'test.j2c'
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, shape, dtype=np.uint8)
    imwrite(filename, data, wavelet='rev12')

    np.testing.assert_array_equal(data, imread(filename))


@pytest.mark.parametrize('shape', [(256, 256), (255, 193), (1024, 512)])
def test_rev12_levels_are_exact_subsampling(shape, tmp_path):
    filename = tmp_path / 'test.j2c'
    rng = np.random.default_rng(42)
    data = _mask_image(shape, rng)
    num_decompositions = 5
    imwrite(filename, data, wavelet='rev12',
            num_decompositions=num_decompositions)

    for level in range(1, num_decompositions + 1):
        image_read = imread(filename, level=level)
        subsampled = data[:: 2 ** level, :: 2 ** level]
        np.testing.assert_array_equal(subsampled, image_read)
        assert set(np.unique(image_read)) <= set(np.unique(data))


def test_rev12_in_memory_roundtrip_and_detection():
    rng = np.random.default_rng(42)
    data = _mask_image((512, 512), rng)
    buffer = imwrite_to_memory(data, wavelet='rev12')

    np.testing.assert_array_equal(data, imread_from_memory(buffer))
    np.testing.assert_array_equal(
        data[::4, ::4], imread_from_memory(buffer, level=2))
    assert OJPHImageFile.from_memory(buffer).is_predict_only


def test_rev12_compresses_masks_better_than_rev13():
    rng = np.random.default_rng(42)
    data = _mask_image((1024, 1024), rng)
    size12 = len(bytes(imwrite_to_memory(data, wavelet='rev12')))
    size13 = len(bytes(imwrite_to_memory(data, wavelet='rev13')))
    assert size12 < size13


def test_rev12_dtypes(tmp_path):
    filename = tmp_path / 'test.j2c'
    rng = np.random.default_rng(42)
    data = rng.integers(0, 4096, (200, 150)).astype(np.uint16)
    imwrite(filename, data, wavelet='rev12', num_decompositions=3)

    np.testing.assert_array_equal(data, imread(filename))
    np.testing.assert_array_equal(data[::2, ::2], imread(filename, level=1))


def test_rev12_wavelet_validation():
    data = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match='contradicts'):
        imwrite_to_memory(data, wavelet='rev12', reversible=False)
