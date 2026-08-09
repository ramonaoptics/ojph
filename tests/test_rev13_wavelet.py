"""Tests for the rev13 (reversible predict-only, 1/3) wavelet.

With ``wavelet='rev13'`` the low-pass subband of every decomposition holds
the even-indexed samples of the previous resolution, untouched by any
filtering.  Decoding with ``level=r`` therefore returns exactly
``image[::2**r, ::2**r]`` -- no interpolation, no overshoot, and no values
that were absent from the original image.  Decoding at full resolution
remains lossless.
"""
from ojph import imwrite, imread, imwrite_to_memory, imread_from_memory
import numpy as np
import pytest


def _mask_image(shape, rng):
    """A mask-like image: a few constant-valued discs on a zero background."""
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
def test_rev13_lossless_full_resolution(shape, tmp_path):
    filename = tmp_path / 'test.j2c'
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, shape, dtype=np.uint8)
    imwrite(filename, data, wavelet='rev13')
    image_read = imread(filename)

    np.testing.assert_array_equal(data, image_read)


@pytest.mark.parametrize('shape', [(256, 256), (255, 193), (1024, 512)])
def test_rev13_levels_are_exact_subsampling(shape, tmp_path):
    filename = tmp_path / 'test.j2c'
    rng = np.random.default_rng(42)
    data = _mask_image(shape, rng)
    num_decompositions = 5
    imwrite(filename, data, wavelet='rev13',
            num_decompositions=num_decompositions)

    for level in range(1, num_decompositions + 1):
        image_read = imread(filename, level=level)
        subsampled = data[:: 2 ** level, :: 2 ** level]
        np.testing.assert_array_equal(subsampled, image_read)
        # no values that do not appear in the original image
        assert set(np.unique(image_read)) <= set(np.unique(data))


def test_rev13_in_memory_roundtrip():
    rng = np.random.default_rng(42)
    data = _mask_image((512, 512), rng)
    buffer = imwrite_to_memory(data, wavelet='rev13')

    np.testing.assert_array_equal(data, imread_from_memory(buffer))
    np.testing.assert_array_equal(
        data[::4, ::4], imread_from_memory(buffer, level=2))


def test_rev13_dtypes(tmp_path):
    filename = tmp_path / 'test.j2c'
    rng = np.random.default_rng(42)
    data = rng.integers(0, 4096, (200, 150)).astype(np.uint16)
    imwrite(filename, data, wavelet='rev13', num_decompositions=3)

    np.testing.assert_array_equal(data, imread(filename))
    np.testing.assert_array_equal(data[::2, ::2], imread(filename, level=1))


def test_rev53_and_irv97_still_selectable(tmp_path):
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, (200, 100), dtype=np.uint8)

    filename = tmp_path / 'test53.j2c'
    imwrite(filename, data, wavelet='rev53')
    np.testing.assert_array_equal(data, imread(filename))

    filename = tmp_path / 'test97.j2c'
    imwrite(filename, data, wavelet='irv97', qstep=0.01)
    image_read = imread(filename)
    assert np.abs(image_read.astype(int) - data.astype(int)).max() <= 8


def test_is_predict_only(tmp_path):
    from ojph._imread import OJPHImageFile

    rng = np.random.default_rng(42)
    data = _mask_image((256, 256), rng)

    filename = tmp_path / 'rev13.j2c'
    imwrite(filename, data, wavelet='rev13')
    f = OJPHImageFile(filename)
    assert f.is_predict_only

    for wavelet, kwargs in (('rev53', {}), ('irv97', dict(qstep=0.01))):
        filename = tmp_path / f'{wavelet}.j2c'
        imwrite(filename, data, wavelet=wavelet, **kwargs)
        f = OJPHImageFile(filename)
        assert not f.is_predict_only

    buffer = imwrite_to_memory(data, wavelet='rev13')
    assert OJPHImageFile.from_memory(buffer).is_predict_only


def test_rev13_wavelet_validation():
    data = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match='Invalid wavelet'):
        imwrite_to_memory(data, wavelet='rev99')
    with pytest.raises(ValueError, match='contradicts'):
        imwrite_to_memory(data, wavelet='rev13', reversible=False)
    with pytest.raises(ValueError, match='contradicts'):
        imwrite_to_memory(data, wavelet='irv97', reversible=True)
