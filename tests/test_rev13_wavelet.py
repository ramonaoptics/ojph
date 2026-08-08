"""Tests for the rev13 (reversible predict-only, 1/3) wavelet.

With ``wavelet='rev13'`` the low-pass subband of every decomposition holds
the even-indexed samples of the previous resolution, untouched by any
filtering.  Decoding with ``level=r`` therefore returns exactly
``image[::2**r, ::2**r]`` -- no interpolation, no overshoot, and no values
that were absent from the original image.  Decoding at full resolution
remains lossless.
"""
from ojph import imwrite, imread, imwrite_to_memory, imread_from_memory
from ojph._imread import OJPHImageFile
from ojph._rev13 import _iter_main_header_segments, declare_rev13_wavelet
from ojph.ojph_bindings import (
    Codestream, MemOutfile, Point, REV13_WAVELET_INDEX,
    rev13_atk_marker_segment,
)
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


@pytest.mark.parametrize('shape', [(1, 64), (64, 1), (1, 1), (3, 7)])
def test_rev13_degenerate_shapes(shape, tmp_path):
    filename = tmp_path / 'test.j2c'
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, shape, dtype=np.uint8)
    imwrite(filename, data, wavelet='rev13', num_decompositions=2)

    np.testing.assert_array_equal(data, imread(filename))
    np.testing.assert_array_equal(data[::2, ::2], imread(filename, level=1))


def test_rev13_multiple_components(tmp_path):
    filename = tmp_path / 'test.j2c'
    rng = np.random.default_rng(7)
    data = np.stack([_mask_image((128, 128), rng) for _ in range(3)], axis=-1)
    imwrite(filename, data, wavelet='rev13', num_decompositions=3)

    np.testing.assert_array_equal(data, imread(filename))
    np.testing.assert_array_equal(data[::4, ::4], imread(filename, level=2))


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


def test_rev13_wavelet_validation():
    data = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match='Invalid wavelet'):
        imwrite_to_memory(data, wavelet='rev99')
    with pytest.raises(ValueError, match='contradicts'):
        imwrite_to_memory(data, wavelet='rev13', reversible=False)
    with pytest.raises(ValueError, match='contradicts'):
        imwrite_to_memory(data, wavelet='irv97', reversible=True)


def test_rev53_is_unchanged_by_the_rev13_support(tmp_path):
    """Adding wavelet= must not have changed the default codestream."""
    rng = np.random.default_rng(3)
    data = _mask_image((256, 256), rng)

    default = bytes(imwrite_to_memory(data))
    explicit = bytes(imwrite_to_memory(data, wavelet='rev53'))
    assert default == explicit


def test_rev13_is_not_rev53(tmp_path):
    """rev13 must actually run a different transform, not just relabel one."""
    rng = np.random.default_rng(3)
    data = _mask_image((256, 256), rng)

    rev53 = bytes(imwrite_to_memory(data, wavelet='rev53'))
    rev13 = bytes(imwrite_to_memory(data, wavelet='rev13'))
    assert rev13 != rev53
    # The 5/3 update step smears mask edges into the low-pass subband, so it
    # is the more expensive of the two on this kind of image.
    assert len(rev13) < len(rev53)

    # ... and the levels of a rev53 stream are *not* exact subsamples, which is
    # the whole reason rev13 exists.
    level2 = imread_from_memory(np.frombuffer(rev53, dtype=np.uint8), level=2)
    assert not np.array_equal(level2, data[::4, ::4])


def _main_header_markers(data):
    return [marker for marker, _, _ in _iter_main_header_segments(data)]


def test_rev13_main_header_markers():
    """The codestream must declare the kernel it was actually encoded with."""
    rng = np.random.default_rng(11)
    data = _mask_image((256, 256), rng)
    stream = bytes(imwrite_to_memory(data, wavelet='rev13'))

    markers = _main_header_markers(stream)
    # ATK must be present, and immediately before COD, which is where a patched
    # OpenJPH writes it.
    assert 0xFF79 in markers
    assert markers.index(0xFF79) + 1 == markers.index(0xFF52)

    segments = {m: (off, length)
                for m, off, length in _iter_main_header_segments(stream)}

    atk_offset, atk_length = segments[0xFF79]
    assert stream[atk_offset:atk_offset + atk_length] == \
        rev13_atk_marker_segment()

    # SIZ-Rsiz signals Part 2 extensions, and a Part 2 wavelet kernel.
    siz_offset, _ = segments[0xFF51]
    rsiz = int.from_bytes(stream[siz_offset + 4:siz_offset + 6], 'big')
    assert rsiz & 0x8000
    assert rsiz & 0x0020

    # COD-SPcod.wavelet_trans points at the ATK index.
    cod_offset, _ = segments[0xFF52]
    assert stream[cod_offset + 13] == REV13_WAVELET_INDEX

    # A rev53 stream keeps the Part 1 signalling.
    plain = bytes(imwrite_to_memory(data, wavelet='rev53'))
    assert 0xFF79 not in _main_header_markers(plain)
    siz_offset, _ = {m: (o, l) for m, o, l in
                     _iter_main_header_segments(plain)}[0xFF51]
    rsiz = int.from_bytes(plain[siz_offset + 4:siz_offset + 6], 'big')
    assert not rsiz & 0x8000


def test_rev13_file_and_memory_agree(tmp_path):
    rng = np.random.default_rng(5)
    data = _mask_image((128, 192), rng)

    filename = tmp_path / 'test.j2c'
    imwrite(filename, data, wavelet='rev13', num_decompositions=4)
    from_memory = bytes(imwrite_to_memory(
        data, wavelet='rev13', num_decompositions=4))

    assert filename.read_bytes() == from_memory


@pytest.mark.parametrize(
    'kwargs, expected', [
        (dict(wavelet='rev13'), True),
        (dict(wavelet='rev53'), False),
        (dict(), False),
        (dict(wavelet='irv97', qstep=0.01), False),
    ]
)
def test_levels_are_exact_subsampling(kwargs, expected):
    rng = np.random.default_rng(17)
    data = _mask_image((128, 128), rng)
    stream = imwrite_to_memory(data, **kwargs)

    handle = OJPHImageFile.from_memory(stream)
    assert handle.levels_are_exact_subsampling is expected
    assert handle.reversible is (kwargs.get('wavelet') != 'irv97')

    # ... and the property has to be telling the truth.
    if expected:
        np.testing.assert_array_equal(
            imread_from_memory(stream, level=2), data[::4, ::4])


def test_exact_subsampling_is_decided_by_lifting_steps_not_kernel_index():
    """A Part 2 kernel index is file-local and means nothing on its own.

    OpenJPH's own test corpus contains a codestream that uses ATK index 2 for
    an ordinary 5/3 kernel, so anything keying off the index would call it
    exact-subsampling. This builds the same trap: a stream whose COD names
    kernel 2, whose ATK segment has index 2, but whose lifting steps are 5/3's.
    """
    rng = np.random.default_rng(19)
    data = _mask_image((128, 128), rng)
    stream = bytearray(imwrite_to_memory(data, wavelet='rev13'))

    segments = {m: (off, length)
                for m, off, length in _iter_main_header_segments(stream)}
    atk_offset, atk_length = segments[0xFF79]
    assert stream[atk_offset:atk_offset + atk_length] == \
        rev13_atk_marker_segment()

    # Restore the 5/3 update step (Eatk=2, Batk=2, LCatk=1, Aatk=1) in place of
    # the nulled one, leaving the ATK index and every length untouched.
    step0 = atk_offset + 7  # past ATK, Latk, Satk, Natk
    stream[step0:step0 + 5] = bytes([0x02, 0x00, 0x02, 0x01, 0x01])
    assert len(stream) == len(imwrite_to_memory(data, wavelet='rev13'))

    handle = OJPHImageFile.from_memory(np.frombuffer(bytes(stream), np.uint8))
    assert handle.reversible is True
    assert handle.levels_are_exact_subsampling is False


def test_install_rev13_wavelet_before_write_headers_raises():
    """The binding must refuse to touch a kernel that is not linked yet."""
    codestream = Codestream()
    with pytest.raises(RuntimeError, match='write_headers'):
        codestream.install_rev13_wavelet()


def test_install_rev13_wavelet_is_visible_mid_encode():
    """A codestream reports the kernel it will actually transform with."""
    mem_outfile = MemOutfile()
    mem_outfile.open(65536, False)
    codestream = Codestream()

    data = np.zeros((64, 64), dtype=np.uint8)
    siz = codestream.access_siz()
    siz.set_image_extent(Point(64, 64))
    siz.set_num_components(1)
    siz.set_component(0, Point(1, 1), 8, False)
    cod = codestream.access_cod()
    cod.set_reversible(True)
    cod.set_color_transform(False)

    assert cod.is_predict_only() is False   # nothing linked yet
    codestream.write_headers(mem_outfile, None, 0)
    assert cod.is_predict_only() is False   # linked, still 5/3
    codestream.install_rev13_wavelet()
    assert cod.is_predict_only() is True

    codestream.push_all_components(data, 1, 'HW')
    codestream.flush()
    codestream.close()
    mem_outfile.close()


def test_declare_rev13_wavelet_rejects_a_non_rev53_stream():
    rng = np.random.default_rng(9)
    data = _mask_image((64, 64), rng)

    irreversible = bytes(imwrite_to_memory(data, wavelet='irv97', qstep=0.01))
    with pytest.raises(ValueError, match='reversible 5/3 kernel'):
        declare_rev13_wavelet(irreversible)

    with pytest.raises(ValueError, match='SOC'):
        declare_rev13_wavelet(b'not a codestream')
