"""Tests for the GIL-free ``read_j2c_into`` decode entry point.

``read_j2c_into(data, out, level, min_val=None, max_val=None)`` performs the
entire reduced-resolution decode (open / read headers / restrict resolution /
create / pull) under a single ``py::gil_scoped_release`` and writes the pixels
into a caller-provided, pre-allocated 2D array. It returns the ``(height,
width)`` actually decoded.

These tests pin the core contract so downstream readers can build on top of it:
for every level, dtype and size the result must be byte-identical to the
reference :func:`ojph._imread.imread_from_memory` path.
"""
import os

import numpy as np
import pytest

from ojph._imwrite import imwrite_to_memory
from ojph._imread import imread_from_memory, OJPHImageFile
from ojph.ojph_bindings import read_j2c_into, peek_j2c_fd, read_j2c_fd_into

# Codestreams are binary; on Windows the fd must be opened O_BINARY or the CRT
# translates CRLF and stops at the first 0x1A byte.
_O_RDONLY_BINARY = os.O_RDONLY | getattr(os, 'O_BINARY', 0)


def _encode(image, *, num_decompositions=5, reversible=True, qstep=None):
    """Encode a single-component image the way a tiled exporter would."""
    channel_order = 'HW' if image.ndim == 2 else 'HWC'
    data = imwrite_to_memory(
        image,
        channel_order=channel_order,
        num_decompositions=num_decompositions,
        reversible=reversible,
        qstep=qstep,
        progression_order='RLCP',
        tlm_marker=True,
        tileparts_at_resolutions=True,
    )
    return np.frombuffer(bytes(data), dtype=np.uint8).copy()


def _clip_bounds(dtype):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return int(info.min), int(info.max)
    return None, None


def _read_into(data, dtype, level):
    """Allocate an exact-shape output and decode into it, as a caller would."""
    ref_shape = OJPHImageFile.from_memory(
        data, channel_order='HWC'
    ).get_level_shape(level)
    out = np.empty(ref_shape, dtype=dtype)
    lo, hi = _clip_bounds(dtype)
    h, w = read_j2c_into(data, out, level, lo, hi)
    return out, (h, w)


# ---------------------------------------------------------------------------
# Core correctness: identical to imread_from_memory at every level.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.int16, np.int8])
@pytest.mark.parametrize('level', [0, 1, 2, 3, 4, 5])
def test_matches_imread_all_levels(dtype, level):
    lo, hi = _clip_bounds(dtype)
    rng = np.random.default_rng(level + 17)
    image = rng.integers(lo, hi + 1, size=(224, 288)).astype(dtype)

    data = _encode(image, num_decompositions=5)

    reference = imread_from_memory(data, level=level)
    out, (h, w) = _read_into(data, dtype, level)

    assert (h, w) == reference.shape
    assert out.shape == reference.shape
    assert out.dtype == dtype
    assert np.array_equal(out, reference)


@pytest.mark.parametrize('num_decompositions', [1, 2, 3, 6, 8])
def test_different_num_decompositions(num_decompositions):
    rng = np.random.default_rng(num_decompositions)
    image = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    data = _encode(image, num_decompositions=num_decompositions)

    for level in range(num_decompositions + 1):
        reference = imread_from_memory(data, level=level)
        out, shape = _read_into(data, np.uint8, level)
        assert shape == reference.shape
        assert np.array_equal(out, reference), f"level {level}"


@pytest.mark.parametrize('size', [(16, 16), (33, 47), (97, 97), (200, 120)])
def test_odd_dimensions_ceiling_division(size):
    # Wavelet level shapes use ceiling division; odd sizes must still line up
    # exactly with imread_from_memory / get_level_shape.
    rng = np.random.default_rng(sum(size))
    image = rng.integers(0, 256, size=size, dtype=np.uint8)
    data = _encode(image, num_decompositions=3)

    for level in range(4):
        reference = imread_from_memory(data, level=level)
        out, shape = _read_into(data, np.uint8, level)
        assert shape == reference.shape
        assert np.array_equal(out, reference), f"size {size} level {level}"


def test_returns_decoded_height_width():
    image = np.random.default_rng(0).integers(0, 256, (240, 320), dtype=np.uint8)
    data = _encode(image, num_decompositions=4)
    for level in range(5):
        expected = OJPHImageFile.from_memory(
            data, channel_order='HWC'
        ).get_level_shape(level)
        out = np.empty(expected, dtype=np.uint8)
        h, w = read_j2c_into(data, out, level, 0, 255)
        assert (h, w) == expected


# ---------------------------------------------------------------------------
# Lossy / clipping behaviour.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('level', [0, 2, 4])
def test_irreversible_matches_imread(level):
    rng = np.random.default_rng(level)
    image = rng.integers(0, 256, size=(192, 256), dtype=np.uint8)
    data = _encode(image, num_decompositions=4, reversible=False, qstep=0.01)

    reference = imread_from_memory(data, level=level)
    out, shape = _read_into(data, np.uint8, level)
    assert shape == reference.shape
    # imread_from_memory clips to the dtype range with the same bounds we pass,
    # so the two must agree bit-for-bit even though the codec is lossy.
    assert np.array_equal(out, reference)


def test_without_clip_lossless_at_full_resolution():
    # min_val/max_val = None means "do not clip". At level 0 a reversible
    # codestream reconstructs the original exactly and in range, so no-clip must
    # return the original image bit-for-bit.
    #
    # NOTE: at *reduced* resolution levels even a reversible codestream can emit
    # LL-band values outside the dtype range, so clipping is required there --
    # which is exactly why callers always pass min_val/max_val for reduced-res
    # reads. That divergence is covered by test_matches_imread_all_levels.
    image = np.random.default_rng(3).integers(0, 256, (128, 128), dtype=np.uint8)
    data = _encode(image, num_decompositions=3)
    out = np.empty(image.shape, dtype=np.uint8)
    read_j2c_into(data, out, 0, None, None)
    assert np.array_equal(out, image)


# ---------------------------------------------------------------------------
# Robustness: the buffer may reference more memory than the codestream needs
# (a caller may hand in an over-sized, partially-filled read buffer).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('level', [0, 2, 4])
def test_oversized_trailing_garbage_buffer(level):
    image = np.random.default_rng(9).integers(0, 256, (256, 256), dtype=np.uint8)
    data = _encode(image, num_decompositions=5)
    # Pad the codestream with garbage after the terminating EOC marker; the
    # decoder must ignore it (this mimics an aligned over-read buffer).
    padded = np.empty(data.nbytes + 8192, dtype=np.uint8)
    padded[:data.nbytes] = data
    padded[data.nbytes:] = 0xA5
    reference = imread_from_memory(data, level=level)
    out = np.empty(reference.shape, dtype=np.uint8)
    read_j2c_into(padded, out, level, 0, 255)
    assert np.array_equal(out, reference)


def test_repeated_reuse_of_same_buffer():
    # Decoding many codestreams into the same output buffer (as a viewport read
    # would) must not leak state between calls.
    rng = np.random.default_rng(11)
    out = np.empty(OJPHImageFile.from_memory(
        _encode(np.zeros((160, 160), np.uint8)), channel_order='HWC'
    ).get_level_shape(2), dtype=np.uint8)
    for _ in range(5):
        image = rng.integers(0, 256, (160, 160), dtype=np.uint8)
        data = _encode(image, num_decompositions=4)
        reference = imread_from_memory(data, level=2)
        read_j2c_into(data, out, 2, 0, 255)
        assert np.array_equal(out, reference)


# ---------------------------------------------------------------------------
# Error handling: the entry point must fail loudly, never corrupt memory.
# ---------------------------------------------------------------------------
def test_error_out_must_be_2d():
    data = _encode(np.zeros((64, 64), np.uint8))
    with pytest.raises(ValueError, match='2-dimensional'):
        read_j2c_into(data, np.empty((32, 32, 1), np.uint8), 1, 0, 255)


def test_error_out_must_be_contiguous():
    data = _encode(np.zeros((64, 64), np.uint8))
    shape = OJPHImageFile.from_memory(data, channel_order='HWC').get_level_shape(1)
    non_contig = np.empty((shape[0], shape[1] * 2), np.uint8)[:, ::2]
    assert not non_contig.flags['C_CONTIGUOUS']
    with pytest.raises(ValueError, match='C-contiguous'):
        read_j2c_into(data, non_contig, 1, 0, 255)


def test_error_shape_mismatch():
    data = _encode(np.zeros((64, 64), np.uint8))
    with pytest.raises(ValueError, match='does not match'):
        read_j2c_into(data, np.empty((7, 7), np.uint8), 1, 0, 255)


# ---------------------------------------------------------------------------
# read_j2c_fd_into / peek_j2c_fd: the whole read (file I/O + TLM + decode) done
# GIL-free in C++ straight from a file descriptor. Must match imread_from_memory
# for every level, at any byte offset (a codestream is usually embedded in a
# TIFF tile, not at the start of the file), for both the plain and O_DIRECT-style
# aligned read paths.
# ---------------------------------------------------------------------------
def _write_at_offset(tmp_path, data, pad):
    path = tmp_path / f"cs_{pad}.j2c"
    with open(path, 'wb') as f:
        f.write(b'\x00' * pad)
        f.write(data.tobytes())
    return str(path)


@pytest.mark.parametrize('o_direct', [False, True])
@pytest.mark.parametrize('pad', [0, 4096, 1234])
def test_read_j2c_fd_into_matches_imread(tmp_path, pad, o_direct):
    # o_direct=True with pad in {0, 4096} keeps the offset sector-aligned (the
    # only case O_DIRECT applies to); pad=1234 is only exercised with the
    # regular path.
    if o_direct and pad == 1234:
        pytest.skip("O_DIRECT requires a sector-aligned offset")
    rng = np.random.default_rng(pad + int(o_direct))
    image = rng.integers(0, 256, size=(256, 320), dtype=np.uint8)
    data = _encode(image, num_decompositions=5)
    path = _write_at_offset(tmp_path, data, pad)

    fd = os.open(path, _O_RDONLY_BINARY)
    try:
        nd, height, width = peek_j2c_fd(fd, pad, data.nbytes, o_direct)
        assert (height, width) == image.shape
        assert nd == 5
        for level in range(nd + 1):
            reference = imread_from_memory(data, level=level)
            out = np.empty(reference.shape, dtype=np.uint8)
            h, w = read_j2c_fd_into(
                fd, pad, data.nbytes, out, level, 0, 255, o_direct
            )
            assert (h, w) == reference.shape
            assert np.array_equal(out, reference), f"level {level}"
    finally:
        os.close(fd)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.int16])
def test_read_j2c_fd_into_dtypes(tmp_path, dtype):
    lo, hi = _clip_bounds(dtype)
    rng = np.random.default_rng(hash(dtype) & 0xFFFF)
    image = rng.integers(lo, hi + 1, size=(200, 200)).astype(dtype)
    data = _encode(image, num_decompositions=4)
    path = _write_at_offset(tmp_path, data, 4096)

    fd = os.open(path, _O_RDONLY_BINARY)
    try:
        for level in range(5):
            reference = imread_from_memory(data, level=level)
            out = np.empty(reference.shape, dtype=dtype)
            read_j2c_fd_into(fd, 4096, data.nbytes, out, level, lo, hi, False)
            assert np.array_equal(out, reference), f"level {level}"
    finally:
        os.close(fd)


def test_read_j2c_fd_into_no_tlm_falls_back(tmp_path):
    # Encoded without a TLM marker: the C++ TLM parse returns 0 and the function
    # must fall back to reading the whole tile and still decode correctly.
    image = np.random.default_rng(5).integers(0, 256, (192, 192), dtype=np.uint8)
    data = np.frombuffer(bytes(imwrite_to_memory(
        image, channel_order='HW', num_decompositions=4, reversible=True,
        progression_order='RLCP', tlm_marker=False,
    )), dtype=np.uint8).copy()
    path = _write_at_offset(tmp_path, data, 0)

    fd = os.open(path, _O_RDONLY_BINARY)
    try:
        for level in range(5):
            reference = imread_from_memory(data, level=level)
            out = np.empty(reference.shape, dtype=np.uint8)
            read_j2c_fd_into(fd, 0, data.nbytes, out, level, 0, 255, False)
            assert np.array_equal(out, reference), f"level {level}"
    finally:
        os.close(fd)


def test_read_j2c_fd_into_error_contracts(tmp_path):
    data = _encode(np.zeros((64, 64), np.uint8))
    path = _write_at_offset(tmp_path, data, 0)
    fd = os.open(path, _O_RDONLY_BINARY)
    try:
        with pytest.raises(ValueError, match='2-dimensional'):
            read_j2c_fd_into(fd, 0, data.nbytes, np.empty((8, 8, 1), np.uint8),
                             1, 0, 255, False)
        with pytest.raises(ValueError, match='does not match'):
            read_j2c_fd_into(fd, 0, data.nbytes, np.empty((3, 3), np.uint8),
                             1, 0, 255, False)
    finally:
        os.close(fd)
