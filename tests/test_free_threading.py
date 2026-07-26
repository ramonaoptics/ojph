"""Free-threading (PEP 703) compatibility tests.

Two distinct things are checked here.

1. :func:`test_extension_declares_gil_not_used` is the canary. A C extension
   that does not declare ``Py_mod_gil = Py_MOD_GIL_NOT_USED`` makes a
   free-threaded interpreter turn the GIL back *on* when the module is
   imported, which silently undoes the whole point of a ``cp314t`` build. The
   binding declares it via ``py::mod_gil_not_used()``; this test fails if that
   declaration is ever dropped (or if the extension is built against a
   pybind11 older than 2.13, which does not have the API).

2. The remaining tests exercise the encode/decode paths from many threads at
   once. On a free-threaded build those threads genuinely run in parallel
   through the C++ code, so a data race in the bindings or in OpenJPH shows up
   as corrupted pixels, an exception, or a crash. They are useful on a
   GIL-enabled build too: every hot path already runs under
   ``py::gil_scoped_release``, so the decode itself is concurrent there as well.

Each thread owns its own Codestream/infile objects -- that is the supported
contract. Sharing one object across threads is not, and is not tested.
"""
import os
import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from ojph import imread, imwrite_to_memory, imread_from_memory
from ojph.ojph_bindings import read_j2c_into, peek_j2c_fd, read_j2c_fd_into

_O_RDONLY_BINARY = os.O_RDONLY | getattr(os, 'O_BINARY', 0)

FREE_THREADED = bool(sysconfig.get_config_var('Py_GIL_DISABLED'))

# Enough threads to oversubscribe a small CI runner, and enough repetitions per
# thread that an unlucky interleaving has many chances to happen.
NUM_THREADS = 8
ITERATIONS = 12


def _encode(image, **kwargs):
    kwargs.setdefault('num_decompositions', 5)
    kwargs.setdefault('progression_order', 'RLCP')
    kwargs.setdefault('tlm_marker', True)
    channel_order = 'HW' if image.ndim == 2 else 'HWC'
    data = imwrite_to_memory(image, channel_order=channel_order, **kwargs)
    return np.frombuffer(bytes(data), dtype=np.uint8).copy()


def _run_concurrently(func, num_threads=NUM_THREADS):
    """Run ``func(i)`` on ``num_threads`` threads released from a barrier.

    The barrier makes every worker enter the C++ code at roughly the same
    moment, which is what maximises the chance of catching a race. Exceptions
    propagate out of ``.result()``.
    """
    barrier = threading.Barrier(num_threads)

    def worker(i):
        barrier.wait()
        return func(i)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        return [f.result() for f in
                [executor.submit(worker, i) for i in range(num_threads)]]


# ---------------------------------------------------------------------------
# The canary: does the extension keep the GIL disabled?
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not FREE_THREADED, reason='requires a free-threaded (Py_GIL_DISABLED) build'
)
def test_extension_declares_gil_not_used():
    # PYTHON_GIL=1 / -Xgil=1 legitimately forces the GIL back on; that is the
    # user asking for it, not a missing declaration.
    if sys._xoptions.get('gil') == '1' or os.environ.get('PYTHON_GIL') == '1':
        pytest.skip('the GIL was explicitly re-enabled by the caller')

    import ojph.ojph_bindings  # noqa: F401 -- importing is the point

    assert not sys._is_gil_enabled(), (
        'importing ojph.ojph_bindings re-enabled the GIL: the extension is '
        'missing py::mod_gil_not_used(), or was built against pybind11 < 2.13'
    )


@pytest.mark.skipif(
    FREE_THREADED, reason='only meaningful on a GIL-enabled build'
)
def test_gil_enabled_build_keeps_the_gil():
    """The mirror image of the canary, so the pair covers both build flavours.

    ``sys._is_gil_enabled`` exists on every 3.13+ build, not just free-threaded
    ones; on a normal build it must report the GIL as enabled. This catches a
    test that accidentally passes because it inspected the wrong interpreter.
    """
    if not hasattr(sys, '_is_gil_enabled'):
        pytest.skip('sys._is_gil_enabled() requires Python 3.13+')
    assert sys._is_gil_enabled()


# ---------------------------------------------------------------------------
# Concurrency: independent work in each thread must not interfere.
# ---------------------------------------------------------------------------
def test_concurrent_roundtrip_distinct_images():
    """Every thread encodes and decodes its own image, and must get it back."""
    rng = np.random.default_rng(0)
    images = [
        rng.integers(0, 256, size=(96, 128), dtype=np.uint8)
        for _ in range(NUM_THREADS)
    ]

    def work(i):
        image = images[i]
        for _ in range(ITERATIONS):
            decoded = imread_from_memory(_encode(image))
            # Compare inside the thread so a mismatch is attributed to it.
            assert np.array_equal(decoded, image)
        return True

    assert all(_run_concurrently(work))


def test_concurrent_decode_of_shared_codestream():
    """Many threads decoding one shared, immutable compressed buffer.

    The input array is read-only shared state, which is the common pattern for
    a tile cache. Each thread still builds its own Codestream.
    """
    rng = np.random.default_rng(1)
    image = rng.integers(0, 65536, size=(160, 192), dtype=np.uint16)
    data = _encode(image)

    def work(i):
        for level in range(ITERATIONS):
            decoded = imread_from_memory(data, level=level % 6)
            if level % 6 == 0:
                assert np.array_equal(decoded, image)
        return True

    assert all(_run_concurrently(work))


def test_concurrent_read_j2c_into():
    """The GIL-free decode entry point, hammered from every thread.

    ``read_j2c_into`` holds the GIL released for the whole decode, so on a
    free-threaded build these calls overlap almost completely.
    """
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(224, 288), dtype=np.uint8)
    data = _encode(image)
    reference = {level: imread_from_memory(data, level=level)
                 for level in range(6)}

    def work(i):
        for n in range(ITERATIONS):
            level = (i + n) % 6
            expected = reference[level]
            out = np.empty(expected.shape, dtype=np.uint8)
            h, w = read_j2c_into(data, out, level, 0, 255)
            assert (h, w) == expected.shape
            assert np.array_equal(out, expected)
        return True

    assert all(_run_concurrently(work))


def test_concurrent_read_j2c_fd_into(tmp_path):
    """The fd-based read path: concurrent pread + decode on separate fds."""
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(224, 288), dtype=np.uint8)
    data = _encode(image)

    filename = tmp_path / 'concurrent.j2c'
    filename.write_bytes(data.tobytes())
    nbytes = filename.stat().st_size
    reference = {level: imread_from_memory(data, level=level)
                 for level in range(6)}

    def work(i):
        # A fresh fd per thread: an fd carries a shared file offset, so sharing
        # one across threads would be a bug in the caller, not in the binding.
        fd = os.open(filename, _O_RDONLY_BINARY)
        try:
            assert peek_j2c_fd(fd, 0, nbytes)[1:] == reference[0].shape
            for n in range(ITERATIONS):
                level = (i + n) % 6
                expected = reference[level]
                out = np.empty(expected.shape, dtype=np.uint8)
                h, w = read_j2c_fd_into(fd, 0, nbytes, out, level, 0, 255)
                assert (h, w) == expected.shape
                assert np.array_equal(out, expected)
        finally:
            os.close(fd)
        return True

    assert all(_run_concurrently(work))


def test_concurrent_imread_from_file(tmp_path):
    """Whole-file reads, including opening the file, from every thread."""
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, size=(128, 160), dtype=np.uint8)
    filename = tmp_path / 'shared.j2c'
    filename.write_bytes(_encode(image).tobytes())

    def work(i):
        for _ in range(ITERATIONS):
            assert np.array_equal(imread(filename), image)
        return True

    assert all(_run_concurrently(work))


def test_concurrent_encode_only(tmp_path):
    """Encoding is the other half of the library, and has its own tables.

    OpenJPH builds the block-encoder VLC tables lazily on first use; the
    barrier here is what makes several threads reach that initialisation at
    once.
    """
    rng = np.random.default_rng(5)
    images = [
        rng.integers(0, 256, size=(64 + 8 * i, 96), dtype=np.uint8)
        for i in range(NUM_THREADS)
    ]

    def work(i):
        return [len(_encode(images[i])) for _ in range(ITERATIONS)]

    results = _run_concurrently(work)
    for i, sizes in enumerate(results):
        # Encoding is deterministic: every repetition of the same image must
        # produce the same number of bytes. A torn lookup table would not.
        assert len(set(sizes)) == 1, f'thread {i} produced varying output sizes'
        reference = imwrite_to_memory(images[i], channel_order='HW',
                                      num_decompositions=5,
                                      progression_order='RLCP')
        assert len(bytes(reference)) == sizes[0]


def test_concurrent_mixed_read_and_write():
    """Readers and writers running at the same time, sharing nothing."""
    rng = np.random.default_rng(6)
    image = rng.integers(0, 256, size=(112, 144), dtype=np.uint8)
    data = _encode(image)

    def work(i):
        for _ in range(ITERATIONS):
            if i % 2:
                assert np.array_equal(imread_from_memory(data), image)
            else:
                assert len(_encode(image)) == len(data)
        return True

    assert all(_run_concurrently(work))
