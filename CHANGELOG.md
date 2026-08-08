# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Add `wavelet=` to `imwrite` / `imwrite_to_memory`, accepting `'irv97'`,
  `'rev53'` (the Part 1 kernels, previously selected through `reversible=`),
  and the new `'rev13'`: a reversible predict-only kernel (the 5/3 kernel
  with a null update step) signaled with a JPEG 2000 Part 2 ATK marker
  segment. With `'rev13'`, the low-pass subband of every decomposition holds
  the even-indexed samples of the previous resolution untouched, so
  `imread(..., level=r)` returns exactly `image[::2**r, ::2**r]` — no
  interpolation, no overshoot, and no values absent from the original image —
  while full-resolution decoding remains lossless. This is designed for
  label/mask images where in-between values are illegal. Encoding requires an
  OpenJPH build with `param_cod::set_wavelet_kern` (see
  https://github.com/aous72/OpenJPH/issues/261); the produced codestreams
  decode with stock OpenJPH >= 0.30 (including existing `ojph` wheels).
- Add `wavelet='rev12'`: a reversible predict-only kernel whose high-pass
  subbands hold exact previous-sample differences (`H = X(2n+1) - X(2n)`).
  Every resolution level is still an exact subsample and full resolution is
  still lossless, and on mask-like images the one-sided prediction halves
  the nonzero detail coefficients, measuring ~35% smaller codestreams than
  `'rev13'` (and smaller than optimized PNG). The kernel is signaled as a
  JPEG 2000 Part 2 *arbitrary* (ARB) filter with constant boundary
  extension, which stock OpenJPH does not implement — decoding `'rev12'`
  codestreams requires an OpenJPH build with ARB kernel support, unlike
  `'rev13'` which stock decoders already read.
- Add `OJPHImageFile.is_predict_only`: True when the codestream's wavelet
  kernel has no effective update steps, i.e. every resolution level is an
  exact subsample of the full-resolution image. The decision inspects the
  lifting steps signaled in the ATK marker segment rather than the kernel
  index (which is file-local in Part 2 and carries no meaning across
  files), so it is reliable for codestreams produced by other encoders.

## [0.9.1] - 2026-08-04

- Reshape a caller-supplied `out` buffer with `np.reshape(..., copy=False)`
  instead of assigning to `image.shape`, which numpy has deprecated. `copy=False`
  keeps the guarantee we relied on: numpy raises rather than quietly handing back
  a reshaped *copy* that we would decode into instead of the caller's buffer.
  This requires `numpy >= 2.1`, which is now the minimum version.

## [0.9.0] - 2026-07-25

- Support free-threaded (PEP 703) CPython. The extension now declares
  `py::mod_gil_not_used()`, so importing it on a free-threaded interpreter
  (3.13t/3.14t) leaves the GIL disabled. Previously the interpreter re-enabled
  the GIL at import time with a `RuntimeWarning`, which silently undid the
  benefit of running a free-threaded build. Building the extension requires
  `pybind11 >= 2.13` for that API; older pybind11 still compiles, but the
  resulting module re-enables the GIL as before.
- Add `tests/test_free_threading.py`: asserts the GIL stays disabled after
  import, and exercises the encode/decode paths from a barrier-synchronised
  thread pool (shared read-only codestreams, `read_j2c_into`,
  `read_j2c_fd_into`, file reads, and concurrent encoding). These tests run on
  GIL-enabled builds too, where the hot paths are already concurrent thanks to
  `py::gil_scoped_release`.
- Test the free-threaded wheels in CI. `cp314t` was already being *built* by
  cibuildwheel but its test run was skipped; numpy now ships `cp314t` wheels for
  every target we test on, so the skip is gone. `.github/workflows/tests.yml`
  also gained a `free-threaded` job covering Linux and macOS.
- Select the interpreter ABI explicitly in CI, by `python_abi` build string
  (`*_cp314` vs `*_cp314t`, the two builds conda-forge's python 3.14 migration
  produces). conda-forge publishes both and a bare `python=3.14` resolves to
  either depending on the rest of the solve, so the regular matrix jobs could
  silently run on a free-threaded interpreter instead of the one named in the
  matrix.
- Advertise free-threading support with the
  `Programming Language :: Python :: Free Threading :: 3 - Stable` classifier.
- Declare `long_description_content_type='text/markdown'`. The README has always
  been Markdown, but without this PyPI parses it as reStructuredText and rejects
  the upload once it contains anything that is not also valid RST -- such as a
  fenced code block. `twine check --strict` now runs on every pull request, so
  this fails there instead of after a release tag has been pushed.

## [0.7.0] - 2026-07-03

- Add `read_j2c_into(data, out, level, min_val=None, max_val=None)`: a single,
  GIL-free entry point that performs the whole reduced-resolution decode
  (open / read headers / restrict resolution / create / pull) into a
  caller-provided 2D buffer under one `py::gil_scoped_release`. This lets callers
  decode many small images concurrently from a Python thread pool without
  serialising on the GIL (a ~2.3x threaded speedup for viewport-sized reads).
- Add `read_j2c_fd_into(fd, offset, nbytes, out, level, min_val, max_val,
  o_direct)` and `peek_j2c_fd(fd, offset, nbytes, o_direct)`: perform the entire
  reduced-resolution read straight from a file descriptor -- an aligned
  (O_DIRECT-compatible) file read, a TLM-trimmed partial read, and the decode --
  all under one `py::gil_scoped_release`, so a thread pool can run many tile
  reads truly concurrently. A portable aligned allocator keeps the read buffers
  sector-aligned; Windows is supported.
- Build against the latest OpenJPH. OpenJPH PR
  [#312](https://github.com/aous72/OpenJPH/pull/312) ("Removes direct access to
  COC segment marker") added COC-segment overloads to `ojph::param_cod`, which
  made the unqualified member-function pointers used by the bindings ambiguous
  and broke compilation. The affected `param_cod` `.def(...)` bindings are now
  disambiguated with explicit `static_cast` to the COD (no `comp_idx`) overloads.
  This change is backward-compatible and still compiles against older OpenJPH
  where those methods are not overloaded.
- Require OpenJPH >= 0.30.1.
- Ship binary wheels. CI (`.github/workflows/wheels.yml`) uses
  [cibuildwheel](https://cibuildwheel.pypa.io/) to build a static OpenJPH
  (`tools/build_openjph.py`) and statically link it into self-contained wheels
  for CPython 3.12/3.13/3.14 (plus 3.14 free-threading) across Linux
  (x86_64/aarch64), macOS (x86_64/arm64) and Windows (x86_64/ARM64). Building
  against a system/conda OpenJPH shared library (e.g. the conda-forge feedstock)
  is still supported and is the default when no prebuilt static OpenJPH is
  present.

## [0.6.2] - 2026-02-20

- Fix encoding errors associated with datatypes where the order is explicitely defined.

## [0.6.1] - 2026-01-23

- Fix memory leak in `imwrite_to_memory()`. The codestream and memory outfile are now
  properly closed after encoding.



## [0.6.0] - 2026-01-22

- Provide a new method `get_level_shape` to help get the shape after decoding for
  the image at a given resolution level.

## [0.5.1] - 2025-12-29

- Fix writing 3D arrays with a single channel dimension (shape `(H, W, 1)`) as monochrome
  images. The last dimension is now automatically collapsed when `num_components == 1`,
  restoring compatibility with version 0.4.6 behavior.

## [0.5.0] - 2025-12-29

- Optimize image reading and writing for multi-threaded workloads by releasing the
  GIL for entire operations instead of per-line. All tight loops (component and line
  iterations) are now executed in C++ with the GIL released, significantly improving
  performance in multi-threaded scenarios. Single-threaded performance is also improved
  due to reduced Python overhead and better cache locality.
- Remove temporary buffer allocations during image reading by writing directly to
  the output array with clipping and dtype conversion handled in C++.

## [0.4.6] - 2025-12-29

- Fix reading from memory files when the offset parameter is provided.
- Fix the fact that `tlm_marker`, `tileparts_at_resolutions`,
  `tileparts_at_components` arguments were not exposed to
  `imwrite_to_memory`.

## [0.4.5] - 2025-12-28

- Add `tlm_marker` in JPEG2000 codestream by default. A new option `tlm_marker` is
  added to `imwrite` to control this behavior.
- Add options for `tileparts_at_resolutions` and `tileparts_at_components`. By default
  `tileparts_at_resolutions` is set to True for writing files.

## [0.4.4] - 2025-12-15

- Fix bug in writing HWC images with 1 channel component.

## [0.4.3] - 2025-12-15

- Fix bug with integer overflow visible when non-reversible compression is used.

## [0.4.2] - 2025-12-15

- Provide the out parameter `imread_from_memory`.

## [0.4.1] - 2025-12-15

- Provide controls over the progression order.

## [0.4.0] - 2025-12-15

- Provide parameters for irriversible compression.
- Provide an `out` parameter to help output images to pre-allocated arrays

## [0.3.1] - 2025-12-14

- Unify channel order handling.
- Expose level parameters for imread.
- Allow users to specify the number of levels for writing an image.

## [0.3.0] - 2025-12-14

- Update channel order argument so that it works better with jpeg2000.

## [0.2.0] - 2025-12-08

- Use `__buffer__` introduced in Python 3.12 for imporved memory management.

## [0.1.1] - 2025-12-07

- Use new get_used_size api from openjph to tell the used size of `memout_file`.

## [0.1.0] - 2025-08-24

### Added
- Memory-based compression and decompression functionality
  - `imwrite_to_memory()` function for compressing images to memory
  - `imread_from_memory()` function for decompressing images from memory
  - `CompressedData` class for handling compressed data in memory
- Support for multi-component images (RGB, RGBA) with proper color transform handling
- Enhanced channel order support (HWC, CHW) with automatic format detection
- Comprehensive test coverage for memory operations and multi-component images

### Changed
- Bumped minimum Python version requirement from 3.9 to 3.10
- Improved GitHub Actions workflow with proper Git configuration for version generation
- Enhanced error messages for unsupported bit depths and invalid channel orders
- Optimized multi-component image processing with planar mode for better efficiency

### Fixed
- Fixed version generation in GitHub Actions by ensuring full Git history is fetched
- Fixed Git configuration in CI environment to prevent "unknown+geb2aff9" version issues
- Improved handling of different image formats and data types

## [0.0.2] - 2024-10-20

### Added
- Windows compatibility testing and fixes
- Enhanced error handling for imread failures
- Additional test coverage for edge cases

### Changed
- Improved GitHub Actions workflow configuration for Windows testing
- Enhanced imread resilience for various input formats
- Updated Python compatibility declarations for conda-forge

### Fixed
- Fixed Windows-specific build and test failures
- Improved error handling in imread function
- Enhanced workflow stability across different platforms

## [0.0.1] - 2024-10-15

### Added
- Initial implementation of imread and imwrite functions
- Basic JPEG2000 compression and decompression support
- Comprehensive test suite for lossless compression
- GitHub Actions CI/CD workflows for automated testing
- Package configuration and setup for PyPI distribution

### Changed
- Established project structure with proper Python packaging
- Configured build system with pybind11 for C++ bindings
- Set up development environment with proper dependencies

### Fixed
- Initial workflow configuration and testing setup
- Package installation and import issues
