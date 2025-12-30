# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
