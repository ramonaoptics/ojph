OpenJPH bindings

## Requirements

These bindings link against [OpenJPH](https://github.com/aous72/OpenJPH) and
require a version containing PR
[#312](https://github.com/aous72/OpenJPH/pull/312) ("Removes direct access to
COC segment marker"). That change was merged after the 0.30.1 release and, at
the time of writing, is only available on OpenJPH `main` (unreleased). Building
against OpenJPH 0.30.1 or earlier is not supported.

## Free-threaded Python

`ojph` supports free-threaded (PEP 703) CPython 3.13t/3.14t: the extension
declares that it does not need the GIL, so importing it leaves the GIL
disabled. Wheels are published for `cp314t`, and the test suite runs on a
free-threaded interpreter in CI.

The decode and encode hot paths already release the GIL, so a thread pool
scales across cores on a free-threaded interpreter:

```python
from concurrent.futures import ThreadPoolExecutor
from ojph.ojph_bindings import read_j2c_into

with ThreadPoolExecutor() as pool:
    pool.map(lambda t: read_j2c_into(t.data, t.out, t.level), tiles)
```

As with a file object, a single `Codestream`, infile or outfile object must not
be shared between threads without external synchronisation. Give each thread its
own; a compressed buffer that is only *read* can safely be shared.

## libojph: the C++ library

The repository also builds **libojph**, a C++ JPEG 2000 / HTJ2K codec from the
OpenJPH fork in `subprojects/ojph`. Besides the standard 5/3 and 9/7 kernels,
it encodes and decodes the reversible predict-only kernels `rev13` and `rev12`
(JPEG 2000 Part 2, signalled with ATK markers). With these kernels, decoding at
`r` skipped resolutions returns exactly `image[::2**r, ::2**r]`.

### Building and installing

```sh
git clone --recurse-submodules https://github.com/ramonaoptics/ojph.git
cmake -S ojph -B build -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build build --config Release
cmake --install build --config Release
```

`--config Release` only matters for multi-config generators such as Visual
Studio; single-config generators default to a Release build.

Options:

- `-DBUILD_SHARED_LIBS=OFF` builds a static `libojph.a` (`ojph.lib` on
  Windows) instead of the shared library.
- On x86-64, the library uses Google Highway SIMD kernels when it finds a
  recent Highway (>= 1.2) with its library. Point it at one with
  `-DOJPH_HWY_INCLUDE_DIR=<dir> -DOJPH_HWY_LIBRARY=<file>` and add
  `-DOJPH_REQUIRE_HWY=ON` to fail instead of falling back to the generic
  kernels. `tools/build_hwy.sh <version> <prefix>` builds a static,
  position independent Highway that links into a self-contained shared
  libojph. `-DOJPH_ENABLE_HWY=OFF` disables Highway.
- `-DOJPH_BUILD_EXECUTABLES=ON` also builds `ojph_compress` and `ojph_expand`.
  They are off by default because upstream OpenJPH installs tools with the
  same names.

The install layout:

```
<prefix>/include/ojph/ojph_*.h               public headers
<prefix>/include/ojph/libojph_version.h      LIBOJPH_VERSION
<prefix>/lib/libojph.so.X.Y                  Linux (libojph.X.Y.dylib on macOS)
<prefix>/bin/ojph.dll, <prefix>/lib/ojph.lib Windows
<prefix>/lib/cmake/ojph/ojph-config.cmake    CMake package
<prefix>/lib/pkgconfig/ojph.pc               pkg-config file
```

None of these names overlap with upstream OpenJPH's (`libopenjph`,
`include/openjph`, `openjph-config.cmake`, `openjph.pc`), so you can install
both into the same prefix.

### Using it

Include the headers with the `ojph/` prefix:

```cpp
#include <ojph/ojph_codestream.h>
#include <ojph/ojph_file.h>
#include <ojph/ojph_mem.h>
#include <ojph/ojph_params.h>
```

With CMake:

```cmake
find_package(ojph CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE ojph::ojph)
```

Configure with `-DCMAKE_PREFIX_PATH=/path/to/prefix` if the prefix is not a
default search path.

With pkg-config:

```sh
export PKG_CONFIG_PATH=/path/to/prefix/lib/pkgconfig
c++ -std=c++14 my_program.cpp $(pkg-config --cflags --libs ojph)
```

Select a kernel with `param_cod::set_wavelet_kern` after
`set_reversible(true)`: `OJPH_WAVELET_REV53`, `OJPH_WAVELET_REV13` or
`OJPH_WAVELET_REV12`. To decode at a reduced resolution, call
`codestream::restrict_input_resolution(r, r)` between `read_headers` and
`create`. `tests/consumer/roundtrip.cpp` encodes and decodes an image with each
of these kernels, and CI builds it against the installed library on Linux,
macOS and Windows.

### Versions

libojph and the `ojph` Python package share one version: the one
`ojph.__version__` reports for the same source tree. CMake works it out the
way miniver does (`ojph/_version.py`): from `git describe --tags` in a git
checkout, honouring `OJPH_GIT_DESCRIBE`, or from `ojph/_static_version.py` in
an sdist or a `git archive`. A tagged release is `X.Y.Z`; a commit after it is
for example `0.10.2.post5+g47ee1aa`.

- `ojph.pc` (`pkg-config --modversion ojph`) and `LIBOJPH_VERSION` in
  `<ojph/libojph_version.h>` carry the full string.
- The CMake package version (`ojph_VERSION`), `LIBOJPH_VERSION_MAJOR`,
  `_MINOR` and `_PATCH` are its leading `X.Y.Z` (`0.0.0` if there is none).
- The shared library is `libojph.so.X.Y` / `libojph.X.Y.dylib`; a new
  minor release may break the ABI.

`OPENJPH_VERSION_MAJOR`, `_MINOR` and `_PATCH` in `<ojph/ojph_version.h>`
are a different number: the version of the OpenJPH codec fork that libojph is
built from.

The release tags that publish the wheels (`X.Y.Z`, `X.Y.ZrcN`) also build the
libojph binary archives, named `libojph-<version>-<platform>`, with build
provenance attestations.
