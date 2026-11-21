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

## Building against an installed libojph

By default, `pip install .` builds a static copy of the OpenJPH fork in
`subprojects/ojph` (with `tools/build_openjph.py`) and links it into the
extension, which is how the wheels are built. To link a shared libojph that is
already installed instead, for example from a package manager, set
`OJPH_SYSTEM_LIBOJPH=1`. `setup.py` then reads the compile and link flags from
pkg-config's `ojph` module (`ojph.pc`); point `PKG_CONFIG_PATH` at its
directory if it is not on the default search path:

```sh
OJPH_SYSTEM_LIBOJPH=1 PKG_CONFIG_PATH=/path/to/prefix/lib/pkgconfig pip install .
```
