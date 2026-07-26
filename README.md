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
