import glob
import platform
import sys
import sysconfig
import os
from setuptools import setup, find_packages, Extension
import nanobind

with open('README.md', 'r', encoding='utf-8') as fh:
    readme = fh.read()

def get_version_and_cmdclass(pkg_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(pkg_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.get_cmdclass(pkg_path)


version, cmdclass = get_version_and_cmdclass("ojph")

# nanobind ships as headers plus a small support library that is compiled
# into the extension (see ojph/nanobind_support.cpp). robin_map is nanobind's
# vendored hash-table dependency; the src dir is on the include path so the
# support shim can reach nb_combined.cpp.
_nanobind_root = os.path.dirname(os.path.abspath(nanobind.__file__))
include_dirs = [
    os.path.join(_nanobind_root, 'include'),
    os.path.join(_nanobind_root, 'ext', 'robin_map', 'include'),
    os.path.join(_nanobind_root, 'src'),
]
library_dirs = []
runtime_library_dirs = []
libraries = []
extra_objects = []


def _find_static_openjph(install_dir):
    """Return (include_dir, static_archive_path) for a static OpenJPH install.

    ``install_dir`` is a CMake install prefix produced by
    ``tools/build_openjph.py``. Returns ``None`` if it does not look like one.
    The static archive is linked via ``extra_objects`` rather than ``-lopenjph``
    because OpenJPH names the archive after its version on MSVC
    (``openjph.0.30.lib``), which ``-l`` / ``libraries=`` cannot locate.
    """
    include_dir = os.path.join(install_dir, 'include')
    if not os.path.isdir(os.path.join(include_dir, 'ojph')):
        return None
    if platform.system() == 'Windows':
        patterns = ('ojph*.lib',)
    else:
        patterns = ('libojph*.a',)
    for libsubdir in ('lib', 'lib64'):
        libdir = os.path.join(install_dir, libsubdir)
        for pattern in patterns:
            matches = sorted(glob.glob(os.path.join(libdir, pattern)))
            if matches:
                return include_dir, matches[0]
    return None


# The primary configuration statically links the ojph fork of OpenJPH,
# built from the subprojects/ojph submodule by tools/build_openjph.py
# into ./openjph-install; the extension is then self-contained and fast,
# with no runtime library to locate. Linking a shared libojph from the
# environment remains available as a fallback for development setups.
_install_dir = os.environ.get('OPENJPH_INSTALL_DIR')
if not _install_dir:
    _default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'openjph-install')
    if os.path.isdir(_default):
        _install_dir = _default

_static = _find_static_openjph(_install_dir) if _install_dir else None
if _static is not None:
    ojph_include_dir, ojph_archive = _static
    print(f"setup.py: statically linking OpenJPH from {ojph_archive}")
    include_dirs.append(ojph_include_dir)
    extra_objects.append(ojph_archive)
    # The hwy kernels dispatch at run time through the Google Highway
    # library. Wheel builds drop a static hwy archive beside libojph
    # (tools/build_openjph.py); conda/dev builds link the environment's
    # shared libhwy instead.
    _hwy_archives = []
    for _libsubdir in ('lib', 'lib64'):
        for _pat in ('libhwy*.a', 'hwy.lib'):
            _hwy_archives += sorted(
                glob.glob(os.path.join(_install_dir, _libsubdir, _pat)))
    if _hwy_archives:
        print(f"setup.py: statically linking Highway from {_hwy_archives[0]}")
        extra_objects.append(_hwy_archives[0])
    else:
        # Only link -lhwy when a libhwy is actually present (a build made
        # with OJPH_ALLOW_NO_HWY has no hwy references at all). CONDA_PREFIX
        # covers an activated developer environment; PREFIX is the host
        # prefix under conda-build, which does not set CONDA_PREFIX for
        # build scripts.
        _hwy_dirs = []
        for _env_var in ('CONDA_PREFIX', 'PREFIX'):
            _env_prefix = os.environ.get(_env_var)
            if _env_prefix:
                _hwy_dirs += [os.path.join(_env_prefix, 'lib'),
                              os.path.join(_env_prefix, 'Library', 'lib')]
        _hwy_dirs += ['/usr/local/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu']
        for _d in _hwy_dirs:
            if glob.glob(os.path.join(_d, 'libhwy.*')) or \
               glob.glob(os.path.join(_d, 'hwy.lib')):
                libraries.append('hwy')
                library_dirs.append(_d)
                if platform.system() != 'Windows':
                    runtime_library_dirs.append(_d)
                break
else:
    # Link the shared ojph library (the ojph fork of OpenJPH, which is
    # co-installable with upstream OpenJPH). When building inside a conda
    # environment, point the compiler and the runtime loader at its
    # lib/include explicitly, so the build works with a non-conda compiler
    # as well.
    libraries.append('ojph')
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        include_dirs.append(os.path.join(conda_prefix, 'include'))
        library_dirs.append(os.path.join(conda_prefix, 'lib'))
        runtime_library_dirs.append(os.path.join(conda_prefix, 'lib'))

# Check for windows, add PREFIX/Library to the include dirs for compatibility with conda-forge
# This doesn't really hurt...
if platform.system() == 'Windows':
    prefix = sys.prefix
    # For conda environments
    include_dirs.append(os.path.join(prefix, 'Library', 'include'))
    library_dirs.append(os.path.join(prefix, 'Library', 'lib'))

# nanobind (and OpenJPH's headers) require a modern C++ standard. Some
# compilers -- notably AppleClang -- still default to an ancient standard when
# none is given, so request C++17 explicitly rather than relying on the default.
if platform.system() == 'Windows':
    extra_compile_args = ['/std:c++17']
else:
    # -O3 (over distutils' default -O2) is needed to auto-vectorize the
    # contiguous per-dtype conversion loops in the bindings.
    extra_compile_args = ['-std=c++17', '-O3']
    # The clipping loops need SSE4.1 min/max to vectorize; x86-64-v2
    # (Nehalem 2008 and later) is the same baseline conda-forge uses.
    if platform.machine() in ('x86_64', 'AMD64'):
        extra_compile_args.append('-march=x86-64-v2')

define_macros = [
    # Cheap nanobind-internal assertions only; full checks are for debugging.
    ('NB_COMPACT_ASSERTIONS', None),
]
if platform.system() == 'Windows':
    # The amalgamated nanobind support build includes <windows.h> (via the
    # free-threading code in nb_internals.cpp) ahead of translation units
    # that use std::max, so the min/max macros must be suppressed.
    define_macros.append(('NOMINMAX', None))
if sysconfig.get_config_var('Py_GIL_DISABLED'):
    # Free-threaded CPython: nanobind then declares Py_MOD_GIL_NOT_USED for
    # the module, so the interpreter keeps the GIL disabled at import.
    define_macros.append(('NB_FREE_THREADED', None))

ojph_module = Extension(
    'ojph.ojph_bindings',
    sources=['ojph/ojph_bindings.cpp', 'ojph/nanobind_support.cpp'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    runtime_library_dirs=runtime_library_dirs,
    libraries=libraries,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args,
    define_macros=define_macros
)

setup(
    name='ojph',
    version=version,
    cmdclass=cmdclass,
    description='OpenJPH Bindings for Python and Numpy',
    long_description=readme,
    # README.md is Markdown. Without this, PyPI (and `twine check`) fall back to
    # reStructuredText and reject the upload as soon as the README contains
    # anything that is not also valid RST -- a fenced code block, for instance.
    long_description_content_type='text/markdown',
    url='https://github.com/ramonaoptics/ojph',
    author='Mark Harfouche',
    author_email='mark@ramonaoptics.com',
    license='BSD-3-Clause',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Programming Language :: Python :: Free Threading :: 3 - Stable',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=["tests*"]),
    python_requires='>=3.12',
    install_requires=[
        # np.reshape(..., copy=False) is used in ojph/_imread.py
        'numpy>=2.1',
    ],
    license_files=('LICENSE.txt',),
    ext_modules=[ojph_module],
    include_package_data=True,
    zip_safe=False
)
