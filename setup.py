import glob
import platform
import sys
import os
from setuptools import setup, find_packages, Extension
# Hmm consider nanobind
import pybind11

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

# Include the pybind11 include directory
include_dirs = [pybind11.get_include()]
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


# The shared ojph library is the primary use case: link -lojph from the
# active environment (e.g. a conda env where the ojph fork of OpenJPH is
# installed next to upstream libopenjph). A static build is only used
# when OPENJPH_INSTALL_DIR is set explicitly, which the wheel-building CI
# does via tools/build_openjph.py.
_install_dir = os.environ.get('OPENJPH_INSTALL_DIR')

_static = _find_static_openjph(_install_dir) if _install_dir else None
if _static is not None:
    ojph_include_dir, ojph_archive = _static
    print(f"setup.py: statically linking OpenJPH from {ojph_archive}")
    include_dirs.append(ojph_include_dir)
    extra_objects.append(ojph_archive)
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

# pybind11 (and OpenJPH's headers) require a modern C++ standard. Some
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

ojph_module = Extension(
    'ojph.ojph_bindings',
    sources=['ojph/ojph_bindings.cpp'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    runtime_library_dirs=runtime_library_dirs,
    libraries=libraries,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args
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
