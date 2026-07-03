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
    if not os.path.isdir(os.path.join(include_dir, 'openjph')):
        return None
    if platform.system() == 'Windows':
        patterns = ('openjph*.lib',)
    else:
        patterns = ('libopenjph*.a',)
    for libsubdir in ('lib', 'lib64'):
        libdir = os.path.join(install_dir, libsubdir)
        for pattern in patterns:
            matches = sorted(glob.glob(os.path.join(libdir, pattern)))
            if matches:
                return include_dir, matches[0]
    return None


# When a static OpenJPH has been prebuilt (e.g. by CI via
# tools/build_openjph.py), link it directly so the wheel is self-contained.
# Otherwise fall back to linking a system/conda ``openjph`` shared library,
# which is how the editable dev/test builds work.
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
else:
    # Link a system/conda OpenJPH shared library (>= 0.30.1). This is the path
    # used by editable dev/test builds and by the conda-forge feedstock.
    libraries.append('openjph')

# Check for windows, add PREFIX/Library to the include dirs for compatibility with conda-forge
# This doesn't really hurt...
if platform.system() == 'Windows':
    prefix = sys.prefix
    # For conda environments
    include_dirs.append(os.path.join(prefix, 'Library', 'include'))
    library_dirs.append(os.path.join(prefix, 'Library', 'lib'))

ojph_module = Extension(
    'ojph.ojph_bindings',
    sources=['ojph/ojph_bindings.cpp'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_objects=extra_objects,
    extra_compile_args=[]
)

setup(
    name='ojph',
    version=version,
    cmdclass=cmdclass,
    description='OpenJPH Bindings for Python and Numpy',
    long_description=readme,
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
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=["tests*"]),
    python_requires='>=3.12',
    install_requires=[
        'numpy>=1.24.0',
    ],
    license_files=('LICENSE.txt',),
    ext_modules=[ojph_module],
    include_package_data=True,
    zip_safe=False
)
