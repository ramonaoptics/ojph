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
    libraries=['openjph'],
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
