from setuptools import setup, Extension
import pybind11

# Include the pybind11 include directory
pybind11_include = pybind11.get_include()

# Define the extension module
ojph_module = Extension(
    'ojph_bindings',
    sources=['ojph_bindings.cpp'],
    include_dirs=[pybind11_include],
    libraries=['openjph'],
    extra_compile_args=[]
)

# Setup
setup(
    name='ojph_bindings',
    version='0.1',
    ext_modules=[ojph_module],
    include_package_data=True,
    zip_safe=False
)

