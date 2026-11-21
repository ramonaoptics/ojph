cmake_minimum_required(VERSION 3.15)

project(libojph LANGUAGES C CXX)

# libojph: the OpenJPH fork vendored in subprojects/ojph, built as an
# installable C++ library.  The fork's own CMake project builds the library
# target ``ojph``, installs its headers under <prefix>/include/ojph and
# exports the CMake package ``ojph`` (imported target ``ojph::ojph``) plus
# ``ojph.pc``.  Every installed name differs from upstream OpenJPH's
# (libopenjph, include/openjph, openjph-config.cmake, openjph.pc), so both
# can live in one prefix.  This file chooses the defaults and
# the version.

# libojph carries the version of the ojph Python package built from the same
# tree; see cmake/libojph_version.cmake.
include(cmake/libojph_version.cmake)
set(OJPH_PROJECT_VERSION
    "${LIBOJPH_VERSION_MAJOR}.${LIBOJPH_VERSION_MINOR}.${LIBOJPH_VERSION_PATCH}")
set(OJPH_PKG_CONFIG_VERSION "${LIBOJPH_VERSION}")
message(STATUS "libojph version ${LIBOJPH_VERSION}")

# The fork defaults the build type to Release inside its own directory
# scope; set it here too, otherwise the installed ojph-targets-<config>.cmake
# is written for a different configuration than the one installed.
get_property(_libojph_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (NOT _libojph_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_SHARED_LIBS "Build libojph as a shared library" ON)

# The fork's command line tools are named ojph_compress and ojph_expand,
# the same as upstream OpenJPH's, so they are off by default to keep the
# install free of clashes.
set(OJPH_BUILD_EXECUTABLES OFF CACHE BOOL
    "Build the ojph_compress and ojph_expand command line tools")
set(OJPH_ENABLE_TIFF_SUPPORT OFF CACHE BOOL
    "Enable TIFF input and output in the command line tools")
set(OJPH_BUILD_TESTS OFF CACHE BOOL "Build the fork's own unit tests")

add_subdirectory(subprojects/ojph)

configure_file(cmake/libojph_version.h.in
               "${CMAKE_CURRENT_BINARY_DIR}/libojph_version.h" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libojph_version.h"
        DESTINATION include/ojph)

if (WIN32)
  # The fork names its Windows library ojph.<major>.<minor>; use the plain
  # name so that ``-lojph`` in ojph.pc matches ojph.lib.
  set_target_properties(ojph PROPERTIES OUTPUT_NAME ojph)
endif()
