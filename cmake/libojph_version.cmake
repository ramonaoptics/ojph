# Computes LIBOJPH_VERSION, the version string that ojph.__version__ reports
# for this source tree, by following ojph/_version.py (miniver) step by step:
#
#   1. a version written into ojph/_static_version.py (sdists, wheels) wins;
#   2. otherwise `git describe --tags --long --always [--first-parent]`,
#      replaced by $ENV{OJPH_GIT_DESCRIBE} when that is set, formatted as
#      <tag>[.post<distance>][+g<hash>[.dirty]];
#   3. otherwise the refnames that `git archive` expands into
#      ojph/_static_version.py;
#   4. otherwise "unknown".
#
# LIBOJPH_VERSION_MAJOR/MINOR/PATCH are the leading X.Y.Z of it, or 0.0.0
# when it does not start with one.
#
# Keep this in step with ojph/_version.py.

set(_libojph_package_dir "${CMAKE_CURRENT_SOURCE_DIR}/ojph")
set(_libojph_static_file "${_libojph_package_dir}/_static_version.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${_libojph_static_file}")

function(_libojph_static_value name out)
  file(STRINGS "${_libojph_static_file}" _line
       REGEX "^${name} = ")
  if (_line MATCHES "^${name} = [\"']([^\"']*)[\"']")
    set(${out} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  else()
    set(${out} "" PARENT_SCOPE)
  endif()
endfunction()

function(_libojph_pep440 release dev local_parts out)
  set(_version "${release}")
  if (NOT dev STREQUAL "")
    if (release MATCHES "(-dev|\\.dev)$")
      string(APPEND _version "${dev}")
    else()
      string(APPEND _version ".post${dev}")
    endif()
  endif()
  if (NOT local_parts STREQUAL "")
    string(REPLACE ";" "." _local_parts "${local_parts}")
    string(APPEND _version "+${_local_parts}")
  endif()
  set(${out} "${_version}" PARENT_SCOPE)
endfunction()

function(_libojph_version_from_git out)
  set(${out} "" PARENT_SCOPE)
  find_package(Git QUIET)
  if (NOT GIT_FOUND)
    return()
  endif()
  set(_result 1)
  foreach(_options "--first-parent" "")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" describe --tags --long --always ${_options}
      WORKING_DIRECTORY "${_libojph_package_dir}"
      RESULT_VARIABLE _result
      OUTPUT_VARIABLE _describe
      ERROR_QUIET)
    if (_result EQUAL 0)
      break()
    endif()
  endforeach()
  if (NOT _result EQUAL 0)
    return()
  endif()

  # Configure again when a commit, checkout, tag or staged change could
  # change the answer, so the installed version never goes stale.
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --absolute-git-dir
    WORKING_DIRECTORY "${_libojph_package_dir}"
    OUTPUT_VARIABLE _git_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  foreach(_file HEAD logs/HEAD index packed-refs)
    if (EXISTS "${_git_dir}/${_file}")
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                   "${_git_dir}/${_file}")
    endif()
  endforeach()

  # str.strip("v") runs before rstrip("\n"), so trailing v's are only
  # removed from a describe string that has no newline.
  if (DEFINED ENV{OJPH_GIT_DESCRIBE} AND NOT "$ENV{OJPH_GIT_DESCRIBE}" STREQUAL "")
    set(_describe "$ENV{OJPH_GIT_DESCRIBE}")
    string(REGEX REPLACE "^v+" "" _describe "${_describe}")
    string(REGEX REPLACE "v+$" "" _describe "${_describe}")
  else()
    string(REGEX REPLACE "^v+" "" _describe "${_describe}")
  endif()
  string(REGEX REPLACE "\n+$" "" _describe "${_describe}")

  if (_describe MATCHES "^(.*)-([^-]*)-([^-]*)$")
    set(_release "${CMAKE_MATCH_1}")
    set(_dev "${CMAKE_MATCH_2}")
    set(_git "${CMAKE_MATCH_3}")
  else()
    string(REGEX REPLACE "-.*$" "" _first "${_describe}")
    set(_git "g${_first}")
    set(_release "unknown")
    set(_dev "")
  endif()

  set(_local_parts "")
  if (_dev STREQUAL "0")
    set(_dev "")
  else()
    list(APPEND _local_parts "${_git}")
  endif()

  execute_process(
    COMMAND "${GIT_EXECUTABLE}" diff --quiet
    WORKING_DIRECTORY "${_libojph_package_dir}"
    RESULT_VARIABLE _dirty)
  if (_dirty EQUAL 1)
    list(APPEND _local_parts "dirty")
  endif()

  _libojph_pep440("${_release}" "${_dev}" "${_local_parts}" _version)
  set(${out} "${_version}" PARENT_SCOPE)
endfunction()

function(_libojph_version_from_git_archive out)
  set(${out} "" PARENT_SCOPE)
  _libojph_static_value(refnames _refnames)
  _libojph_static_value(git_hash _git_hash)
  if (_refnames MATCHES "^\\$Format" OR _git_hash MATCHES "^\\$Format")
    return()
  endif()
  string(REPLACE "," ";" _refs "${_refnames}")
  set(_tags "")
  foreach(_ref IN LISTS _refs)
    string(STRIP "${_ref}" _ref)
    if (_ref MATCHES "^tag: v(.*)$")
      list(APPEND _tags "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  if (_tags)
    list(SORT _tags)
    list(GET _tags 0 _release)
    set(${out} "${_release}" PARENT_SCOPE)
  else()
    set(${out} "unknown+g${_git_hash}" PARENT_SCOPE)
  endif()
endfunction()

_libojph_static_value(version LIBOJPH_VERSION)
if (LIBOJPH_VERSION STREQUAL "__use_git__")
  _libojph_version_from_git(LIBOJPH_VERSION)
  if (LIBOJPH_VERSION STREQUAL "")
    _libojph_version_from_git_archive(LIBOJPH_VERSION)
  endif()
  if (LIBOJPH_VERSION STREQUAL "")
    set(LIBOJPH_VERSION "unknown")
  endif()
endif()

if (LIBOJPH_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
  set(LIBOJPH_VERSION_MAJOR "${CMAKE_MATCH_1}")
  set(LIBOJPH_VERSION_MINOR "${CMAKE_MATCH_2}")
  set(LIBOJPH_VERSION_PATCH "${CMAKE_MATCH_3}")
else()
  set(LIBOJPH_VERSION_MAJOR 0)
  set(LIBOJPH_VERSION_MINOR 0)
  set(LIBOJPH_VERSION_PATCH 0)
endif()
