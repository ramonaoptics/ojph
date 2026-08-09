// Compiles nanobind's support library into this extension. nanobind is not a
// header-only library: its type/function machinery lives in a set of .cpp
// files that are normally built by its CMake integration. Building through
// setuptools instead, this shim pulls the amalgamated source in from the
// installed nanobind package (setup.py puts nanobind's src/ directory on the
// include path).
#include <nb_combined.cpp>
