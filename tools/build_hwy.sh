#!/usr/bin/env bash
# Build Google Highway as a static, position independent library and install
# it into a prefix, so that libojph can link it into its own shared library
# and run the x86 SIMD kernels without a run time dependency on libhwy.
#
# Usage: tools/build_hwy.sh <version> <install-prefix> [<work-dir>]
set -euo pipefail

version="$1"
prefix="$2"
work="${3:-build/hwy}"

mkdir -p "$work"
if [ ! -d "$work/highway-$version" ]; then
  curl --fail --silent --show-error --location --retry 5 \
    --output "$work/highway-$version.tar.gz" \
    "https://github.com/google/highway/archive/refs/tags/$version.tar.gz"
  # Extract from inside the directory: GNU tar reads "D:/..." as a remote
  # host.
  (cd "$work" && tar -xzf "highway-$version.tar.gz")
fi

cmake -S "$work/highway-$version" -B "$work/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_TESTING=OFF \
  -DHWY_ENABLE_CONTRIB=OFF \
  -DHWY_ENABLE_EXAMPLES=OFF \
  -DHWY_ENABLE_TESTS=OFF
cmake --build "$work/build" --config Release --parallel
cmake --install "$work/build" --config Release
