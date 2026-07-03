#!/usr/bin/env python3
"""Build a *static* OpenJPH library for bundling into the ``ojph`` wheel.

The Python bindings require OpenJPH 0.30.1 or newer. Rather than depend on a
system/conda OpenJPH at runtime, the wheels statically link a build produced by
this script so the resulting wheel is self-contained.

This is invoked from ``CIBW_BEFORE_ALL`` (see ``pyproject.toml``). It runs once
per platform, before any wheel is built, and installs into ``<project>/openjph-install``
by default -- a location inside the (cibuildwheel-mounted) project tree so it
survives into the per-Python build steps. ``setup.py`` then discovers that
directory and links ``libopenjph`` statically.

Usage::

    python tools/build_openjph.py [--prefix DIR] [--source-dir DIR] [--jobs N]

Environment overrides (used by CI):

    OPENJPH_INSTALL_DIR   install prefix (same as --prefix)
    OPENJPH_GIT_URL       git URL to clone (default: upstream OpenJPH)
    OPENJPH_GIT_REF       commit/tag/branch to build (default: pinned below)
    CMAKE_OSX_ARCHITECTURES  forwarded to CMake on macOS (e.g. "arm64")
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Pinned to a released OpenJPH tag (>= 0.30.1, the minimum the bindings need)
# so the wheels are reproducible. Bump it deliberately, not automatically.
DEFAULT_GIT_URL = "https://github.com/aous72/OpenJPH.git"
DEFAULT_GIT_REF = "0.30.1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd, **kwargs):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def clone(source_dir: Path, url: str, ref: str) -> None:
    if (source_dir / ".git").is_dir():
        print(f"Reusing existing OpenJPH checkout at {source_dir}", flush=True)
    else:
        if source_dir.exists():
            shutil.rmtree(source_dir)
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        # A blobless partial clone is enough to fetch an arbitrary commit cheaply.
        run(["git", "init", "-q", source_dir])
        run(["git", "-C", source_dir, "remote", "add", "origin", url])
    run(["git", "-C", source_dir, "fetch", "--depth", "1", "origin", ref])
    run(["git", "-C", source_dir, "checkout", "-q", "FETCH_HEAD"])


def cmake_configure(source_dir: Path, build_dir: Path, prefix: Path) -> None:
    args = [
        "cmake",
        "-S", source_dir,
        "-B", build_dir,
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        "-DCMAKE_BUILD_TYPE=Release",
        # Static library only -- nothing to bundle/repair at runtime.
        "-DBUILD_SHARED_LIBS=OFF",
        # -fPIC so the static lib can be linked into our shared extension module.
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        # Trim everything we don't need for the bindings.
        "-DOJPH_BUILD_EXECUTABLES=OFF",
        "-DOJPH_BUILD_TESTS=OFF",
        "-DOJPH_ENABLE_TIFF_SUPPORT=OFF",
    ]

    osx_archs = os.environ.get("CMAKE_OSX_ARCHITECTURES")
    if sys.platform == "darwin" and osx_archs:
        args.append(f"-DCMAKE_OSX_ARCHITECTURES={osx_archs}")

    if os.environ.get("MACOSX_DEPLOYMENT_TARGET"):
        args.append(
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={os.environ['MACOSX_DEPLOYMENT_TARGET']}"
        )

    # Use a multi-arch-friendly generator on Windows; default elsewhere.
    if os.name == "nt":
        # Ninja is available in the cibuildwheel Windows image and avoids the
        # MSBuild architecture guessing that trips up ARM64 cross builds.
        if shutil.which("ninja"):
            args += ["-G", "Ninja"]

    run(args)


def cmake_build_install(build_dir: Path, jobs: int) -> None:
    run([
        "cmake", "--build", build_dir,
        "--config", "Release",
        "--parallel", str(jobs),
    ])
    run(["cmake", "--install", build_dir, "--config", "Release"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default=os.environ.get(
            "OPENJPH_INSTALL_DIR", str(PROJECT_ROOT / "openjph-install")
        ),
        help="install prefix for the static OpenJPH (default: <project>/openjph-install)",
    )
    parser.add_argument(
        "--source-dir",
        default=str(PROJECT_ROOT / "build" / "openjph-src"),
        help="where to clone the OpenJPH sources",
    )
    parser.add_argument(
        "--jobs", type=int, default=os.cpu_count() or 2,
        help="parallel build jobs",
    )
    args = parser.parse_args()

    prefix = Path(args.prefix).resolve()
    source_dir = Path(args.source_dir).resolve()
    build_dir = source_dir.parent / "openjph-build"

    url = os.environ.get("OPENJPH_GIT_URL", DEFAULT_GIT_URL)
    ref = os.environ.get("OPENJPH_GIT_REF", DEFAULT_GIT_REF)

    print(f"Building OpenJPH {ref}\n  from   {url}\n  into   {prefix}", flush=True)

    clone(source_dir, url, ref)
    # Reconfigure from scratch so stale cache (e.g. a prior arch) never leaks in.
    if build_dir.exists():
        shutil.rmtree(build_dir)
    cmake_configure(source_dir, build_dir, prefix)
    cmake_build_install(build_dir, args.jobs)

    # Sanity check: the static archive and headers must exist where setup.py looks.
    libdir_candidates = [prefix / "lib", prefix / "lib64"]
    found = []
    for libdir in libdir_candidates:
        if libdir.is_dir():
            found += [p for p in libdir.iterdir()
                      if p.suffix in (".a", ".lib")]
    incdir = prefix / "include" / "openjph"
    print(f"Installed headers present: {incdir.is_dir()}", flush=True)
    print(f"Installed static libs: {[str(p) for p in found]}", flush=True)
    if not found:
        print("ERROR: no static OpenJPH library was installed", file=sys.stderr)
        return 1
    if not incdir.is_dir():
        print("ERROR: OpenJPH headers were not installed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
