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
import tarfile
import urllib.request
from pathlib import Path

# Pinned to a released OpenJPH tag (>= 0.30.1, the minimum the bindings need)
# so the wheels are reproducible. Bump it deliberately, not automatically.
DEFAULT_GIT_URL = "https://github.com/hmaarrfk/OpenJPH.git"
DEFAULT_GIT_REF = "ojph"

# Google Highway is a hard build dependency on Linux, Windows, and macOS:
# without its headers the library silently falls back to generic (scalar)
# kernels. Header-only usage -- nothing is linked -- so a source tree is
# enough when no installed copy is found. Keep the version in sync with
# what conda-forge ships.
HWY_VERSION = "1.4.0"
HWY_URL = (
    "https://github.com/google/highway/archive/refs/tags/"
    f"{HWY_VERSION}.tar.gz"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd, **kwargs):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


SUBMODULE_DIR = PROJECT_ROOT / "subprojects" / "ojph"


def clone(source_dir: Path, url: str, ref: str) -> None:
    if (source_dir / ".git").is_dir() or (source_dir / ".git").is_file():
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


def find_or_fetch_hwy(download_root: Path, jobs: int):
    """Locate the Google Highway headers AND library, downloading and
    building a pinned release when no installed copy is found. Since the
    kernels use run-time dispatch (foreach_target), the hwy *library* is
    required, not just the headers. Returns (include_dir, library_path);
    library_path is None when an installed copy is used (CMake finds the
    library beside the headers)."""
    candidates = []
    if os.environ.get("OJPH_HWY_INCLUDE_DIR"):
        candidates.append(Path(os.environ["OJPH_HWY_INCLUDE_DIR"]))
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        candidates.append(Path(conda) / "include")
        candidates.append(Path(conda) / "Library" / "include")  # conda on Windows
    candidates += [Path("/usr/local/include"), Path("/usr/include")]
    for cand in candidates:
        if (cand / "hwy" / "highway.h").is_file():
            print(f"Using Google Highway from {cand}", flush=True)
            return cand, None

    src = download_root / f"highway-{HWY_VERSION}"
    if not (src / "hwy" / "highway.h").is_file():
        download_root.mkdir(parents=True, exist_ok=True)
        tarball = download_root / f"highway-{HWY_VERSION}.tar.gz"
        print(f"Downloading Google Highway {HWY_VERSION} from {HWY_URL}",
              flush=True)
        urllib.request.urlretrieve(HWY_URL, tarball)
        with tarfile.open(tarball) as tf:
            tf.extractall(download_root)
        tarball.unlink()
    if not (src / "hwy" / "highway.h").is_file():
        raise RuntimeError(f"Highway download did not produce {src}/hwy/highway.h")

    # Build the static hwy library so the wheel stays self-contained
    # (no runtime libhwy dependency).
    build_dir = download_root / f"highway-build-{HWY_VERSION}"
    lib_names = ("hwy.lib", "libhwy.a")
    def _find_lib():
        for name in lib_names:
            p = build_dir / name
            if p.is_file():
                return p
        return None
    if _find_lib() is None:
        args = [
            "cmake", "-S", src, "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            "-DBUILD_TESTING=OFF",
            "-DHWY_ENABLE_EXAMPLES=OFF",
            "-DHWY_ENABLE_CONTRIB=OFF",
            "-DHWY_ENABLE_TESTS=OFF",
        ]
        if os.name == "nt":
            args += ["-G", os.environ.get("CMAKE_GENERATOR", "Ninja")]
            if shutil.which("cl"):
                args += ["-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl"]
        if sys.platform == "darwin" and os.environ.get("CMAKE_OSX_ARCHITECTURES"):
            args.append(
                f"-DCMAKE_OSX_ARCHITECTURES={os.environ['CMAKE_OSX_ARCHITECTURES']}"
            )
        if os.environ.get("MACOSX_DEPLOYMENT_TARGET"):
            args.append(
                f"-DCMAKE_OSX_DEPLOYMENT_TARGET={os.environ['MACOSX_DEPLOYMENT_TARGET']}"
            )
        run(args)
        run(["cmake", "--build", build_dir, "--config", "Release",
             "--parallel", str(jobs), "--target", "hwy"])
        # Multi-config generators (VS) put the archive under Release/
        if _find_lib() is None:
            for name in lib_names:
                p = build_dir / "Release" / name
                if p.is_file():
                    p.replace(build_dir / name)
    lib = _find_lib()
    if lib is None:
        raise RuntimeError("building the static Highway library failed")
    print(f"Using Google Highway from {src} (static {lib.name})", flush=True)
    return src, lib


def cmake_configure(source_dir: Path, build_dir: Path, prefix: Path,
                    hwy_include, hwy_library) -> None:
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

    if hwy_include is not None:
        args += [
            f"-DOJPH_HWY_INCLUDE_DIR={hwy_include}",
            # Hard dependency: fail the build rather than silently fall
            # back to the scalar kernels.
            "-DOJPH_REQUIRE_HWY=ON",
        ]
    if hwy_library is not None:
        args.append(f"-DOJPH_HWY_LIBRARY={hwy_library}")

    osx_archs = os.environ.get("CMAKE_OSX_ARCHITECTURES")
    if sys.platform == "darwin" and osx_archs:
        args.append(f"-DCMAKE_OSX_ARCHITECTURES={osx_archs}")

    if os.environ.get("MACOSX_DEPLOYMENT_TARGET"):
        args.append(
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={os.environ['MACOSX_DEPLOYMENT_TARGET']}"
        )

    if os.name == "nt":
        # Build with Ninja driving MSVC's cl.exe, so OpenJPH uses the same
        # toolchain/ABI as the Python extension (setuptools uses cl.exe) and
        # produces an MSVC ``.lib``. This requires an active MSVC developer
        # environment (CI sets one up with ilammy/msvc-dev-cmd); cl.exe is then
        # first on PATH and Ninja auto-detects it. The target architecture comes
        # from that environment, so no ``-A`` is needed. We avoid the default
        # Visual Studio generator because it fails to locate a VS instance when
        # invoked from this (non-developer-prompt) subprocess.
        generator = os.environ.get("CMAKE_GENERATOR", "Ninja")
        args += ["-G", generator]
        # Pin both compilers to cl.exe. Conda-forge's `compilers` package now
        # ships clang on Windows, and with the conda env on PATH CMake would
        # otherwise pick clang for C and cl for CXX and reject the mix.
        if shutil.which("cl"):
            args += ["-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl"]
        if not shutil.which("cl") and generator == "Ninja":
            print(
                "WARNING: cl.exe not found on PATH; the MSVC developer "
                "environment may not be active.",
                file=sys.stderr,
            )

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
        default=None,
        help="OpenJPH sources; defaults to the subprojects/ojph submodule "
             "when present, else a fresh clone under build/openjph-src",
    )
    parser.add_argument(
        "--jobs", type=int, default=os.cpu_count() or 2,
        help="parallel build jobs",
    )
    args = parser.parse_args()

    prefix = Path(args.prefix).resolve()
    if args.source_dir is not None:
        source_dir = Path(args.source_dir).resolve()
    elif (SUBMODULE_DIR / "CMakeLists.txt").is_file():
        source_dir = SUBMODULE_DIR
    elif (PROJECT_ROOT / ".gitmodules").is_file():
        # submodule declared but not initialized
        run(["git", "-C", PROJECT_ROOT, "submodule", "update", "--init",
             "--depth", "1", str(SUBMODULE_DIR)])
        source_dir = SUBMODULE_DIR
    else:
        source_dir = (PROJECT_ROOT / "build" / "openjph-src").resolve()
    build_dir = PROJECT_ROOT / "build" / "openjph-build"

    url = os.environ.get("OPENJPH_GIT_URL", DEFAULT_GIT_URL)
    ref = os.environ.get("OPENJPH_GIT_REF", DEFAULT_GIT_REF)

    print(f"Building OpenJPH\n  from   {source_dir}\n  into   {prefix}",
          flush=True)

    if source_dir != SUBMODULE_DIR:
        clone(source_dir, url, ref)

    # Google Highway is required on the mainstream platforms; special
    # platforms (or an explicit OJPH_ALLOW_NO_HWY=1) fall back to the
    # fork's auto-detection.
    if os.environ.get("OJPH_ALLOW_NO_HWY") == "1":
        hwy_include, hwy_library = None, None
    elif sys.platform.startswith(("linux", "darwin", "win")):
        hwy_include, hwy_library = find_or_fetch_hwy(PROJECT_ROOT / "build",
                                                     args.jobs)
    else:
        hwy_include, hwy_library = None, None

    # Reconfigure from scratch so stale cache (e.g. a prior arch) never leaks in.
    if build_dir.exists():
        shutil.rmtree(build_dir)
    cmake_configure(source_dir, build_dir, prefix, hwy_include, hwy_library)
    cmake_build_install(build_dir, args.jobs)

    # Place the static hwy archive beside libojph so setup.py links both
    # into the extension (the hwy kernels dispatch through the library).
    if hwy_library is not None:
        libdir = prefix / "lib"
        libdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hwy_library, libdir / hwy_library.name)

    # Sanity check: the static archive and headers must exist where setup.py looks.
    libdir_candidates = [prefix / "lib", prefix / "lib64"]
    found = []
    for libdir in libdir_candidates:
        if libdir.is_dir():
            found += [p for p in libdir.iterdir()
                      if p.suffix in (".a", ".lib")]
    incdir = prefix / "include" / "ojph"
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
