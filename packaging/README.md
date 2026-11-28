# 3p-lodestar-common

`lodestar_common` ships to the Vulkanstorm viewer as an autobuild package, the way meshoptimizer does. This
folder builds the package. The `3p-lodestar-common` repository itself is a later step.

## Build the archive

```
cmake -DPLATFORM=windows64 -P packaging/package-common.cmake
cmake -DPLATFORM=linux64 -DGENERATOR=Ninja -P packaging/package-common.cmake
```

Optional: `-DOUTPUT_DIR=<dir>`, and `-DGLM_SOURCE=<glm checkout>` to build offline against a glm that is already
on disk (otherwise glm 1.0.1 is fetched). The script configures with `-DLODESTAR_COMMON_ONLY=ON`, builds
`lodestar_common`, installs it and writes `lodestar-common-<version>-<platform>.tar.bz2` plus its MD5.

Contents:

```
include/lodestar/{lodestar,surface_calculus,record,metric,mesh_metrics,silhouette_metrics,topology}.hpp
lib/release/lodestar_common.lib | liblodestar_common.a
lib/debug/lodestar_common.lib            (Windows only)
LICENSES/lodestar-common.txt             (MIT)
LODESTAR_COMMON_VERSION.txt              <project version>.r<record format>m<metric version>
```

The library depends on glm and the standard library only. glm is not bundled; the viewer links its own.
The importer (`loadMesh`, Assimp) is left out of the package (`LODESTAR_COMMON_IMPORT=OFF`).
The code is built with strict floating point (`/fp:strict`, `-ffp-contract=off`), so records agree across
platforms (the printed test records are identical between MSVC and GCC).

## What the `3p-lodestar-common` repository needs

- A copy of this repository's `LICENSE` (MIT) and the `NOTICE` text the viewer's `doc/licensing.md` audit lists.
- A build script for autobuild (`build-cmd.sh`) that runs `package-common.cmake` for the platform, or the same
  steps: configure with `-DLODESTAR_COMMON_ONLY=ON`, build and install `Release` (and `Debug` on Windows), with the
  Release runtime `/MD` and `/MDd` for Debug (the CMake default).
- A pin on a Lodestar commit or tag (a submodule or a fetched archive) and the version string above.
- The viewer's `autobuild.xml` entry (`lodestar_common`, platforms `windows64` and `linux64`; FreeBSD later) and a
  `cmake/LodestarCommon.cmake` that adds the include directory and links `lodestar_common`, next to
  `cmake/GLM.cmake`.
- Viewer builds that use LTO link the package fine (the objects are ordinary, not LTO objects). Build the package
  with the viewer's LTO flags if the viewer wants cross-module inlining.
- CI on the package repository: the package script on both platforms, then compile and run a one-file consumer
  (include `lodestar/metric.hpp`, call `measureAsset`, serialise the record).

## Licence

MIT. It is compatible with GPL-3.0-or-later; add the package to the viewer's licence audit.
