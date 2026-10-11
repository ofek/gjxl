# WebGPU dependency setup

`GJXL_ENABLE_WEBGPU=ON` builds [Dawn](https://dawn.googlesource.com/dawn),
the native WebGPU implementation, in preparation for GJXL's WebGPU backend.
This currently provides the `dawn::webgpu_dawn` CMake target and an initialization
smoke test. It does not yet implement a GJXL backend, codec kernels, a public
WebGPU API, or Rust bindings. The option defaults to `OFF`; ordinary builds do
not configure or fetch Dawn's dependencies.

Dawn is a Git submodule at `third_party/dawn`, initially pinned to
`b73d662f73d57bb42209546910666a38e6db95a1` (2026-10-08). Its own `DEPS` file pins
the transitive dependencies. Dawn's BSD license and dependency notices are in
that checkout.

## Build and verify

Use CMake 3.25+, Ninja, Git, Python 3, and a C++20 compiler satisfying GJXL's
[storage toolchain contract](storage-toolchain.md). On Windows, run these commands
from a developer shell with the matching toolchain selected: MSVC 14.37 for
the audited x64 setup, or MSVC 14.44 for native ARM64. In PowerShell, the
repository's setup script selects the toolchain for the host architecture:

```powershell
& ./.github/actions/setup-native-toolchain/setup-msvc.ps1
```

See the [Windows ARM64 instructions](windows-arm64.md) for required components.
The initial configure downloads Dawn's dependencies and needs network access;
no `depot_tools` installation is needed.

```sh
git submodule update --init --depth 1 third_party/dawn
cmake -S . -B build/webgpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGJXL_ENABLE_WEBGPU=ON -DGJXL_ENABLE_METAL=OFF -DGJXL_ENABLE_CUDA=OFF -DGJXL_BUILD_TESTS=ON
cmake --build build/webgpu --target gjxl_webgpu_smoke_test --parallel 4
ctest --test-dir build/webgpu -R '^webgpu_smoke$' --output-on-failure
```

The smoke test creates, processes events on, and releases a native WebGPU
instance. It checks generated headers and native linking without requiring a
GPU or display server. It does not qualify adapter/device creation or shader
execution. To build only the dependency with `GJXL_BUILD_TESTS=OFF`, build the
`gjxl_dawn` target; it is also included in the default build when WebGPU is on.

## Configuration

The headless defaults select D3D12 on Windows, Metal on Apple platforms, and
Vulkan on Linux. Dawn's Metal backend is independent of `GJXL_ENABLE_METAL`.
OpenGL, OpenGL ES, D3D11, GLFW, X11, and Wayland support default to off, as do
Dawn/Tint tests, samples, benchmarks, and Tint command-line tools. Standard
`DAWN_*` and `TINT_*` CMake cache options can override these defaults. Use a
separate build directory when changing platform or toolchain.

Dawn defaults to a static library. With `-DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED`,
the Windows smoke-test build copies Dawn's DLL beside the test executable.

`DAWN_FETCH_DEPENDENCIES` defaults to `ON`. For an offline configure, populate
the pinned checkout's dependencies first, then pass
`-DDAWN_FETCH_DEPENDENCIES=OFF`. Updating the submodule may require fetching its
dependencies again. See Dawn's
[CMake quickstart](https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/quickstart-cmake.md)
for upstream dependency setup.

Future source-tree backend targets can link privately to `dawn::webgpu_dawn`
and include `<webgpu/webgpu_cpp.h>`. No Dawn dependency is added to GJXL's
installed package at this stage.
