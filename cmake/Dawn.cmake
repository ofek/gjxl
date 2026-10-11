# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

include_guard(GLOBAL)

function(gjxl_add_dawn)
  set(dawn_source_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../third_party/dawn")
  if(NOT EXISTS "${dawn_source_dir}/CMakeLists.txt")
    message(FATAL_ERROR
      "GJXL_ENABLE_WEBGPU requires the pinned Dawn submodule. Run: "
      "git submodule update --init --depth 1 third_party/dawn")
  endif()

  # Dawn's DEPS file pins its transitive dependencies. Its fetcher avoids a
  # depot_tools requirement; a pre-populated checkout can opt out for offline use.
  set(DAWN_FETCH_DEPENDENCIES ON CACHE BOOL "Fetch Dawn's pinned dependencies")

  # Headless native compute defaults. Cache defaults deliberately allow callers
  # to select other Dawn backends without editing the vendored project.
  set(DAWN_ENABLE_D3D11 OFF CACHE BOOL "Build Dawn's D3D11 backend")
  set(DAWN_ENABLE_DESKTOP_GL OFF CACHE BOOL "Build Dawn's OpenGL backend")
  set(DAWN_ENABLE_OPENGLES OFF CACHE BOOL "Build Dawn's OpenGL ES backend")
  set(dawn_vulkan_default OFF)
  if(UNIX AND NOT APPLE)
    set(dawn_vulkan_default ON)
  endif()
  set(DAWN_ENABLE_VULKAN ${dawn_vulkan_default} CACHE BOOL "Build Dawn's Vulkan backend")
  set(DAWN_USE_GLFW OFF CACHE BOOL "Build Dawn's GLFW helpers")
  set(DAWN_USE_X11 OFF CACHE BOOL "Build Dawn's X11 surface support")
  set(DAWN_USE_WAYLAND OFF CACHE BOOL "Build Dawn's Wayland surface support")

  set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "Build Dawn samples")
  set(DAWN_BUILD_TESTS OFF CACHE BOOL "Build Dawn tests")
  set(DAWN_BUILD_BENCHMARKS OFF CACHE BOOL "Build Dawn benchmarks")
  set(DAWN_BUILD_PROTOBUF OFF CACHE BOOL "Build Dawn protobuf support")
  set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "Build Tint command-line tools")
  set(TINT_BUILD_TESTS OFF CACHE BOOL "Build Tint tests")
  set(TINT_BUILD_BENCHMARKS OFF CACHE BOOL "Build Tint benchmarks")
  set(TINT_BUILD_GLSL_VALIDATOR OFF CACHE BOOL "Build Tint's GLSL validator")
  set(DAWN_BUILD_MONOLITHIC_LIBRARY STATIC CACHE STRING "Dawn library type")

  # This dependency is private to the source build for now. Do not add Dawn's
  # headers, install rules, or compiler flags to GJXL's existing public targets.
  # Scope BUILD_SHARED_LIBS here because Dawn's monolithic library requires its
  # component libraries to be static, even when embedding projects use shared libs.
  set(BUILD_SHARED_LIBS OFF)
  add_subdirectory("${dawn_source_dir}" "${CMAKE_CURRENT_BINARY_DIR}/third_party/dawn"
    EXCLUDE_FROM_ALL)
  if(NOT TARGET dawn::webgpu_dawn)
    message(FATAL_ERROR
      "GJXL_ENABLE_WEBGPU requires DAWN_BUILD_MONOLITHIC_LIBRARY=STATIC or SHARED")
  endif()
  add_custom_target(gjxl_dawn ALL DEPENDS dawn::webgpu_dawn)
endfunction()
