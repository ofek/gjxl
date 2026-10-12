cmake_minimum_required(VERSION 3.25)
project(modular_reference_learning LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(GJXL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
foreach(var GJXL_BUILD REFERENCE_SOURCE REFERENCE_BUILD GENERATED_SOURCE)
  if(NOT DEFINED ${var} OR NOT IS_ABSOLUTE "${${var}}")
    message(FATAL_ERROR "Set ${var} to an absolute path")
  endif()
endforeach()
add_executable(reference_learning runner.cpp reference_capture.cpp
  ${GJXL_SOURCE}/src/io/pnm.cpp
  ${GENERATED_SOURCE}/reference_codec.cpp
  ${GENERATED_SOURCE}/lib/jxl/modular/encoding/enc_ma.cc
  ${GENERATED_SOURCE}/lib/jxl/modular/encoding/enc_encoding.cc)
target_include_directories(reference_learning PRIVATE
  ${GENERATED_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}
  ${GJXL_SOURCE}/include ${GJXL_SOURCE}/src ${GJXL_SOURCE}/tests
  ${REFERENCE_SOURCE} ${REFERENCE_SOURCE}/lib/include
  ${REFERENCE_BUILD}/lib/include ${REFERENCE_SOURCE}/third_party/highway)
target_compile_definitions(reference_learning PRIVATE JXL_INTERNAL_LIBRARY_BUILD
  JPEGXL_ENABLE_SKCMS=1 JPEGXL_ENABLE_TRANSCODE_JPEG=0 JPEGXL_ENABLE_BOXES=1
  "HWY_DISABLED_TARGETS=(HWY_AVX3|HWY_AVX3_SPR|HWY_AVX3_ZEN4|HWY_RVV|HWY_SSSE3|HWY_SVE_256)")
# Match the pinned reference's floating-point/vectorization configuration.
target_compile_options(reference_learning PRIVATE -fno-slp-vectorize -fno-vectorize)
set_source_files_properties(
  ${GENERATED_SOURCE}/lib/jxl/modular/encoding/enc_ma.cc
  ${GENERATED_SOURCE}/lib/jxl/modular/encoding/enc_encoding.cc
  # Pair these flags as in the pinned build. On Apple libc++, no RTTI with
  # exceptions enabled emits private std::exception typeinfo that breaks
  # catches in the RTTI-enabled harness.
  PROPERTIES COMPILE_OPTIONS "-fno-rtti;-fno-exceptions")
target_link_libraries(reference_learning PRIVATE
  ${GJXL_BUILD}/libgjxl_codestream.a ${GJXL_BUILD}/libgjxl_gpu_ops.a
  ${GJXL_BUILD}/libgjxl_codec.a ${GJXL_BUILD}/libgjxl_gpu.a
  ${REFERENCE_BUILD}/lib/libjxl-internal.a
  ${REFERENCE_BUILD}/third_party/highway/libhwy.a
  ${REFERENCE_BUILD}/lib/libjxl_cms.dylib
  ${REFERENCE_BUILD}/third_party/brotli/libbrotlienc.dylib
  ${REFERENCE_BUILD}/third_party/brotli/libbrotlidec.dylib
  ${REFERENCE_BUILD}/third_party/brotli/libbrotlicommon.dylib)
