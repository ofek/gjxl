cmake_minimum_required(VERSION 3.25)
project(modular_learning_cost LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(GJXL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
foreach(var GJXL_BUILD REFERENCE_SOURCE REFERENCE_BUILD STUDY_DATA)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "Set ${var} to an absolute path")
  endif()
endforeach()
function(native name predictors keys skip)
  add_executable(native_${name} native_runner.cpp ${GJXL_SOURCE}/src/io/pnm.cpp
    ${STUDY_DATA}/overrides/search.cpp ${STUDY_DATA}/overrides/tree_learning.cpp)
  target_include_directories(native_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${GJXL_SOURCE}/include ${GJXL_SOURCE}/src
    ${GJXL_SOURCE}/tests ${REFERENCE_SOURCE} ${REFERENCE_SOURCE}/lib/include
    ${REFERENCE_BUILD}/lib/include ${REFERENCE_SOURCE}/third_party/highway)
  target_compile_definitions(native_${name} PRIVATE STUDY_PREDICTORS=${predictors} STUDY_KEYS=${keys} STUDY_SKIP_CONSTANT=${skip} STUDY_NAME="${name}")
  target_link_libraries(native_${name} PRIVATE
    ${GJXL_BUILD}/libgjxl_codestream.a ${GJXL_BUILD}/libgjxl_gpu_ops.a
    ${GJXL_BUILD}/libgjxl_codec.a ${GJXL_BUILD}/libgjxl_gpu.a ${GJXL_BUILD}/libgjxl_modular_reference.a
    ${REFERENCE_BUILD}/lib/libjxl-internal.a ${REFERENCE_BUILD}/lib/libjxl.dylib
    ${REFERENCE_BUILD}/third_party/highway/libhwy.a ${REFERENCE_BUILD}/lib/libjxl_cms.dylib
    ${REFERENCE_BUILD}/third_party/brotli/libbrotlienc.dylib
    ${REFERENCE_BUILD}/third_party/brotli/libbrotlidec.dylib
    ${REFERENCE_BUILD}/third_party/brotli/libbrotlicommon.dylib)
endfunction()
native(baseline 14 0 0)
native(two 2 0 0)
native(keys 14 1 0)
native(two_keys 2 1 0)
native(skip_constant 14 0 1)
if(REFERENCE_STUDY_DATA)
  add_executable(reference_runner ${REFERENCE_STUDY_DATA}/reference_runner.cpp
    ${REFERENCE_STUDY_DATA}/reference_effort.cpp
    ${REFERENCE_STUDY_DATA}/overrides/lib/jxl/enc_modular.cc
    ${REFERENCE_STUDY_DATA}/overrides/lib/jxl/modular/encoding/enc_encoding.cc
    ${REFERENCE_STUDY_DATA}/overrides/lib/jxl/modular/encoding/enc_ma.cc)
  target_include_directories(reference_runner PRIVATE
    ${REFERENCE_STUDY_DATA}/overrides ${CMAKE_CURRENT_SOURCE_DIR} ${GJXL_SOURCE}/tests
    ${REFERENCE_SOURCE} ${REFERENCE_SOURCE}/lib/jxl ${REFERENCE_SOURCE}/lib/include
    ${REFERENCE_BUILD}/lib/include ${REFERENCE_SOURCE}/third_party/highway)
  target_compile_definitions(reference_runner PRIVATE JXL_INTERNAL_LIBRARY_BUILD
    JPEGXL_ENABLE_SKCMS=1 JPEGXL_ENABLE_TRANSCODE_JPEG=0 JPEGXL_ENABLE_BOXES=1
    "HWY_DISABLED_TARGETS=(HWY_AVX3|HWY_AVX3_SPR|HWY_AVX3_ZEN4|HWY_RVV|HWY_SSSE3|HWY_SVE_256)")
  target_compile_options(reference_runner PRIVATE -fno-rtti -fno-slp-vectorize -fno-vectorize)
  target_link_libraries(reference_runner PRIVATE
    ${REFERENCE_BUILD}/lib/libjxl-internal.a ${REFERENCE_BUILD}/third_party/highway/libhwy.a
    ${REFERENCE_BUILD}/lib/libjxl_cms.dylib
    ${REFERENCE_BUILD}/third_party/brotli/libbrotlienc.dylib
    ${REFERENCE_BUILD}/third_party/brotli/libbrotlidec.dylib
    ${REFERENCE_BUILD}/third_party/brotli/libbrotlicommon.dylib)
endif()
