cmake_minimum_required(VERSION 3.25)
project(modular_entropy_study LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(GJXL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
foreach(var GJXL_BUILD REFERENCE_SOURCE REFERENCE_BUILD)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "Set ${var} to the qualified absolute path")
  endif()
endforeach()
add_executable(entropy_replay replay.cpp ${GJXL_SOURCE}/src/io/pnm.cpp)
target_include_directories(entropy_replay PRIVATE ${GJXL_SOURCE}/include ${GJXL_SOURCE}/src
  ${GJXL_SOURCE}/tests ${REFERENCE_SOURCE} ${REFERENCE_SOURCE}/lib/include
  ${REFERENCE_BUILD}/lib/include ${REFERENCE_SOURCE}/third_party/highway)
target_link_libraries(entropy_replay PRIVATE
  ${GJXL_BUILD}/libgjxl_codestream.a ${GJXL_BUILD}/libgjxl_gpu_ops.a
  ${GJXL_BUILD}/libgjxl_codec.a ${GJXL_BUILD}/libgjxl_gpu.a
  ${GJXL_BUILD}/libgjxl_modular_reference.a
  ${REFERENCE_BUILD}/lib/libjxl-internal.a ${REFERENCE_BUILD}/lib/libjxl.dylib
  ${REFERENCE_BUILD}/third_party/highway/libhwy.a ${REFERENCE_BUILD}/lib/libjxl_cms.dylib
  ${REFERENCE_BUILD}/third_party/brotli/libbrotlienc.dylib
  ${REFERENCE_BUILD}/third_party/brotli/libbrotlidec.dylib
  ${REFERENCE_BUILD}/third_party/brotli/libbrotlicommon.dylib)
