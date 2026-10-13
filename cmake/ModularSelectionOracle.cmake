# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
# Extract the private reference function verbatim, under a test-only namespace.
# No production target compiles this file or links a reference implementation.
function(gjxl_generate_wp_cost_oracle reference_source output)
  set(input "${reference_source}/lib/jxl/enc_modular.cc")
  file(SHA256 "${input}" digest)
  if(NOT digest STREQUAL "4bc466a0c7b2e65c03501eff1a81758d63a45568521a22ceb611cdbc67f0fb7a")
    message(FATAL_ERROR "WP cost oracle requires pristine pinned e8ff0976 enc_modular.cc")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${input}")
  file(READ "${input}" source)
  string(FIND "${source}" "float EstimateWPCost(" first)
  string(FIND "${source}" "\nbool do_transform(" stop)
  if(first LESS 0 OR stop LESS_EQUAL first)
    message(FATAL_ERROR "Cannot locate pinned WP estimator")
  endif()
  math(EXPR length "${stop} - ${first}")
  string(SUBSTRING "${source}" ${first} ${length} GJXL_REFERENCE_WP_BODY)
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tests/modular_wp_cost_reference.cpp.in"
    "${output}" @ONLY)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Match the pinned reference's math/vectorization settings, only here.
    set_source_files_properties("${output}" PROPERTIES COMPILE_OPTIONS
      "-fno-vectorize;-fno-slp-vectorize;-fno-sanitize=vptr;-fno-rtti;-fno-exceptions")
  endif()
endfunction()
