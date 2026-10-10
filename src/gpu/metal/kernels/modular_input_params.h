// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#ifndef __METAL_VERSION__
#include <cstdint>
#endif

namespace gjxl_modular_input {
#ifdef __METAL_VERSION__
using Word = uint;
using Address = ulong;
#else
using Word = uint32_t;
using Address = uint64_t;
#endif

// Shared host/shader ABI. Input addresses are bytes; output addresses are int32
// elements relative to the bound buffer's start. All products use 64 bits.
struct Params {
  Address input_offset_bytes;
  Address input_row_stride_bytes;
  Address output_offset_elements[4];
  Word width;
  Word height;
  Word channels;
  Word bytes_per_sample;
  Word big_endian;
  Word rct;
};
} // namespace gjxl_modular_input
