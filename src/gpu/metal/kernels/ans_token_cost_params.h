// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
namespace gjxl_ans_cost {
using Word = uint;
using Half = ushort;
using Address = ulong;
using Byte = uchar;
#else
#include <cstdint>
namespace gjxl_ans_cost {
using Word = uint32_t;
using Half = uint16_t;
using Address = uint64_t;
using Byte = uint8_t;
#endif
struct Section {
  Address offset, count;
};
struct Result {
  Address bits;
  Word final_state, reserved;
};
struct Model {
  Address context_offset, clusters_offset;
  Word context_count, cluster_count, reserved[2];
};
struct Cluster {
  Word split_exponent, msb_in_token, lsb_in_token, symbol_count;
  Half frequencies[256], reverse_offsets[256];
  Address reciprocals[256];
  Half reverse_map[4096];
  Byte padding[240];
};
struct Params {
  Address models_offset, status_offset;
  Word sections, candidates;
};
enum : Word { kContextError = 1, kSymbolError = 2, kStateError = 4 };
} // namespace gjxl_ans_cost
