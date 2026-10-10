// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#ifndef __METAL_VERSION__
#include <cstdint>
#endif

namespace gjxl_modular_tokens {
#ifdef __METAL_VERSION__
using Word = uint;
using Signed = int;
using Address = ulong;
#else
using Word = uint32_t;
using Signed = int32_t;
using Address = uint64_t;
#endif
// Matches the fixed-width private operation ABI; the host checks every offset.
struct Slice {
  Address token_offset;
  Word stream_id, channel, x, y, width, height;
};
struct Node {
  Signed property, split;
  Word left, right, predictor;
  Signed offset;
  Word multiplier, context;
};
struct Params {
  Address plane_offset_elements[4];
  Address contexts_offset_bytes, nodes_offset_bytes, status_offset_bytes;
  Word source_width, slice_count, node_count;
};
// Qualification-only trace. Production kernels do not allocate or write it.
struct Diagnostic {
  Signed properties[15], predictions[14];
  Word leaf;
  Signed selected_prediction, sample, residual;
  Word value, context;
};
enum : Word { kResidualError = 1 };
} // namespace gjxl_modular_tokens
