// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
namespace gjxl_tree_statistics {
using Word = uint;
using Cost = long;
using Byte = uchar;
#else
#include <cstdint>
namespace gjxl_tree_statistics {
using Word = uint32_t;
using Cost = int64_t;
using Byte = uint8_t;
#endif
struct Code {
  Byte symbol, extra_bits;
};
struct Result {
  Cost low[14], high[14];
};
// The 2^20-sample ceiling bounds every byte offset below 2^32. Codes and
// ordered indices start at zero in their respective immutable/dispatch arenas.
struct Params {
  Word samples, leaf_samples, thresholds, reserved;
  Word count_costs_offset, boundaries_offset, histograms_offset, status_offset;
};
enum : Word { kIndexError = 1, kCodeError = 2, kHistogramError = 4 };
} // namespace gjxl_tree_statistics
