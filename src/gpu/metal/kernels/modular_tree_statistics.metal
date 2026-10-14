// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "modular_tree_statistics_params.h"
#include <metal_stdlib>
using namespace metal;
using namespace gjxl_tree_statistics;

// One group per bucket/predictor, one lane per residual symbol. The sorted
// indices and boundaries are immutable CPU-selected inputs. Group-local atomics
// count samples exactly, independent of execution order.
kernel void
gjxl_tree_bucket_histograms(device const uchar *prepared [[buffer(0)]],
                            device uchar *dispatch [[buffer(1)]],
                            constant Params &p [[buffer(3)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_threadgroup]]) {
  const uint predictor = group.x, bucket = group.y;
  device const Code *codes = reinterpret_cast<device const Code *>(prepared);
  device const uint *order = reinterpret_cast<device const uint *>(dispatch);
  device const uint *boundaries =
      reinterpret_cast<device const uint *>(dispatch + p.boundaries_offset);
  device uint *histograms =
      reinterpret_cast<device uint *>(dispatch + p.histograms_offset);
  device atomic_uint *status =
      reinterpret_cast<device atomic_uint *>(dispatch + p.status_offset);
  threadgroup atomic_uint counts[128];
  atomic_store_explicit(&counts[lane], 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = boundaries[bucket] + lane; i < boundaries[bucket + 1];
       i += 128) {
    const uint sample = order[i];
    if (sample >= p.samples) {
      atomic_fetch_or_explicit(status, uint(kIndexError), memory_order_relaxed);
      continue;
    }
    const Code code = codes[sample * 14 + predictor];
    const uint extra = code.symbol < 16 ? 0 : 2 + (uint(code.symbol) - 16) / 4;
    if (code.symbol >= 128 || code.extra_bits != extra) {
      atomic_fetch_or_explicit(status, uint(kCodeError), memory_order_relaxed);
      continue;
    }
    atomic_fetch_add_explicit(&counts[code.symbol], 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  histograms[(bucket * 14 + predictor) * 128 + lane] =
      atomic_load_explicit(&counts[lane], memory_order_relaxed);
}

// One group per threshold/side/predictor. All reductions are bounded integers;
// there is no floating-point entropy approximation or parallel tie selection.
kernel void gjxl_tree_split_costs(device const uchar *prepared [[buffer(0)]],
                                  device uchar *dispatch [[buffer(1)]],
                                  device Result *output [[buffer(2)]],
                                  constant Params &p [[buffer(3)]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_threadgroup]]) {
  const uint predictor = group.x, threshold = group.y, side = group.z;
  device const long *table =
      reinterpret_cast<device const long *>(prepared + p.count_costs_offset);
  device const uint *boundaries =
      reinterpret_cast<device const uint *>(dispatch + p.boundaries_offset);
  device const uint *histograms =
      reinterpret_cast<device const uint *>(dispatch + p.histograms_offset);
  device atomic_uint *status =
      reinterpret_cast<device atomic_uint *>(dispatch + p.status_offset);
  const uint begin = side ? threshold + 1 : 0;
  const uint end = side ? p.thresholds + 1 : threshold + 1;
  const uint population = side ? p.leaf_samples - boundaries[threshold + 1]
                               : boundaries[threshold + 1];
  uint count = 0;
  for (uint bucket = begin; bucket < end; ++bucket)
    count += histograms[(bucket * 14 + predictor) * 128 + lane];
  threadgroup long costs[128];
  threadgroup uint populations[128];
  populations[lane] = count;
  costs[lane] = 0;
  if (count > p.samples) {
    atomic_fetch_or_explicit(status, uint(kHistogramError),
                             memory_order_relaxed);
  } else {
    const uint extra = lane < 16 ? 0 : 2 + (lane - 16) / 4;
    costs[lane] =
        -table[count] + (long(count) * extra + 2 * long(count != 0)) * 4096;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 64; stride; stride >>= 1) {
    if (lane < stride) {
      costs[lane] += costs[lane + stride];
      populations[lane] += populations[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) {
    if (populations[0] != population) {
      atomic_fetch_or_explicit(status, uint(kHistogramError),
                               memory_order_relaxed);
      return;
    }
    const long cost = table[population] + costs[0] + 8 * 4096;
    if (side)
      output[threshold].high[predictor] = cost;
    else
      output[threshold].low[predictor] = cost;
  }
}
