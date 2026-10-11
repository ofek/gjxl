// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "modular_population_params.h"
using namespace metal;
using namespace gjxl_modular_population;

kernel void gjxl_modular_population_count(
    device const uint *values [[buffer(0)]], device const ushort *contexts [[buffer(1)]],
    device uchar *scratch [[buffer(2)]], device ulong *output [[buffer(3)]],
    constant Params &p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  device atomic_uint *counts = reinterpret_cast<device atomic_uint *>(scratch) +
      ulong(group % p.shards) * p.bins;
  device atomic_uint *status = reinterpret_cast<device atomic_uint *>(scratch + p.status_offset);
  uint previous = 0, run = 0, wide = 0, invalid = 0;
  const ulong end = min(ulong(p.tokens), (ulong(group) + 1) * kGroupTokens);
  for (ulong i = ulong(group) * kGroupTokens + lane; i < end; i += kThreads) {
    const uint context = contexts[i], value = values[i];
    if (context >= p.contexts) { invalid = 1; continue; }
    if (value >= kValueLimit) { ++wide; continue; }
    const uint index = context * kValueLimit + value;
    if (run && index != previous) {
      atomic_fetch_add_explicit(counts + previous, run, memory_order_relaxed);
      run = 0;
    }
    previous = index;
    ++run;
  }
  if (run) atomic_fetch_add_explicit(counts + previous, run, memory_order_relaxed);
  if (invalid) atomic_fetch_or_explicit(status, invalid, memory_order_relaxed);
  if (wide) atomic_fetch_add_explicit(status + 1, wide, memory_order_relaxed);
}
kernel void gjxl_modular_population_reduce(
    device const uint *values [[buffer(0)]], device const ushort *contexts [[buffer(1)]],
    device const uint *scratch [[buffer(2)]], device ulong *output [[buffer(3)]],
    constant Params &p [[buffer(4)]], uint index [[thread_position_in_grid]]) {
  if (index >= p.bins) return;
  ulong sum = 0;
  for (uint shard = 0; shard < p.shards; ++shard) sum += scratch[ulong(shard) * p.bins + index];
  output[index] = sum;
}
