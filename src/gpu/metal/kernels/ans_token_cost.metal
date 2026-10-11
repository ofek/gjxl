// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Exact recurrence and HybridUint arithmetic follow the scalar ANS encoder.
#include "ans_token_cost_params.h"
#include <metal_stdlib>
using namespace metal;
using namespace gjxl_ans_cost;

kernel void gjxl_ans_token_cost(device const uint *values [[buffer(0)]],
                                device const ushort *contexts [[buffer(1)]],
                                device uchar *dispatch [[buffer(2)]],
                                device Result *output [[buffer(3)]],
                                constant Params &p [[buffer(4)]],
                                uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.sections || id.y >= p.candidates)
    return;
  const Section section = reinterpret_cast<device const Section *>(dispatch)[id.x];
  const Model model = reinterpret_cast<device const Model *>(dispatch + p.models_offset)[id.y];
  device const uchar *context_map = dispatch + model.context_offset;
  device const Cluster *clusters =
      reinterpret_cast<device const Cluster *>(dispatch + model.clusters_offset);
  device atomic_uint *status = reinterpret_cast<device atomic_uint *>(dispatch + p.status_offset);
  uint state = 0x13u << 16;
  ulong bits = 32;
  for (ulong i = section.count; i != 0; --i) {
    const ulong index = section.offset + i - 1;
    const uint context = contexts[index];
    if (context >= model.context_count) {
      atomic_fetch_or_explicit(status, uint(kContextError), memory_order_relaxed);
      return;
    }
    device const Cluster &cluster = clusters[context_map[context]];
    const uint value = values[index], split = 1u << cluster.split_exponent;
    uint symbol = value;
    if (value >= split) {
      const uint exponent = 31u - clz(value);
      const uint mantissa = value - (1u << exponent);
      symbol =
          split +
          ((exponent - cluster.split_exponent) << (cluster.msb_in_token + cluster.lsb_in_token)) +
          ((mantissa >> (exponent - cluster.msb_in_token)) << cluster.lsb_in_token) +
          (mantissa & ((1u << cluster.lsb_in_token) - 1));
      bits += exponent - cluster.msb_in_token - cluster.lsb_in_token;
    }
    if (symbol >= cluster.symbol_count || cluster.frequencies[symbol] == 0) {
      atomic_fetch_or_explicit(status, uint(kSymbolError), memory_order_relaxed);
      return;
    }
    const uint frequency = cluster.frequencies[symbol];
    if ((state >> 20) >= frequency) {
      bits += 16;
      state >>= 16;
    }
    const uint quotient = uint((ulong(state) * cluster.reciprocals[symbol]) >> 44);
    const uint remainder = state - quotient * frequency;
    const uint reverse_index = cluster.reverse_offsets[symbol] + remainder;
    if (remainder >= frequency || reverse_index >= 4096 ||
        cluster.reverse_map[reverse_index] >= 4096) {
      atomic_fetch_or_explicit(status, uint(kStateError), memory_order_relaxed);
      return;
    }
    state = (quotient << 12) + cluster.reverse_map[reverse_index];
  }
  output[ulong(id.y) * p.sections + id.x] = {bits, state, 0};
}
