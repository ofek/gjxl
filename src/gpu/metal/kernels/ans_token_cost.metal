// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Exact recurrence and HybridUint arithmetic follow the scalar ANS encoder.
#include "ans_token_cost_params.h"
#include <metal_stdlib>
using namespace metal;
using namespace gjxl_ans_cost;

// One SIMD group owns one ordered recurrence. Lanes prepare a reverse-order
// block in parallel; shuffles then expose those records in the scalar order.
// Every lane maintains the same state, making the dependent reverse-map read
// uniform. This increases occupancy and coalesces value/context reads without
// splitting a section into different rANS states or allocating token scratch.
kernel void
gjxl_ans_token_cost(device const uint *values [[buffer(0)]],
                    device const ushort *contexts [[buffer(1)]],
                    device uchar *dispatch [[buffer(2)]], device Result *output [[buffer(3)]],
                    constant Params &p [[buffer(4)]], uint2 id [[thread_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]) {
  const uint section_id = id.x / width;
  if (section_id >= p.sections || id.y >= p.candidates)
    return;
  const Section section = reinterpret_cast<device const Section *>(dispatch)[section_id];
  const Model model = reinterpret_cast<device const Model *>(dispatch + p.models_offset)[id.y];
  device const uchar *context_map = dispatch + model.context_offset;
  device const Cluster *clusters =
      reinterpret_cast<device const Cluster *>(dispatch + model.clusters_offset);
  device atomic_uint *status = reinterpret_cast<device atomic_uint *>(dispatch + p.status_offset);
  uint state = 0x13u << 16;
  ulong bits = 32;
  for (ulong remaining = section.count; remaining != 0;) {
    const uint active = uint(min(remaining, ulong(width)));
    uint cluster_id = 0, frequency = 0, reverse_offset = 0, extra_bits = 0, error = 0;
    uint reciprocal = 0;
    if (lane < active) {
      const ulong index = section.offset + remaining - 1 - lane;
      const uint context = contexts[index];
      if (context >= model.context_count) {
        error = kContextError;
      } else {
        cluster_id = context_map[context];
        device const Cluster &cluster = clusters[cluster_id];
        const uint value = values[index], split = 1u << cluster.split_exponent;
        uint symbol = value;
        if (value >= split) {
          const uint exponent = 31u - clz(value);
          const uint mantissa = value - (1u << exponent);
          symbol = split +
                   ((exponent - cluster.split_exponent)
                    << (cluster.msb_in_token + cluster.lsb_in_token)) +
                   ((mantissa >> (exponent - cluster.msb_in_token)) << cluster.lsb_in_token) +
                   (mantissa & ((1u << cluster.lsb_in_token) - 1));
          extra_bits = exponent - cluster.msb_in_token - cluster.lsb_in_token;
        }
        if (symbol >= cluster.symbol_count || cluster.frequencies[symbol] == 0) {
          error = kSymbolError;
        } else {
          frequency = cluster.frequencies[symbol];
          reverse_offset = cluster.reverse_offsets[symbol];
          // ceil(ceil(2^44/f)/2^12) = ceil(2^32/f). For f=1,
          // the truncated zero is unused by the direct quotient below.
          reciprocal = uint((cluster.reciprocals[symbol] + 4095) >> 12);
        }
      }
    }
    if (error)
      atomic_fetch_or_explicit(status, error, memory_order_relaxed);
    if (simd_any(error != 0))
      return;
    // f uses 13 bits (including 4096), extra bits use 5 and cluster uses 6.
    const uint record = frequency | (extra_bits << 13) | (cluster_id << 18);
    for (uint token = 0; token < active; ++token) {
      const uint info = simd_shuffle(record, token);
      const uint f = info & 8191u, cluster = info >> 18;
      const uint offset = simd_shuffle(reverse_offset, token);
      const uint multiplier = simd_shuffle(reciprocal, token);
      bits += (info >> 13) & 31u;
      if ((state >> 20) >= f) {
        bits += 16;
        state >>= 16;
      }
      // ceil(2^32/f) overestimates the quotient by at most one. For
      // f<4096, normalized state < f*2^20 leaves >1M of uint32 headroom,
      // so quotient*f cannot wrap; f=4096 divides exactly. One correction
      // therefore gives ordinary integer division without a 64-bit multiply.
      uint quotient = f == 1 ? state : mulhi(state, multiplier);
      uint product = quotient * f;
      if (product > state) {
        --quotient;
        product -= f;
      }
      const uint remainder = state - product;
      const uint reverse_index = offset + remainder;
      if (remainder >= f || reverse_index >= 4096 ||
          clusters[cluster].reverse_map[reverse_index] >= 4096) {
        if (lane == 0)
          atomic_fetch_or_explicit(status, uint(kStateError), memory_order_relaxed);
        return;
      }
      state = (quotient << 12) + clusters[cluster].reverse_map[reverse_index];
    }
    remaining -= active;
  }
  if (lane == 0)
    output[ulong(id.y) * p.sections + section_id] = {bits, state, 0};
}
