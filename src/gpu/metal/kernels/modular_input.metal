// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// RCT arithmetic adapted from the pinned scalar libjxl oracle.
#include "modular_input_params.h"
#include <metal_stdlib>
using namespace metal;

namespace {
// Floor division, including negative odd differences. Inputs are in [-M, M]
// for M <= 65535, so negation and the rounding adjustment cannot overflow.
int floor_half(int v) { return v >= 0 ? v / 2 : -((-v + 1) / 2); }

int load_sample(device const uchar *packed, ulong offset, constant gjxl_modular_input::Params &p) {
  const uint first = packed[offset];
  if (p.bytes_per_sample == 1)
    return int(first);
  const uint second = packed[offset + 1];
  return int(p.big_endian ? (first << 8) | second : first | (second << 8));
}
} // namespace

kernel void gjxl_modular_unpack_rct(device const uchar *packed [[buffer(0)]],
                                    device int *output [[buffer(1)]],
                                    constant gjxl_modular_input::Params &p [[buffer(2)]],
                                    uint2 xy [[thread_position_in_grid]]) {
  if (xy.x >= p.width || xy.y >= p.height)
    return;
  const ulong source = p.input_offset_bytes + ulong(xy.y) * p.input_row_stride_bytes +
                       ulong(xy.x) * p.channels * p.bytes_per_sample;
  const ulong pixel = ulong(xy.y) * p.width + xy.x;
  int values[4];
  for (uint c = 0; c < p.channels; ++c)
    values[c] = load_sample(packed, source + ulong(c) * p.bytes_per_sample, p);

  if (p.channels >= 3) {
    const uint permutation = p.rct / 7, transform = p.rct % 7;
    int a = values[permutation % 3];
    int b = values[(permutation + 1 + permutation / 3) % 3];
    int c = values[(permutation + 2 - permutation / 3) % 3];
    // For unsigned source samples in [0,M], M <= 65535: a+c <= 2M,
    // differences are in [-M,M], and tmp=floor((a+c)/2) remains in [0,M].
    // In type 6 the final luma is floor((tmp+b)/2), also in [0,M]. Thus
    // every value and intermediate fits signed int32 (largest sum 131070).
    if (transform == 6) {
      const int co = a - c, tmp = c + floor_half(co), cg = b - tmp;
      a = tmp + floor_half(cg);
      b = co;
      c = cg;
    } else {
      if (transform / 2 == 1)
        b -= a;
      if (transform / 2 == 2)
        b -= (a + c) / 2;
      if (transform & 1)
        c -= a;
    }
    values[0] = a;
    values[1] = b;
    values[2] = c;
  }
  // Alpha, including fully transparent pixels' color values, is not altered.
  for (uint c = 0; c < p.channels; ++c)
    output[p.output_offset_elements[c] + pixel] = values[c];
}
