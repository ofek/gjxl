// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Nonweighted prediction arithmetic follows the pinned scalar libjxl oracle.
#include "modular_token_params.h"
#include <metal_stdlib>
using namespace metal;
using namespace gjxl_modular_tokens;

namespace {
struct Neighbors {
  int left, top, top_left, top_right, left_left, top_top, top_right_right;
};
Neighbors neighbors(device const int *plane, uint stride, uint width, uint x, uint y) {
  const ulong row = ulong(y) * stride;
  const int left = x ? plane[row + x - 1] : y ? plane[row - stride + x] : 0;
  const int top = y ? plane[row - stride + x] : left;
  const int right = y && x + 1 < width ? plane[row - stride + x + 1] : top;
  return {left,
          top,
          x && y ? plane[row - stride + x - 1] : left,
          right,
          x > 1 ? plane[row + x - 2] : left,
          y > 1 ? plane[row - 2 * ulong(stride) + x] : top,
          y && x + 2 < width ? plane[row - stride + x + 2] : right};
}
int gradient(Neighbors n) {
  return clamp(n.left + n.top - n.top_left, min(n.left, n.top), max(n.left, n.top));
}
// Unpack/RCT bounds every sample by [-65535,65535]. All neighborhood/property
// arithmetic and even Average4's numerator fit int32. Signed division here is
// truncation toward zero, unlike the floor division inside unpack/RCT.
int predict(uint p, Neighbors n) {
  switch (p) {
  case 0:
    return 0;
  case 1:
    return n.left;
  case 2:
    return n.top;
  case 3:
    return (n.left + n.top) / 2;
  case 4:
    return abs(n.top - n.top_left) < abs(n.left - n.top_left) ? n.left : n.top;
  case 5:
    return gradient(n);
  case 7:
    return n.top_right;
  case 8:
    return n.top_left;
  case 9:
    return n.left_left;
  case 10:
    return (n.left + n.top_left) / 2;
  case 11:
    return (n.top_left + n.top) / 2;
  case 12:
    return (n.top + n.top_right) / 2;
  case 13:
    return (6 * n.top - 2 * n.top_top + 7 * n.left + n.left_left + n.top_right_right +
            3 * n.top_right + 8) /
           16;
  default:
    return 0; // The host rejects weighted/unknown predictors.
  }
}
int property(int p, Neighbors n, uint channel, uint stream, uint x, uint y, device const int *plane,
             uint stride, uint width) {
  switch (p) {
  case 0:
    return int(channel);
  case 1:
    return int(stream);
  case 2:
    return int(y);
  case 3:
    return int(x);
  case 4:
    return int(abs(n.top));
  case 5:
    return int(abs(n.left));
  case 6:
    return n.top;
  case 7:
    return n.left;
  case 8: {
    // The scalar row starts with previous_gradient=0. Recompute the preceding
    // pixel's UNCLAMPED gradient, using the same slice-local substitutions.
    int previous = 0;
    if (x) {
      const Neighbors prior = neighbors(plane, stride, width, x - 1, y);
      previous = prior.left + prior.top - prior.top_left;
    }
    return n.left - previous;
  }
  case 9:
    return n.left + n.top - n.top_left;
  case 10:
    return n.left - n.top_left;
  case 11:
    return n.top_left - n.top;
  case 12:
    return n.top - n.top_right;
  case 13:
    return n.top - n.top_top;
  case 14:
    return n.left - n.left_left;
  default:
    return 0; // The host rejects property 15 and malformed trees.
  }
}
uint pack_signed(int v) { return (uint(v) << 1) ^ ((uint(~v) >> 31) - 1); }

template <bool GradientLeaf, bool Diagnose>
void tokenize(device const int *c0, device const int *c1, device const int *c2,
              device const int *c3, device uchar *output, device uchar *dispatch,
              constant Params &p, uint2 id, device Diagnostic *trace) {
  if (id.y >= p.slice_count)
    return;
  const Slice s = reinterpret_cast<device const Slice *>(dispatch)[id.y];
  if (id.x >= s.width * s.height)
    return; // At most 256*256 samples per slice.
  device const int *base = s.channel == 0 ? c0 : s.channel == 1 ? c1 : s.channel == 2 ? c2 : c3;
  device const int *plane =
      base + p.plane_offset_elements[s.channel] + ulong(s.y) * p.source_width + s.x;
  const uint x = id.x % s.width, y = id.x / s.width;
  const Neighbors n = neighbors(plane, p.source_width, s.width, x, y);
  const int sample = plane[ulong(y) * p.source_width + x];
  int residual, prediction;
  uint context = 0, leaf = 0;
  if (GradientLeaf) {
    prediction = gradient(n);
    residual = sample - prediction; // Bounded by +/-131070; no leaf adjustment.
  } else {
    device const Node *nodes =
        reinterpret_cast<device const Node *>(dispatch + p.nodes_offset_bytes);
    while (nodes[leaf].property != -1) {
      const Node node = nodes[leaf];
      const int value =
          property(node.property, n, s.channel, s.stream_id, x, y, plane, p.source_width, s.width);
      leaf = value > node.split ? node.left : node.right;
    }
    const Node node = nodes[leaf];
    // Full signed offsets can cross int32 even though sample predictions cannot.
    prediction = predict(node.predictor, n);
    const long adjusted = long(sample) - long(prediction) - long(node.offset);
    const long divisor = long(node.multiplier);
    const long quotient = adjusted / divisor;
    if (adjusted % divisor != 0 || quotient < -2147483648L || quotient > 2147483647L) {
      atomic_fetch_or_explicit(
          reinterpret_cast<device atomic_uint *>(dispatch + p.status_offset_bytes),
          uint(kResidualError), memory_order_relaxed);
      return;
    }
    residual = int(quotient);
    context = node.context;
  }
  const ulong token = s.token_offset + id.x;
  reinterpret_cast<device uint *>(output)[token] = pack_signed(residual);
  reinterpret_cast<device ushort *>(output + p.contexts_offset_bytes)[token] = ushort(context);
  if (Diagnose) {
    for (int i = 0; i < 15; ++i)
      trace[token].properties[i] =
          property(i, n, s.channel, s.stream_id, x, y, plane, p.source_width, s.width);
    for (uint i = 0; i < 14; ++i)
      trace[token].predictions[i] = predict(i, n);
    trace[token].leaf = leaf;
    trace[token].selected_prediction = prediction;
    trace[token].sample = sample;
    trace[token].residual = residual;
    trace[token].value = pack_signed(residual);
    trace[token].context = context;
  }
}
} // namespace

kernel void
gjxl_modular_tokens_gradient(device const int *c0 [[buffer(0)]], device const int *c1 [[buffer(1)]],
                             device const int *c2 [[buffer(2)]], device const int *c3 [[buffer(3)]],
                             device uchar *output [[buffer(4)]],
                             device uchar *dispatch [[buffer(5)]], constant Params &p [[buffer(6)]],
                             uint2 id [[thread_position_in_grid]]) {
  tokenize<true, false>(c0, c1, c2, c3, output, dispatch, p, id, nullptr);
}
kernel void
gjxl_modular_tokens_tree(device const int *c0 [[buffer(0)]], device const int *c1 [[buffer(1)]],
                         device const int *c2 [[buffer(2)]], device const int *c3 [[buffer(3)]],
                         device uchar *output [[buffer(4)]], device uchar *dispatch [[buffer(5)]],
                         constant Params &p [[buffer(6)]], uint2 id [[thread_position_in_grid]]) {
  tokenize<false, false>(c0, c1, c2, c3, output, dispatch, p, id, nullptr);
}

kernel void gjxl_modular_tokens_diagnostic(
    device const int *c0 [[buffer(0)]], device const int *c1 [[buffer(1)]],
    device const int *c2 [[buffer(2)]], device const int *c3 [[buffer(3)]],
    device uchar *output [[buffer(4)]], device uchar *dispatch [[buffer(5)]],
    constant Params &p [[buffer(6)]], device Diagnostic *trace [[buffer(7)]],
    uint2 id [[thread_position_in_grid]]) {
  tokenize<false, true>(c0, c1, c2, c3, output, dispatch, p, id, trace);
}
