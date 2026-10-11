// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// General transformed-plane arithmetic follows the pinned scalar predictor.
#include "modular_adaptive_params.h"
#include <metal_stdlib>
using namespace metal;
using namespace gjxl_modular_adaptive;
namespace {
struct Neighborhood {
  long left, top, top_left, top_right, left_left, top_top, top_right_right;
};
Neighborhood neighbors(device const int *plane, ulong stride, uint width, uint x, uint y) {
  const ulong row = ulong(y) * stride;
  const long left = x ? plane[row + x - 1] : y ? plane[row - stride + x] : 0;
  const long top = y ? plane[row - stride + x] : left;
  const long right = y && x + 1 < width ? plane[row - stride + x + 1] : top;
  return {left,
          top,
          x && y ? plane[row - stride + x - 1] : left,
          right,
          x > 1 ? plane[row + x - 2] : left,
          y > 1 ? plane[row - 2 * stride + x] : top,
          y && x + 2 < width ? plane[row - stride + x + 2] : right};
}
long predict(uint p, Neighborhood n, long weighted) {
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
    return clamp(n.left + n.top - n.top_left, min(n.left, n.top), max(n.left, n.top));
  case 6:
    return weighted;
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
    return 0; // Host validation rejects unknown predictors.
  }
}
long property(int p, Neighborhood n, uint channel, uint stream, uint x, uint y,
              device const int *plane, ulong stride, uint width, long weighted) {
  switch (p) {
  case 0:
    return long(channel);
  case 1:
    return long(stream);
  case 2:
    return long(y);
  case 3:
    return long(x);
  case 4:
    return long(abs(n.top));
  case 5:
    return long(abs(n.left));
  case 6:
    return n.top;
  case 7:
    return n.left;
  case 8: {
    long previous = 0;
    if (x) {
      const auto prior = neighbors(plane, stride, width, x - 1, y);
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
  case 15:
    return weighted;
  default:
    return 0; // Host validation rejects unknown properties.
  }
}
template <bool Diagnose>
void tokenize(device const int *planes, device const WeightedFeature *features,
              device uchar *output, device uchar *dispatch, constant Params &p, uint2 id,
              device Diagnostic *trace) {
  if (id.y >= p.slice_count)
    return;
  const Slice s = reinterpret_cast<device const Slice *>(dispatch)[id.y];
  if (id.x >= ulong(s.width) * s.height)
    return;
  const ulong token = s.token_offset + id.x;
  const uint x = id.x % s.width, y = id.x / s.width;
  device const int *plane = planes + s.plane_offset_elements;
  const Neighborhood n = neighbors(plane, s.plane_stride_elements, s.width, x, y);
  const WeightedFeature weighted = p.has_weighted ? features[token] : WeightedFeature{0, 0};
  device const Node *nodes = reinterpret_cast<device const Node *>(dispatch + p.nodes_offset_bytes);
  uint leaf = 0;
  while (nodes[leaf].property != -1) {
    const Node node = nodes[leaf];
    const long value = property(node.property, n, s.property_channel, s.stream_id, x, y, plane,
                                s.plane_stride_elements, s.width, weighted.property15);
    leaf = value > node.split ? node.left : node.right;
  }
  const Node node = nodes[leaf];
  const int sample = plane[ulong(y) * s.plane_stride_elements + x];
  const long prediction = predict(node.predictor, n, weighted.prediction);
  const long adjusted = long(sample) - prediction - long(node.offset);
  const long quotient = adjusted / long(node.multiplier);
  if (adjusted % long(node.multiplier) || quotient < -2147483648L || quotient > 2147483647L) {
    atomic_fetch_or_explicit(
        reinterpret_cast<device atomic_uint *>(dispatch + p.status_offset_bytes),
        uint(gjxl_modular_tokens::kResidualError), memory_order_relaxed);
    return;
  }
  const int residual = int(quotient);
  const uint value = (uint(residual) << 1) ^ ((uint(~residual) >> 31) - 1);
  reinterpret_cast<device uint *>(output)[token] = value;
  reinterpret_cast<device ushort *>(output + p.contexts_offset_bytes)[token] = ushort(node.context);
  if (Diagnose) {
    for (int i = 0; i < 16; ++i)
      trace[token].properties[i] = property(i, n, s.property_channel, s.stream_id, x, y, plane,
                                            s.plane_stride_elements, s.width, weighted.property15);
    for (uint i = 0; i < 14; ++i)
      trace[token].predictions[i] = predict(i, n, weighted.prediction);
    trace[token].selected_prediction = prediction;
    trace[token].leaf = leaf;
    trace[token].context = node.context;
    trace[token].sample = sample;
    trace[token].residual = residual;
    trace[token].value = value;
    trace[token].reserved = 0;
  }
}
} // namespace
kernel void gjxl_modular_adaptive_tokens(device const int *planes [[buffer(0)]],
                                         device const WeightedFeature *features [[buffer(1)]],
                                         device uchar *output [[buffer(2)]],
                                         device uchar *dispatch [[buffer(3)]],
                                         constant Params &p [[buffer(4)]],
                                         uint2 id [[thread_position_in_grid]]) {
  tokenize<false>(planes, features, output, dispatch, p, id, nullptr);
}
kernel void gjxl_modular_adaptive_diagnostic(device const int *planes [[buffer(0)]],
                                             device const WeightedFeature *features [[buffer(1)]],
                                             device uchar *output [[buffer(2)]],
                                             device uchar *dispatch [[buffer(3)]],
                                             constant Params &p [[buffer(4)]],
                                             device Diagnostic *trace [[buffer(5)]],
                                             uint2 id [[thread_position_in_grid]]) {
  tokenize<true>(planes, features, output, dispatch, p, id, trace);
}
