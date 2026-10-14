// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Exact weighted-predictor arithmetic adapted from context_predict.h, with
// immutable errors instead of the scalar implementation's mutable row sums.
#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <bit>
#include <cstdint>
#endif

namespace gjxl_weighted_features {
#ifdef __METAL_VERSION__
using Word = uint;
using Signed = int;
using Wide = long;
#else
using Word = uint32_t;
using Signed = int32_t;
using Wide = int64_t;
#endif
struct Parameters {
  Word coefficients[7];
  Word weights[4];
};
// State is the error produced at this pixel, before any neighbor accumulation.
struct State {
  Signed error;
  Word suberror[4];
};
struct Neighborhood {
  Wide n, w, ne, nw, nn;
};
struct Result {
  Wide prediction, property15;
  State state;
  bool valid;
};
#ifdef __METAL_VERSION__
constant Word kReciprocal[64] = {
#else
inline constexpr Word kReciprocal[64] = {
#endif
    16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152, 1864135, 1677721,
    1525201,  1398101, 1290555, 1198372, 1118481, 1048576, 986895,  932067,  883011,  838860,
    798915,   762600,  729444,  699050,  671088,  645277,  621378,  599186,  578524,  559240,
    541200,   524288,  508400,  493447,  479349,  466033,  453438,  441505,  430185,  419430,
    409200,   399457,  390167,  381300,  372827,  364722,  356962,  349525,  342392,  335544,
    328965,   322638,  316551,  310689,  305040,  299593,  294337,  289262,  284359,  279620,
    275036,   270600,  266305,  262144};
inline Wide Abs(Wide v) { return v < 0 ? -v : v; }
inline Wide Min(Wide a, Wide b) { return a < b ? a : b; }
inline Wide Max(Wide a, Wide b) { return a > b ? a : b; }
inline Word BitWidth(Word v) {
#ifdef __METAL_VERSION__
  return 32 - metal::clz(v);
#else
  return std::bit_width(v);
#endif
}
// Update(x-1,y) added A(W) to the north slot, and Update(x-2,y)
// added A(WW) to northwest. NE is not yet modified except at the right
// edge, where the scalar implementation aliases it to north. All additions
// below deliberately wrap at 32 bits, including duplicated edge slots.
inline Word ErrorSum(Word x, Word width, Word n, Word ne, Word nw, Word w, Word ww) {
  const Word north = n + (x ? w : 0);
  const Word northeast = x + 1 < width ? ne : north;
  const Word northwest = x ? nw + (x > 1 ? ww : 0) : north;
  return north + northeast + northwest;
}

// The caller supplies zero states for missing W/WW or the first row, and
// clamps N/NE/NW coordinates exactly as the scalar row indices do. This
// function has no traversal state, allocation or synchronization.
inline Result PredictAndUpdate(Word x, Word width, Neighborhood pixels, Signed sample,
                               Parameters parameters, State west, State westwest, State north,
                               State northeast, State northwest) {
  Word weights[4];
  Word weight_sum = 0;
  for (Word i = 0; i < 4; ++i) {
    const Word error = ErrorSum(x, width, north.suberror[i], northeast.suberror[i],
                                northwest.suberror[i], west.suberror[i], westwest.suberror[i]);
    // Equivalent to bit_width(uint64_t(error) + 1), including UINT32_MAX.
    const Word bits = error == 0xffffffffu ? 33 : BitWidth(error + 1);
    const Word shift = bits > 6 ? bits - 6 : 0;
    weights[i] = 4 + ((parameters.weights[i] * kReciprocal[error >> shift]) >> shift);
    weight_sum += weights[i];
  }
  const Word log_weight = BitWidth(weight_sum) - 1;
  weight_sum = 0;
  for (Word i = 0; i < 4; ++i) {
    weights[i] >>= log_weight - 4;
    weight_sum += weights[i];
  }
  const Wide ew = x ? west.error : 0;
  const Wide en = north.error, ene = northeast.error, enw = northwest.error;
  Wide property = ew;
  if (Abs(en) > Abs(property))
    property = en;
  if (Abs(enw) > Abs(property))
    property = enw;
  if (Abs(ene) > Abs(property))
    property = ene;
  pixels.n *= 8;
  pixels.w *= 8;
  pixels.ne *= 8;
  pixels.nw *= 8;
  pixels.nn *= 8;
  Wide predictions[4];
  predictions[0] = pixels.w + pixels.ne - pixels.n;
  predictions[1] = pixels.n - (((en + ew + ene) * parameters.coefficients[0]) >> 5);
  predictions[2] = pixels.w - (((en + ew + enw) * parameters.coefficients[1]) >> 5);
  predictions[3] = pixels.n - ((enw * parameters.coefficients[2] + en * parameters.coefficients[3] +
                                ene * parameters.coefficients[4] +
                                (pixels.nn - pixels.n) * parameters.coefficients[5] +
                                (pixels.nw - pixels.w) * parameters.coefficients[6]) >>
                               5);
  Wide sum = (weight_sum >> 1) - 1;
  for (Word i = 0; i < 4; ++i)
    sum += predictions[i] * weights[i];
  Wide prediction = (sum * kReciprocal[weight_sum - 1]) >> 24;
  if (((en ^ ew) | (en ^ enw)) <= 0)
    prediction = Max(Min(pixels.w, Min(pixels.ne, pixels.n)),
                     Min(prediction, Max(pixels.w, Max(pixels.ne, pixels.n))));
  Result result{(prediction + 3) >> 3, property, {}, true};
  const Wide scaled = Wide(sample) * 8, error = prediction - scaled;
  if (error < -2147483647 - Wide(1) || error > 2147483647) {
    result.valid = false;
    return result;
  }
  State next{};
  next.error = Signed(error);
  for (Word i = 0; i < 4; ++i) {
    const Wide magnitude = (Abs(predictions[i] - scaled) + 3) >> 3;
    if (magnitude > 0xffffffffu) {
      result.valid = false;
      return result;
    }
    next.suberror[i] = Word(magnitude);
  }
  result.state = next;
  return result;
}
} // namespace gjxl_weighted_features
