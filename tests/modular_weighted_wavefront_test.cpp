// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/weighted_features.h"
#include "codec/modular/weighted_feature_math.h"
#include "codec/modular/scalar_prediction.h"
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
namespace wf = gjxl_weighted_features;
void Check(bool v, const char *message) {
  if (!v)
    throw std::runtime_error(message);
}
size_t checked = 0, rejected = 0, cases = 0;
bool Wavefront(ModularChannelView input, const WeightedPredictorParameters &parameters,
               std::span<WeightedPredictorFeature> output, bool rolling) {
  const auto [width, height] = input.descriptor.extent;
  std::vector<wf::State> errors(rolling ? 4 * height : width * height);
  wf::Parameters p{};
  for (size_t i = 0; i < 7; ++i)
    p.coefficients[i] = parameters.coefficients[i];
  for (size_t i = 0; i < 4; ++i)
    p.weights[i] = parameters.weights[i];
  auto address = [&](size_t x, size_t y) {
    return rolling ? ((x + 2 * y) % 4) * height + y : y * width + x;
  };
  const auto state = [&](size_t x, size_t y) { return errors[address(x, y)]; };
  bool valid = true;
  for (size_t step = 0; step < width + 2 * height - 2; ++step)
    for (size_t y = 0; y < height; ++y) {
      if (step < 2 * y)
        continue;
      const size_t x = step - 2 * y;
      if (x >= width)
        continue;
      const auto n = Neighbors(input, x, y);
      const auto result = wf::PredictAndUpdate(
          static_cast<uint32_t>(x), static_cast<uint32_t>(width),
          {n.top, n.left, n.top_right, n.top_left, n.top_top}, input.Row(y)[x], p,
          x ? state(x - 1, y) : wf::State{}, x > 1 ? state(x - 2, y) : wf::State{},
          y ? state(x, y - 1) : wf::State{},
          y ? state(std::min(x + 1, width - 1), y - 1) : wf::State{},
          y ? state(x ? x - 1 : x, y - 1) : wf::State{});
      output[y * width + x] = {result.prediction, result.property15};
      errors[address(x, y)] = result.state;
      valid &= result.valid;
    }
  return valid;
}
void Compare(Extent2D extent, WeightedPredictorParameters parameters, unsigned pattern,
             uint32_t seed) {
  const auto [width, height] = extent;
  const size_t stride = width + 5, origin = stride + 1;
  std::vector<int32_t> backing((height + 2) * stride, INT32_MIN);
  for (size_t y = 0; y < height; ++y)
    for (size_t x = 0; x < width; ++x) {
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;
      int32_t value = 0;
      switch (pattern) {
      case 0:
        value = 17;
        break;
      case 1:
        value = int32_t(x * 41 + y * 23) - 32768;
        break;
      case 2:
        value = seed % 256;
        break;
      case 3:
        value = int32_t(seed % 131071) - 65535;
        break;
      case 4:
        value = int32_t(seed % (1u << 26)) - (1 << 25);
        break;
      case 5:
        value = (x + y) % 2 ? (1 << 27) - 1 : -(1 << 27);
        break;
      case 6:
        value = (x + y) % 2 ? INT32_MIN : INT32_MAX;
        break;
      }
      backing[origin + y * stride + x] = value;
    }
  const ModularChannelView input{std::span(backing).subspan(origin), {extent}, stride};
  std::vector<WeightedPredictorFeature> scalar(width * height), full(scalar.size()),
      rolling(scalar.size());
  const auto status = PrepareWeightedPredictorFeatures(input, parameters, scalar);
  const bool reference = Wavefront(input, parameters, full, false);
  const bool frontier = Wavefront(input, parameters, rolling, true);
  Check((status.ok() || status.code() == StatusCode::kUnsupported) && status.ok() == reference &&
            reference == frontier,
        "Wavefront numerical eligibility differs");
  if (reference) {
    Check(scalar == full && scalar == rolling,
          "Full/rolling wavefront differs from scalar features");
    checked += scalar.size();
  } else
    ++rejected;
  ++cases;
}
void Arithmetic() {
  for (size_t i = 0; i < 64; ++i)
    Check(wf::kReciprocal[i] == kReciprocal[i], "Reciprocal table differs");
  for (const auto [x, width] :
       {std::pair<uint32_t, uint32_t>{0, 1}, {0, 3}, {1, 3}, {2, 3}, {2, 4}, {5, 7}})
    for (uint32_t v : {0u, 1u, 0x7fffffffu, 0x80000000u, 0xffffffffu}) {
      const uint32_t n = v, ne = v - 1, nw = v + 1, w = v ^ 0x80000000u, ww = v + 17;
      const uint64_t north = uint64_t(n) + (x ? w : 0);
      const uint64_t east = x + 1 < width ? ne : north;
      const uint64_t west = x ? uint64_t(nw) + (x > 1 ? ww : 0) : north;
      Check(wf::ErrorSum(x, width, n, ne, nw, w, ww) == uint32_t(north + east + west),
            "Immutable error accumulation lost unsigned wrap or edge duplication");
    }
  wf::Parameters p{{16, 10, 7, 7, 7, 0, 0}, {13, 12, 12, 12}};
  auto r = wf::PredictAndUpdate(1, 3, {0, 0, 0, 0, 0}, 0, p, {-23, {}}, {}, {23, {}}, {23, {}},
                                {-23, {}});
  Check(r.property15 == -23, "Property 15 did not retain west on a magnitude tie");
  r = wf::PredictAndUpdate(0, 3, {0, 0, 0, 0, 0}, 0, p, {}, {}, {23, {}}, {-23, {}}, {23, {}});
  Check(r.property15 == 23, "Property 15 did not retain north on an edge tie");
}
} // namespace
int main() try {
  Arithmetic();
  std::array<WeightedPredictorParameters, 4> parameters{};
  parameters[1] = {{7, 15, 11, 5, 3, 2, 1}, {15, 0, 7, 3}};
  parameters[2] = {{0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0}};
  parameters[3] = {{31, 31, 31, 31, 31, 31, 31}, {15, 15, 15, 15}};
  for (auto extent : {Extent2D{1, 1},
                      {1, 256},
                      {256, 1},
                      {2, 2},
                      {3, 5},
                      {17, 19},
                      {255, 256},
                      {256, 255},
                      {256, 256},
                      {257, 7}})
    for (auto p : parameters)
      for (unsigned pattern = 0; pattern < 7; ++pattern)
        Compare(extent, p, pattern, 0x8a12ec37);
  for (uint32_t seed = 1; seed <= 32; ++seed) {
    WeightedPredictorParameters p;
    for (size_t i = 0; i < 7; ++i)
      p.coefficients[i] = uint8_t((seed * 977 + i * 71) % 32);
    for (size_t i = 0; i < 4; ++i)
      p.weights[i] = uint8_t((seed * 131 + i * 31) % 16);
    Compare({1 + (seed * 13) % 41, 1 + (seed * 29) % 37}, p, 3, seed);
  }
  Check(rejected > 0 && checked > 1000000,
        "Wavefront qualification omitted extremes or common tiles");
  std::cout << cases << " cases: " << checked << " exact scalar/full/rolling features, " << rejected
            << " matching unsupported-range cases\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
