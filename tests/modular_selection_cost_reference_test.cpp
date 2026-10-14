// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/selection_cost.h"
#include "codec/modular/transform/rct.h"
#include "lib/jxl/enc_modular_simd.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/transform/enc_rct.h"
#include "lib/jxl/modular/transform/transform.h"
#include "modular_wp_cost_reference.h"
#include <algorithm>
#include <array>
#include <bit>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
constexpr std::array<uint8_t, 19> kRcts{0, 6,  5,  10, 26, 40, 12, 19, 8, 4,
                                        9, 15, 16, 17, 32, 33, 2,  1,  3};
size_t cases = 0, estimates = 0, rankings = 0;
size_t wp_estimates = 0, wp_rankings = 0, wp_overflows = 0, wp_wrapped = 0, wp_finite_wrapped = 0;
void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Ok(Status s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
uint32_t Step(uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}
float Compare(std::span<const ModularChannelView> views, const jxl::Image &reference) {
  ModularTransformCost cost;
  ArmNextManagedHostAllocationFailureForTest();
  const auto status = EstimateModularTransformCost(views, &cost);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Ok(status);
  Check(pending, "Cost estimator allocated");
  auto result = jxl::EstimateCost(reference);
  Check(result.ok(), "Pinned estimator failed");
  const float expected = std::move(result).value_();
  if (std::bit_cast<uint32_t>(cost.bits) != std::bit_cast<uint32_t>(expected)) {
    std::cerr << "native " << cost.bits << " reference " << expected << " extra " << cost.extra_bits
              << " histogram " << cost.histogram_bits << " fraction " << cost.histogram_fraction
              << '\n';
    throw std::runtime_error("Pinned SIMD transform cost differs");
  }
  ++estimates;
  return expected;
}
size_t CompareHistory(const ModularChannelView &view, const jxl::weighted::Header &header,
                      size_t mode) {
  const auto [w, h] = view.descriptor.extent;
  WeightedPredictor<ResourceClass::kPreparation> native(w, kSelectionWeightedPresets[mode]);
  jxl::weighted::State reference(header, w, h);
  jxl::Properties property(1);
  size_t wrapped = 0;
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x) {
      const int64_t left = x ? view.Row(y)[x - 1] : y ? view.Row(y - 1)[0] : 0;
      const int64_t north = y ? view.Row(y - 1)[x] : left;
      const int64_t northwest = x && y ? view.Row(y - 1)[x - 1] : left;
      const int64_t northeast = y && x + 1 < w ? view.Row(y - 1)[x + 1] : north;
      const int64_t northnorth = y > 1 ? view.Row(y - 2)[x] : north;
      const auto actual = native.Predict(x, y, north, left, northeast, northwest, northnorth);
      const auto expected = reference.Predict<true>(x, y, w, north, left, northeast, northwest,
                                                    northnorth, &property, 0);
      Check(actual.first == expected && actual.second == property[0],
            "Wrapped WP prediction/property differs");
      // Checked updates remain supported for coding/training. Where they reject,
      // use the explicit reference narrowing and compare the following pixels.
      if (!native.Update(view.Row(y)[x], x, y)) {
        native.UpdateWrapping(view.Row(y)[x], x, y);
        ++wrapped;
      }
      reference.UpdateErrors(view.Row(y)[x], x, y, w);
    }
  return wrapped;
}
void CompareWeighted(std::span<const ModularChannelView> views, const jxl::Image &reference) {
  std::array<float, 5> actual{}, expected{};
  for (size_t mode = 0; mode < 5; ++mode) {
    jxl::weighted::Header header;
    jxl::weighted::PredictorMode(mode, &header);
    const auto &preset = kSelectionWeightedPresets[mode];
    Check(std::ranges::equal(preset.coefficients,
                             std::array{header.p1C, header.p2C, header.p3Ca, header.p3Cb,
                                        header.p3Cc, header.p3Cd, header.p3Ce}) &&
              std::ranges::equal(preset.weights, header.w),
          "Native WP preset differs");
    size_t wrapped = 0;
    // Cost comparisons cover the larger images; direct state comparisons also
    // exercise every pixel of all small full-depth/edge fixtures.
    if (reference.channel[0].w * reference.channel[0].h < 1024)
      for (const auto &view : views)
        wrapped += CompareHistory(view, header, mode);
    wp_wrapped += wrapped;
    HostStorageBound bound;
    ArmNextManagedHostAllocationFailureForTest();
    const auto planned = ComputeModularWeightedCostStorageBound(views, mode, &bound);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(planned.ok() && pending, "WP cost planner failed or allocated");
    ResourceBudget budget(bound.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.peak_bytes, &reservation));
    ModularWeightedCost cost;
    {
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      Ok(EstimateModularWeightedCost(views, mode, &cost));
    }
    Check(budget.snapshot().peak_backing_bytes <= bound.peak_bytes &&
              budget.snapshot().total.backing_count == 0 &&
              budget.snapshot().total.pending_count == 0,
          "WP cost exceeded its bound or retained history");
    actual[mode] = cost.bits;
    expected[mode] = jxl::gjxl_cost_reference::EstimateWPCost(reference, mode);
    if (std::bit_cast<uint32_t>(actual[mode]) != std::bit_cast<uint32_t>(expected[mode])) {
      std::cerr << "WP mode " << mode << " native " << actual[mode] << " reference "
                << expected[mode] << '\n';
      throw std::runtime_error("Pinned weighted cost differs");
    }
    Check(cost.residual_overflow == (expected[mode] == std::numeric_limits<float>::max()),
          "WP residual-overflow rejection differs");
    wp_overflows += cost.residual_overflow;
    wp_finite_wrapped += wrapped && !cost.residual_overflow;
    ++wp_estimates;
  }
  std::array<size_t, 5> a{0, 1, 2, 3, 4}, b = a;
  std::stable_sort(a.begin(), a.end(), [&](size_t x, size_t y) { return actual[x] < actual[y]; });
  std::stable_sort(b.begin(), b.end(),
                   [&](size_t x, size_t y) { return expected[x] < expected[y]; });
  Check(a == b, "WP ranking/tie order differs");
  ++wp_rankings;
}
void Case(size_t w, size_t h, size_t channels, unsigned pattern, bool rank) {
  JxlMemoryManager memory;
  Check(bool(jxl::MemoryManagerInit(&memory, nullptr)), "Reference allocator setup failed");
  auto result = jxl::Image::Create(&memory, w, h, 16, channels);
  Check(result.ok(), "Reference image allocation failed");
  auto reference = std::move(result).value_();
  const size_t stride = w + 3;
  std::vector<std::vector<int32_t>> samples(channels, std::vector<int32_t>((h - 1) * stride + w));
  std::vector<ModularChannelView> views;
  uint32_t random = 0x183567ab;
  const std::array<int32_t, 24> edge{INT32_MIN, INT32_MAX,
                                     -1,        0,
                                     1,         7,
                                     8,         -8,
                                     -9,        15,
                                     16,        31,
                                     32,        127,
                                     128,       255,
                                     256,       65535,
                                     65536,     (1 << 22) - 1,
                                     1 << 22,   (1 << 30) - 1,
                                     1 << 30,   -(1 << 30)};
  for (size_t c = 0; c < channels; ++c) {
    views.push_back({samples[c], {{w, h}}, stride});
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x) {
        const uint32_t r = Step(random);
        int32_t value;
        switch (pattern) {
        case 0:
          value = 0;
          break;
        case 1:
          value = (x + y * 7 + c * 31) & 255;
          break;
        case 2:
          value = r & 65535;
          break;
        case 3:
          value = (r % 100 < 95) ? 5 : r & 255;
          break;
        case 4:
          value = edge[r % edge.size()];
          break;
        case 5:
          value = std::bit_cast<int32_t>(r);
          break;
        default:
          value = int32_t(r & ((1 << 24) - 1)) - (1 << 23);
          break;
        }
        samples[c][y * stride + x] = reference.channel[c].Row(y)[x] = value;
      }
  }
  Compare(views, reference);
  CompareWeighted(views, reference);
  if (rank) {
    std::array<float, 19> native_costs{}, reference_costs{};
    for (size_t i = 0; i < kRcts.size(); ++i) {
      const uint8_t type = kRcts[i];
      auto copy_result = jxl::Image::Create(&memory, w, h, 16, channels);
      Check(copy_result.ok(), "Reference candidate allocation failed");
      auto candidate = std::move(copy_result).value_();
      auto native_pixels = samples;
      auto candidate_views = views;
      for (size_t c = 0; c < channels; ++c) {
        candidate_views[c].backing = native_pixels[c];
        for (size_t y = 0; y < h; ++y)
          std::copy_n(reference.channel[c].Row(y), w, candidate.channel[c].Row(y));
      }
      if (type)
        Check(bool(jxl::FwdRct(candidate, 0, type, nullptr)), "Pinned RCT failed");
      for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) {
          const std::array<int32_t, 3> pixel{samples[0][y * stride + x], samples[1][y * stride + x],
                                             samples[2][y * stride + x]};
          const auto transformed = ForwardRctWide(pixel, type);
          for (size_t c = 0; c < 3; ++c) {
            Check(transformed[c] >= INT32_MIN && transformed[c] <= INT32_MAX,
                  "RCT fixture exceeds the native transform profile");
            native_pixels[c][y * stride + x] = static_cast<int32_t>(transformed[c]);
            Check(transformed[c] == candidate.channel[c].Row(y)[x],
                  "Native/reference candidate differs");
          }
        }
      reference_costs[i] = Compare(candidate_views, candidate);
      ModularTransformCost cost;
      Ok(EstimateModularTransformCost(candidate_views, &cost));
      native_costs[i] = cost.bits;
    }
    std::array<size_t, 19> a{}, b{};
    for (size_t i = 0; i < a.size(); ++i)
      a[i] = b[i] = i;
    std::stable_sort(a.begin(), a.end(),
                     [&](size_t x, size_t y) { return native_costs[x] < native_costs[y]; });
    std::stable_sort(b.begin(), b.end(),
                     [&](size_t x, size_t y) { return reference_costs[x] < reference_costs[y]; });
    Check(a == b, "RCT costs changed ranking/ties");
    ++rankings;
  }
  ++cases;
}
} // namespace
int main() try {
  for (size_t w : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33})
    for (size_t h : {1, 2, 5, 17})
      for (unsigned pattern = 0; pattern < 7; ++pattern)
        Case(w, h, 1 + (w % 4), pattern, w % 4 >= 2 && pattern != 4 && pattern != 5);
  for (unsigned pattern : {1, 2, 3, 6})
    Case(1025, 769, 4, pattern, true);
  std::cout << "Transform-cost reference: " << cases << " cases, " << estimates
            << " exact estimates, " << rankings << " exact RCT rankings\n";
  Check(wp_overflows && wp_wrapped && wp_finite_wrapped, "WP overflow controls were not exercised");
  std::cout << "Weighted-cost reference: " << wp_estimates << " exact estimates, " << wp_rankings
            << " exact rankings, " << wp_overflows << " rejected candidates, " << wp_wrapped
            << " wrapped history updates, " << wp_finite_wrapped << " finite wrapped candidates\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
