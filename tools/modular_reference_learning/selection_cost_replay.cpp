// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "selection_cost_replay.h"
#include "codec/modular/selection_cost.h"
#include "codec/modular/transform/rct.h"
#include "lib/jxl/enc_modular_simd.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/transform/enc_rct.h"
#include "lib/jxl/modular/transform/transform.h"
#include "modular_wp_cost_reference.h"
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace reference_learning {
namespace {
namespace native = gjxl::modular_internal;
void Require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(gjxl::Status s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
constexpr std::array<uint8_t, 19> kRcts{0, 6,  5,  10, 26, 40, 12, 19, 8, 4,
                                        9, 15, 16, 17, 32, 33, 2,  1,  3};
float Compare(std::span<const native::ModularChannelView> views, const jxl::Image &reference,
              std::ostream &out) {
  using namespace gjxl::resource_budget_internal;
  native::ModularTransformCost cost;
  ArmNextManagedHostAllocationFailureForTest();
  const auto status = native::EstimateModularTransformCost(views, &cost);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Ok(status);
  Require(pending, "Native transform estimator allocated");
  auto result = jxl::EstimateCost(reference);
  Require(result.ok(), "Reference transform cost failed");
  const float expected = std::move(result).value_();
  Require(std::bit_cast<uint32_t>(cost.bits) == std::bit_cast<uint32_t>(expected),
          "Native/reference captured transform costs differ");
  out << "{\"native_bits\":" << cost.bits << ",\"reference_bits\":" << expected
      << ",\"histogram_bits\":" << cost.histogram_bits
      << ",\"histogram_fraction\":" << cost.histogram_fraction
      << ",\"extra_bits\":" << cost.extra_bits << ",\"pixels\":" << cost.pixels << '}';
  return cost.bits;
}
jxl::Image ReferenceImage(std::span<const native::ModularChannelView> views, size_t bit_depth,
                          JxlMemoryManager *memory) {
  jxl::Image reference(memory);
  reference.bitdepth = bit_depth;
  for (const auto &view : views) {
    const auto &d = view.descriptor;
    const bool meta = d.role == native::ChannelRole::kMetadata;
    reference.nb_meta_channels += meta;
    auto created = jxl::Channel::Create(memory, d.extent.width, d.extent.height,
                                        meta ? -1 : d.hshift, meta ? -1 : d.vshift);
    Require(created.ok(), "Reference cost channel allocation failed");
    auto channel = std::move(created).value_();
    for (size_t y = 0; y < d.extent.height; ++y)
      std::copy_n(view.Row(y), d.extent.width, channel.Row(y));
    reference.channel.push_back(std::move(channel));
  }
  return reference;
}
} // namespace
void ReplaySelectionCosts(const native::PreparedModularStreamsView &input,
                          const native::ModularStreamPlan &plan, size_t bit_depth,
                          std::ostream &out) {
  JxlMemoryManager memory;
  Require(bool(jxl::MemoryManagerInit(&memory, nullptr)), "Reference allocation setup failed");
  out << "{\"schema\":1,\"costs_exact\":true,\"managed_allocations\":0,\"timing_qualified\":false,"
         "\"streams\":[";
  for (size_t i = 0; i < input.streams.size(); ++i) {
    const auto views = input.streams[i].channels;
    auto reference = ReferenceImage(views, bit_depth, &memory);
    const size_t metadata = reference.nb_meta_channels;
    bool eligible = views.size() >= metadata + 3;
    if (eligible) {
      const auto d = views[metadata].descriptor;
      for (size_t c = 1; c < 3; ++c) {
        const auto other = views[metadata + c].descriptor;
        eligible &=
            d.extent == other.extent && d.hshift == other.hshift && d.vshift == other.vshift;
      }
    }
    out << (i ? "," : "") << "{\"id\":" << plan.streams[i].id << ",\"channels\":" << views.size()
        << ",\"metadata_channels\":" << metadata << ",\"base\":";
    const float base = Compare(views, reference, out);
    out << ",\"rct_candidates\":[";
    size_t best_rct = 0;
    float best_cost = base;
    if (eligible) {
      auto cloned = jxl::Image::Clone(reference);
      Require(cloned.ok(), "Reference candidate copy failed");
      auto candidate = std::move(cloned).value_();
      const auto [w, h] = views[metadata].descriptor.extent;
      std::array<std::vector<int32_t>, 3> transformed;
      std::vector<native::ModularChannelView> transformed_views(views.begin(), views.end());
      const std::array<const jxl::Channel *, 3> source = {&reference.channel[metadata],
                                                          &reference.channel[metadata + 1],
                                                          &reference.channel[metadata + 2]};
      const std::array<jxl::Channel *, 3> target = {&candidate.channel[metadata],
                                                    &candidate.channel[metadata + 1],
                                                    &candidate.channel[metadata + 2]};
      for (size_t c = 0; c < 3; ++c) {
        transformed[c].resize(w * h);
        transformed_views[metadata + c].backing = transformed[c];
        transformed_views[metadata + c].stride = w;
      }
      for (size_t k = 0; k < kRcts.size(); ++k) {
        const auto rct = kRcts[k];
        Require(bool(jxl::FwdRct(source, target, rct, nullptr)), "Pinned candidate RCT failed");
        for (size_t y = 0; y < h; ++y)
          for (size_t x = 0; x < w; ++x) {
            const std::array<int32_t, 3> pixel = {views[metadata].Row(y)[x],
                                                  views[metadata + 1].Row(y)[x],
                                                  views[metadata + 2].Row(y)[x]};
            const auto values = native::ForwardRctWide(pixel, rct);
            for (size_t c = 0; c < 3; ++c) {
              Require(values[c] >= INT32_MIN && values[c] <= INT32_MAX &&
                          values[c] == target[c]->Row(y)[x],
                      "Native/reference candidate pixels differ");
              transformed[c][y * w + x] = values[c];
            }
          }
        out << (k ? "," : "") << "{\"rct\":" << unsigned(rct) << ",\"cost\":";
        const float bits = Compare(transformed_views, candidate, out);
        out << '}';
        if (bits < best_cost) {
          best_cost = bits;
          best_rct = rct;
        }
      }
    }
    out << "],\"selected_rct\":" << best_rct << '}';
  }
  out << "]}\n";
}
void ReplayWeightedCosts(const native::PreparedModularStreamsView &input,
                         const native::ModularStreamPlan &plan, size_t bit_depth,
                         std::ostream &out) {
  using namespace gjxl::resource_budget_internal;
  JxlMemoryManager memory;
  Require(bool(jxl::MemoryManagerInit(&memory, nullptr)), "Reference allocation setup failed");
  out << "{\"schema\":1,\"costs_exact\":true,\"managed_storage_qualified\":true,"
         "\"timing_qualified\":false,\"streams\":[";
  for (size_t i = 0; i < input.streams.size(); ++i) {
    const auto views = input.streams[i].channels;
    auto reference = ReferenceImage(views, bit_depth, &memory);
    out << (i ? "," : "") << "{\"id\":" << plan.streams[i].id << ",\"channels\":" << views.size()
        << ",\"metadata_channels\":" << reference.nb_meta_channels << ",\"candidates\":[";
    float best = std::numeric_limits<float>::max();
    size_t best_mode = 0;
    for (size_t mode = 0; mode < 5; ++mode) {
      HostStorageBound bound;
      ArmNextManagedHostAllocationFailureForTest();
      const auto planned = native::ComputeModularWeightedCostStorageBound(views, mode, &bound);
      const bool pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      Ok(planned);
      Require(pending, "Native WP cost planner allocated");
      ResourceBudget budget(bound.peak_bytes);
      ResourceReservation reservation;
      native::ModularWeightedCost cost;
      if (bound.peak_bytes) {
        Ok(budget.Reserve(bound.peak_bytes, &reservation));
        ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
        Ok(native::EstimateModularWeightedCost(views, mode, &cost));
      } else {
        ArmNextManagedHostAllocationFailureForTest();
        const auto status = native::EstimateModularWeightedCost(views, mode, &cost);
        const bool unallocated = ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        Ok(status);
        Require(unallocated, "Empty WP estimate allocated");
      }
      const auto usage = budget.snapshot();
      Require(usage.peak_backing_bytes <= bound.peak_bytes && usage.total.backing_count == 0 &&
                  usage.total.pending_count == 0 && bound.retained_bytes == 0,
              "Native WP estimator exceeded its plan or retained history");
      const float expected = jxl::gjxl_cost_reference::EstimateWPCost(reference, mode);
      Require(std::bit_cast<uint32_t>(cost.bits) == std::bit_cast<uint32_t>(expected),
              "Native/reference captured WP costs differ");
      Require(cost.residual_overflow == (expected == std::numeric_limits<float>::max()),
              "Native/reference WP residual rejection differs");
      out << (mode ? "," : "") << "{\"mode\":" << mode << ",\"native_bits\":" << cost.bits
          << ",\"reference_bits\":" << expected << ",\"histogram_bits\":" << cost.histogram_bits
          << ",\"extra_bits\":" << cost.extra_bits << ",\"pixels\":" << cost.pixels
          << ",\"residual_overflow\":" << (cost.residual_overflow ? "true" : "false")
          << ",\"storage_bound_bytes\":" << bound.peak_bytes
          << ",\"managed_peak_bytes\":" << usage.peak_backing_bytes << ",\"retained_bytes\":0}";
      if (cost.bits < best) {
        best = cost.bits;
        best_mode = mode;
      }
    }
    out << "],\"selected_mode\":" << best_mode << '}';
  }
  out << "]}\n";
}
} // namespace reference_learning
