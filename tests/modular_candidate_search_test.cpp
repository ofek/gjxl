// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/workflow.h"
#include "codestream/modular/search.h"
#include "codestream/modular/adaptive_gpu_workflow.h"
#include "codec/modular/profile.h"
#ifdef GJXL_CANDIDATE_COST_REFERENCE
#include "modular_reference.h"
#endif
#ifdef GJXL_CANDIDATE_SEARCH_METAL
#include "gpu/metal/metal_backend.h"
#endif
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
void Check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
void Ok(const Status &s) { if (!s.ok()) throw std::runtime_error(std::string(s.message())); }
struct PolicyKey {
  uint8_t rct;
  TransformSequence transforms;
  WeightedPredictorParameters weighted;
  std::vector<DecisionNode> nodes;
  bool operator==(const PolicyKey &) const = default;
};
struct Ledger final : ModularCandidateObserver {
  std::vector<ModularCandidateRecord> records;
  std::vector<PolicyKey> policies;
  size_t fail_at = SIZE_MAX;
  Status Candidate(const ModularCodingPolicy &p, const ModularCandidateRecord &record) override {
    if (record.index == fail_at) return Status::Internal("Injected ledger failure");
    records.push_back(record);
    // Plain diagnostic values do not retain domain-owned DecisionTree storage.
    policies.push_back({p.rct, p.transforms, p.weighted,
                        {p.tree.nodes.begin(), p.tree.nodes.begin() + p.tree.size}});
    return Status::Ok();
  }
};
void Empty(const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  Check(!s.live_capacity_bytes && !s.reserved_unbacked_bytes && !s.active_reservations &&
        !s.active_cpu_participants, "Candidate search leaked admitted storage/participation");
}
void Fixture(PackedModularFormat format, EntropyCodingMode entropy, AnsSelectionMode selector,
             GpuBackend *gpu = nullptr, bool constant = false) {
  const Extent2D extent{33, 17};
  ModularInputProfile description;
  Ok(ResolveModularInput(extent, format, &description));
  const size_t stride = extent.width * description.channel_count * (description.metadata.bits / 8);
  std::vector<uint8_t> pixels(stride * extent.height);
  for (size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = constant ? 0 : uint8_t(i * 13 + (i / 7) % 43);
  PackedModularImageView source{pixels, extent, stride, format, SampleByteOrder::kBigEndian};
  ModularEncodingOptions options;
  options.search = true;
  options.entropy = entropy;
  options.ans_selection = selector;
  options.learning = {1024, 32};
  options.tree_checkpoints = true;
  options.cpu_thread_count = 2;
  options.adaptive_gpu = gpu;
  options.adaptive_ans_cost = gpu != nullptr;
  options.adaptive_tree_statistics = gpu != nullptr;
  const bool default_shared_features = options.adaptive_shared_features;
  Check(default_shared_features, "Adaptive GPU search should share learning features by default");
  options.adaptive_shared_features = false;
  Ledger oracle, measured;
  options.candidate_observer = &oracle;
  ModularProfile reference, result;
  std::vector<uint8_t> full, sized, unprofiled;
  { ProfileSession profile(&reference); Ok(EncodeModularImage(source, options, &full)); }
  // The measured/shared path must preserve every candidate, cost and tie from
  // the unshared complete-emission oracle, including the final winner re-encode.
  options.adaptive_shared_features = default_shared_features;
  options.search_evaluation = ModularSearchEvaluation::kMeasureAll;
  options.candidate_observer = &measured;
  size_t capacity;
  resource_budget_internal::ArmNextManagedHostAllocationFailureForTest();
  if (gpu) {
    AdaptiveModularWorkflowStoragePlan plan;
    Ok(ComputeAdaptiveModularWorkflowStoragePlan(extent, format, options, &plan));
    capacity = plan.peak_bytes;
  } else {
    ModularWorkflowStoragePlan plan;
    Ok(ComputeModularSearchStoragePlan(extent, format, entropy, &plan, 2, options.learning, true, true));
    capacity = plan.working.peak_bytes;
  }
  Check(resource_budget_internal::ManagedHostAllocationFailurePendingForTest(), "Sizing planner allocated");
  resource_budget_internal::DisarmManagedHostAllocationFailureForTest();
  Ok(ExecutionDomain::Create({capacity, 2}, &options.execution_domain));
  { ProfileSession profile(&result); Ok(EncodeModularImage(source, options, &sized)); }
  Check(full == sized && reference.resolved == result.resolved,
        "Exact candidate sizing changed winning policy or bytes");
  Check(oracle.policies == measured.policies && oracle.records.size() == measured.records.size(),
        "Exact candidate sizing changed candidate order/policies");
  for (size_t i = 0; i < oracle.records.size(); ++i) {
    const auto &a = oracle.records[i], &b = measured.records[i];
    Check(a.index == i && b.index == i && a.checkpoint == b.checkpoint && a.improved == b.improved &&
          a.cost == b.cost, "Per-candidate exact cost, checkpoint priority or tie differs from emission");
  }
  Check(reference.encoded_candidates == result.encoded_candidates &&
        reference.candidate_emissions == reference.encoded_candidates && result.candidate_emissions == 1,
        "Exhaustive measurement did not emit only the winner");
  Check(!gpu || result.adaptive_gpu.encoded_candidates + result.adaptive_gpu.cpu_fallback_candidates ==
                    result.encoded_candidates + 1, "Final winner token request is missing from GPU work");
  result.resolved = {};
  Empty(*options.execution_domain);
  Check(options.execution_domain->snapshot().peak_backing_bytes <= capacity, "Sizing exceeded admission");
  options.candidate_observer = nullptr;
  Ok(EncodeModularImage(source, options, &unprofiled));
  Check(unprofiled == full, "Disabled profiling changed measured winner");
  Empty(*options.execution_domain);
  Ledger screened;
  ModularProfile fast;
  options.search_evaluation = ModularSearchEvaluation::kShortlist;
  options.candidate_observer = &screened;
  { ProfileSession profile(&fast); Ok(EncodeModularImage(source, options, &unprofiled)); }
  Check(fast.screened_candidates == oracle.records.size() &&
        fast.exactly_measured_candidates == std::min(size_t{4}, oracle.records.size()) &&
        fast.candidate_emissions == 1 && fast.encoded_candidates == reference.encoded_candidates,
        "Shortlist did not preserve enumeration or bounded exact work");
  size_t best = SIZE_MAX;
  bool checkpoint = false;
  for (const auto &record : screened.records) {
    Check(record.index < oracle.records.size(), "Shortlist candidate index is invalid");
    if (!record.exact) {
      Check(record.estimated_bits > 0 && record.sampled_tokens > 0 &&
            record.sampled_tokens <= 32768 + 16 && !record.cost.bytes,
            "Screening score was reported as an exact cost");
      continue;
    }
    Check(record.cost == oracle.records[record.index].cost,
          "Shortlisted exact cost differs from exhaustive emitted bytes");
    if (record.cost.bytes < best || (record.cost.bytes == best && checkpoint && !record.checkpoint)) {
      best = record.cost.bytes;
      checkpoint = record.checkpoint;
    }
  }
  Check(unprofiled.size() == best && best <= oracle.records[0].cost.bytes,
        "Shortlist selected the wrong finalist or omitted the baseline");
#ifdef GJXL_CANDIDATE_COST_REFERENCE
  Check(test::modular_reference::DecodeLossless(full) == test::modular_reference::DecodeLossless(unprofiled),
        "Shortlist complete file changed decoded samples");
#endif
  fast.resolved = {};
  Empty(*options.execution_domain);
  options.search_evaluation = ModularSearchEvaluation::kMeasureAll;
  options.candidate_observer = nullptr;
  const auto domain = options.execution_domain;
  Ok(ExecutionDomain::Create({capacity - 1, 2}, &options.execution_domain));
  // Include the same evaluation reservation even without the diagnostic ledger.
  const auto prior = sized;
  Check(!EncodeModularImage(source, options, &sized).ok() && sized == prior,
        "Insufficient evaluation admission published an output");
  Empty(*options.execution_domain);
  options.execution_domain = domain;
  measured.fail_at = 0;
  options.candidate_observer = &measured;
  Check(!EncodeModularImage(source, options, &sized).ok() && sized == prior,
        "Observer failure published a partial search output");
  Empty(*options.execution_domain);
  options.search_evaluation = static_cast<ModularSearchEvaluation>(97);
  Check(!EncodeModularImage(source, options, &sized).ok() && sized == prior,
        "Invalid evaluation mode changed output");
  std::cout << "search format " << int(format) << " mode " << int(entropy) << " selector " << int(selector)
            << " gpu " << bool(gpu) << " constant " << constant << ": " << oracle.records.size()
            << " exact candidate costs and winner match\n";
}
} // namespace
int main() try {
  for (auto format : {PackedModularFormat::kGray8, PackedModularFormat::kGray16,
                      PackedModularFormat::kRgb8, PackedModularFormat::kRgba8,
                      PackedModularFormat::kRgb16, PackedModularFormat::kRgba16}) {
    Fixture(format, EntropyCodingMode::kPrefix, AnsSelectionMode::kExact);
    Fixture(format, EntropyCodingMode::kAns, AnsSelectionMode::kExact);
    Fixture(format, EntropyCodingMode::kAns, AnsSelectionMode::kEstimated);
  }
  Fixture(PackedModularFormat::kRgb8, EntropyCodingMode::kAns, AnsSelectionMode::kEstimated, nullptr, true);
#ifdef GJXL_CANDIDATE_SEARCH_METAL
  std::unique_ptr<GpuBackend> gpu;
  Ok(CreateEmbeddedMetalBackend({}, &gpu));
  for (auto selector : {AnsSelectionMode::kExact, AnsSelectionMode::kEstimated})
    Fixture(PackedModularFormat::kRgba16, EntropyCodingMode::kAns, selector, gpu.get());
#endif
  return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
