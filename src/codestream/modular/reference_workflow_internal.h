// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/reference_workflow.h"
#include <chrono>

namespace gjxl::modular_internal::reference_workflow_internal {
template <typename Function>
Status Stage(const resource_budget_internal::ResourceBudget &budget, size_t bytes,
             ReferenceWorkflowStats *stats, Function &&function) {
  using namespace resource_budget_internal;
  ResourceReservation reservation;
  if (auto s = budget.TryReserve(std::max(size_t{1}, bytes), &reservation); !s.ok())
    return s;
  stats->maximum_stage_bytes = std::max(stats->maximum_stage_bytes, bytes);
  auto context = CurrentResourceContext();
  context.reservation = &reservation;
  context.resource_class = ResourceClass::kPreparation;
  ResourceContextScope scope(context);
  return function();
}
class Seconds {
  using Clock = std::chrono::steady_clock;

public:
  explicit Seconds(double &out) : out_(out), start_(Clock::now()) {}
  ~Seconds() { out_ += std::chrono::duration<double>(Clock::now() - start_).count(); }
  Seconds(const Seconds &) = delete;
  Seconds &operator=(const Seconds &) = delete;

private:
  double &out_;
  Clock::time_point start_;
};
using GpuTokenOwners = resource_budget_internal::ManagedVector<
    std::unique_ptr<modular_gpu_internal::PreparedGpuModularTokens>>;
// Records and stream views are unpublished and remain at stable addresses.
// CPU learning policy is shared by both routes. Hybrid token owners outlive
// the completed borrows installed into streams; feature owners never escape.
[[nodiscard]] Status
LearnReferenceModularScopes(std::span<ReferenceModularStream> records,
                            std::span<PreparedModularStreamView> streams, uint64_t population,
                            const ReferenceWorkflowOptions &options,
                            const resource_budget_internal::ResourceBudget &budget,
                            ReferenceWorkflowStats *stats, GpuTokenOwners *tokens);
} // namespace gjxl::modular_internal::reference_workflow_internal
