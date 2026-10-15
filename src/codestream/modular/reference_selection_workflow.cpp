// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/reference_workflow_internal.h"
#include "core/parallel_work_internal.h"

namespace gjxl::modular_internal::reference_workflow_internal {
namespace {
using namespace resource_budget_internal;
using namespace thread_budget_internal;
template <typename T> using SelectionStorage = ManagedVector<T, ResourceClass::kPreparation>;
struct SelectionJob {
  ResourceReservation reservation;
  ReferenceWorkflowStats stats;
  size_t index = 0;
};
Status Bound(const ReferenceModularStream &record, ReferenceSelectionOptions options,
             HostStorageBound *bound) {
  return ComputeReferenceSelectionStorageBound(record.training_channels(),
                                                record.metadata_channels, options, bound);
}
Status Select(ReferenceModularStream &record, ReferenceSelectionOptions options,
              const ResourceBudget &budget, ReferenceWorkflowStats *stats, bool admitted = false) {
  Seconds work(stats->selection_work_seconds);
  const auto run = [&]() {
    return SelectReferenceModularEncoding(record.training_channels(), record.metadata_channels,
                                           options, &record.selection);
  };
  if (admitted) {
    if (auto s = run(); !s.ok()) return s;
  } else {
    HostStorageBound bound;
    if (auto s = Bound(record, options, &bound); !s.ok()) return s;
    if (auto s = Stage(budget, bound.peak_bytes, stats, run); !s.ok()) return s;
  }
  ++stats->selection_calls;
  const auto &image = record.selection.image;
  if (image.channel_count()) {
    if (image.channel_count() > record.channels.size())
      return Status::InvalidArgument("Selected Modular channel count exceeds record capacity");
    record.channel_count = image.channel_count();
    record.metadata_channels = image.metadata_channels();
    for (size_t c = 0; c < image.channel_count(); ++c)
      record.channels[c] = image.view(c);
  }
  return Status::Ok();
}
} // namespace

Status SelectReferenceModularScopes(std::span<ReferenceModularStream> records,
                                    ReferenceSelectionOptions options, size_t participants,
                                    const ResourceBudget &budget,
                                    ReferenceWorkflowStats *stats) try {
  if (!stats || participants > 256)
    return Status::InvalidArgument("Invalid Modular selection participation");
  ProfileScope profile(ProfileStage::kTransforms);
  Seconds wall(stats->selection_seconds);
  size_t active = 0;
  for (const auto &record : records) active += record.channel_count != 0;
  if (!active) return Status::Ok();
  const auto serial = [&](size_t first) {
    ++stats->selection_waves;
    stats->selection_peak_participants = std::max(size_t{1}, stats->selection_peak_participants);
    for (size_t i = first; i < records.size(); ++i)
      if (records[i].channel_count)
        if (auto s = Select(records[i], options, budget, stats); !s.ok()) return s;
    return Status::Ok();
  };
  if (!HasCpuParticipation() || InExplicitParallelScope()) return serial(0);
  size_t desired = std::min({active, current_cpu_execution.job.limit(),
                             current_cpu_execution.domain.limit(), size_t{256}});
  if (CpuThreadCount()) desired = std::min(desired, CpuThreadCount());
  if (participants) desired = std::min(desired, participants);
  if (desired < 2) return serial(0);

  HostStorageBound metadata;
  if (!metadata.AddVector<SelectionJob>(desired, VectorCapacityPolicy::kFreshExact) ||
      !metadata.AddVector<Status>(desired, VectorCapacityPolicy::kFreshExact) ||
      !metadata.AddVector<std::thread>(desired - 1, VectorCapacityPolicy::kFreshExact))
    return Status::InvalidArgument("Modular selection scheduler storage overflow");
  ResourceReservation reservation;
  if (auto s = budget.TryReserve(metadata.peak_bytes, &reservation); !s.ok()) {
    if (s.code() != StatusCode::kUnavailable && s.code() != StatusCode::kOutOfMemory) return s;
    ++stats->selection_memory_limited_waves;
    return serial(0);
  }
  stats->maximum_stage_bytes = std::max(stats->maximum_stage_bytes, metadata.peak_bytes);
  auto context = CurrentResourceContext();
  context.reservation = &reservation;
  context.resource_class = ResourceClass::kPreparation;
  ResourceContextScope resources(context);
  SelectionStorage<SelectionJob> jobs(desired);
  size_t next = 0;
  while (next < records.size()) {
    size_t count = 0;
    for (; next < records.size() && count < desired; ++next) {
      if (!records[next].channel_count) continue;
      HostStorageBound bound;
      if (auto s = Bound(records[next], options, &bound); !s.ok()) return s;
      if (auto s = budget.TryReserve(std::max(size_t{1}, bound.peak_bytes),
                                      &jobs[count].reservation); !s.ok()) {
        if (s.code() != StatusCode::kUnavailable && s.code() != StatusCode::kOutOfMemory) return s;
        ++stats->selection_memory_limited_waves;
        break;
      }
      jobs[count].index = next;
      jobs[count].stats.maximum_stage_bytes = bound.peak_bytes;
      ++count;
    }
    if (!count) {
      if (next == records.size()) break;
      // Preserve the original serial envelope even when scheduler metadata
      // would prevent admission. Do not retry selection or worker failures.
      jobs = SelectionStorage<SelectionJob>{};
      reservation.Reset();
      return serial(next);
    }
    CpuWorkerGroup group(count);
    ++stats->selection_waves;
    stats->selection_peak_participants = std::max(stats->selection_peak_participants,
                                                 group.participants());
    const auto run = [&](size_t index, size_t) {
      auto &job = jobs[index];
      auto context = CurrentResourceContext();
      context.reservation = &job.reservation;
      ResourceContextScope resources(context);
      EncodeScope allowance(1, ParticipantTracker());
      // Only the caller's joined interval contributes top-level stage time.
      ProfileSession private_profile(nullptr);
      return Select(records[job.index], options, budget, &job.stats, true);
    };
    Status status;
    if (group.participants() > 1) {
      status = RunParallelWork<SelectionStorage>(
          count, group, group.participants() - 1, WorkerLaunchSite::kModularSelection,
          {"Modular selection worker allocation failed", "Modular selection worker failed",
           StatusCode::kInvalidArgument, "Modular selection worker storage overflow",
           "Modular selection worker launch allocation failed", LaunchFailureAction::kReturnError,
           "Unable to start Modular selection workers"}, run);
    } else {
      for (size_t i = 0; i < count && status.ok(); ++i) status = run(i, 0);
    }
    // Join precedes canonical counter reduction and reservation release. The
    // selected images remain accounted as retained backing in the same domain.
    for (size_t i = 0; i < count; ++i) {
      stats->selection_calls += jobs[i].stats.selection_calls;
      stats->selection_work_seconds += jobs[i].stats.selection_work_seconds;
      stats->maximum_stage_bytes = std::max(stats->maximum_stage_bytes,
                                            jobs[i].stats.maximum_stage_bytes);
      jobs[i] = {};
    }
    if (!status.ok()) return status;
  }
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &error) {
  return error.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular selection allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular selection storage overflow");
}
} // namespace gjxl::modular_internal::reference_workflow_internal
