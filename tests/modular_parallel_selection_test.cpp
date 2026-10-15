// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/reference_workflow_internal.h"
#include "codestream/workflow_admission_scope.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"
#include "modular_reference_workflow_fixture.h"
#include <barrier>
#include <iostream>
#include <thread>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::thread_budget_internal;
using test::modular_workflow_fixture::Make;
using codestream_internal::CodestreamBuffer;
void Check(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
auto Domain(size_t memory = size_t{256} << 20, size_t cpus = 4) {
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({memory, cpus}, &domain));
  return domain;
}
const ResourceBudget &Budget(const ExecutionDomain &domain) {
  return codestream_internal::WorkflowAdmission::Budget(domain);
}
template <typename Function>
Status WithCpu(const std::shared_ptr<const ExecutionDomain> &domain, size_t cpus,
                Function &&run, bool nested = false) {
  ResourceContextScope resources({.domain = domain.get()});
  CpuExecutionScope cpu;
  if (auto s = cpu.Start(domain, cpus); !s.ok()) return s;
  EncodeScope threads(cpus);
  if (nested) {
    ParallelScope outer(cpus, nullptr, CurrentResourceContext());
    return run();
  }
  return run();
}
void Released(const ExecutionDomain &domain) {
  auto s = domain.snapshot();
  Check(!s.live_capacity_bytes && !s.reserved_unbacked_bytes && !s.active_reservations &&
            !s.active_cpu_participants && !s.reserved_cpu_workers && !s.suspended_cpu_workers &&
            !s.waiting_cpu_callers && !s.waiting_requests,
        "Selection leaked storage or CPU participation");
}
void Same(std::span<const ReferenceModularStream> a, std::span<const ReferenceModularStream> b) {
  Check(a.size() == b.size(), "Stream count differs");
  for (size_t i = 0; i < a.size(); ++i) {
    const auto &p = a[i], &q = b[i];
    Check(p.id == q.id && p.channel_count == q.channel_count &&
              p.metadata_channels == q.metadata_channels && p.tree == q.tree &&
              p.selection.transforms == q.selection.transforms &&
              p.selection.trace == q.selection.trace && p.selection.wp_mode == q.selection.wp_mode,
          "Selection decisions, trace, tree or canonical IDs differ");
    for (size_t c = 0; c < p.channel_count; ++c) {
      const auto &x = p.channels[c], &y = q.channels[c];
      Check(x.descriptor == y.descriptor, "Selected channel geometry differs");
      for (size_t row = 0; row < x.descriptor.extent.height; ++row)
        Check(std::equal(x.Row(row), x.Row(row) + x.descriptor.extent.width, y.Row(row)),
              "Selected samples differ");
    }
  }
}
void Encode(const ReferenceModularFrame &frame, PreparedModularEntropyOptions entropy,
            const ResourceBudget &budget, CodestreamBuffer *out) {
  ModularWorkflowStoragePlan plan;
  Ok(ComputePreparedModularStreamsStoragePlan(frame.view(), entropy, &plan));
  ResourceReservation reservation;
  Ok(budget.TryReserve(plan.working.peak_bytes, &reservation));
  ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
  Ok(EncodePreparedModularStreams(frame.view(), entropy, out));
}
void Parity() {
  struct Case { Extent2D extent; size_t channels; unsigned bits, pattern; bool buffered; };
  for (auto c : {Case{{17, 19}, 1, 8, 0, false}, {{2049, 9}, 3, 8, 0, false},
                 {{1025, 9}, 3, 8, 1, false}, {{2049, 9}, 4, 16, 1, false},
                 {{2049, 9}, 3, 16, 2, true}, {{2049, 9}, 4, 8, 3, false}}) {
    const auto input = Make(c.extent, c.channels, c.bits, c.pattern, SampleByteOrder::kLittleEndian);
    auto domain = Domain();
    ReferenceWorkflowOptions options;
    options.learning_cpu_participants = 1;
    options.selection_cpu_participants = 1;
    options.buffering = c.buffered ? ReferenceBuffering::kBuffered : ReferenceBuffering::kAutomatic;
    ReferenceModularFrame serial, parallel;
    const auto prepare = [&](ReferenceModularFrame *out) {
      return WithCpu(domain, 4, [&] {
        return PrepareReferenceModularFrame(input.view(), options, Budget(*domain), out);
      });
    };
    Ok(prepare(&serial));
    for (size_t requested : {size_t{2}, size_t{4}, size_t{0}}) {
      options.selection_cpu_participants = requested;
      ModularProfile profile;
      const auto start = std::chrono::steady_clock::now();
      {
        ProfileSession session(&profile);
        Ok(prepare(&parallel));
      }
      const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      Same(serial.streams(), parallel.streams());
      Check(serial.stats().selection_calls == parallel.stats().selection_calls &&
                profile.seconds[size_t(ProfileStage::kTransforms)] >= parallel.stats().selection_seconds &&
                profile.seconds[size_t(ProfileStage::kTransforms)] <= elapsed,
            "Selection counters or joined profiling differ");
      if (c.extent.width > 256)
        Check(parallel.stats().selection_peak_participants == (requested ? requested : 4),
              "Selection did not use its CPU allowance");
      else
        Check(parallel.stats().selection_peak_participants == 1, "Global selection fanned out");
      for (auto entropy : {options.entropy,
                           PreparedModularEntropyOptions{PreparedModularEntropyPolicy::kPrefix}}) {
        CodestreamBuffer a, b;
        Encode(serial, entropy, Budget(*domain), &a);
        Encode(parallel, entropy, Budget(*domain), &b);
        Check(std::ranges::equal(a.view(), b.view()), "Parallel selection changed complete bytes");
#ifdef GJXL_REFERENCE_WORKFLOW_ORACLE
        Check(test::modular_reference::DecodeLossless(b.view()) == input.original,
              "Independent decoded samples differ");
#endif
      }
      parallel = {};
    }
    serial = {};
    Released(*domain);
  }
}
// Isolate selection admission from the much larger subsequent fitter bound.
struct SelectionFixture {
  ModularEncoderFrame source;
  SelectionFixture() {
    const auto input = Make({2049, 17}, 3, 8, 3);
    Ok(ModularEncoderFrame::Prepare(input.view(), &source));
  }
  auto Records() const {
    std::vector<ReferenceModularStream> records(12); // Canonical empty slots included.
    for (size_t i = 0; i < 9; ++i) {
      auto &record = records[i + 1];
      record.id = i + 1;
      record.channel_count = 3;
      for (size_t c = 0; c < 3; ++c)
        Ok(BorrowChannelSlice(source.image().view(c),
                              {i * 256, 0, {std::min(size_t{256}, 2049 - i * 256), 17}},
                              &record.channels[c]));
    }
    return records;
  }
};
Status Select(std::span<ReferenceModularStream> records, const ResourceBudget &budget,
              ReferenceWorkflowStats *stats, size_t requested = 0) {
  return reference_workflow_internal::SelectReferenceModularScopes(
      records, {ReferencePaletteStage::kLocal, 8}, requested, budget, stats);
}
void LimitsAndMemory() {
  SelectionFixture f;
  auto control = Domain();
  auto expected = f.Records();
  ReferenceWorkflowStats serial;
  Ok(Select(expected, Budget(*control), &serial)); // Unadmitted calls stay serial.
  Check(serial.selection_peak_participants == 1, "Unadmitted selection spawned workers");
  auto limited = Domain(control->snapshot().peak_committed_bytes);
  auto actual = f.Records();
  ReferenceWorkflowStats stats;
  Ok(WithCpu(limited, 4, [&] { return Select(actual, Budget(*limited), &stats); }));
  Same(expected, actual);
  Check(stats.selection_memory_limited_waves && stats.selection_peak_participants < 4 &&
            limited->snapshot().peak_committed_bytes <= control->snapshot().peak_committed_bytes,
        "Selection did not retain the serial memory envelope");
  actual.clear();
  expected.clear();
  Released(*limited);
  Released(*control);
  for (size_t domain_cpus : {size_t{1}, size_t{2}, size_t{4}})
    for (size_t image_cpus : {size_t{1}, size_t{2}, size_t{4}}) {
      auto domain = Domain(size_t{64} << 20, domain_cpus);
      auto records = f.Records();
      ReferenceWorkflowStats stats;
      Ok(WithCpu(domain, image_cpus, [&] { return Select(records, Budget(*domain), &stats, 8); }));
      Check(stats.selection_peak_participants <= std::min(domain_cpus, image_cpus) &&
                domain->snapshot().peak_cpu_participants <= std::min(domain_cpus, image_cpus),
            "Selection exceeded a CPU limit");
      records = f.Records();
      stats = {};
      Ok(WithCpu(domain, image_cpus, [&] { return Select(records, Budget(*domain), &stats); }, true));
      Check(stats.selection_peak_participants == 1, "Nested selection spawned workers");
      records.clear();
      Released(*domain);
    }
}
void Failures() {
  const auto input = Make({2049, 9}, 3, 8, 3);
  auto domain = Domain();
  ReferenceModularFrame out;
  ReferenceWorkflowOptions options;
  options.learning_cpu_participants = 1;
  Ok(PrepareReferenceModularFrame(Make({3, 3}, 3, 8, 1).view(), options, Budget(*domain), &out));
  const auto *identity = out.streams().data();
  const auto retained = domain->snapshot().live_capacity_bytes;
  const auto prepare = [&] {
    return WithCpu(domain, 4, [&] {
      return PrepareReferenceModularFrame(input.view(), options, Budget(*domain), &out);
    });
  };
  for (auto kind : {WorkerLaunchFailureKind::kSystemError, WorkerLaunchFailureKind::kBadAlloc})
    for (size_t before : {size_t{0}, size_t{1}}) {
      WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kModularSelection, before, kind};
      WorkerLaunchFaultScopeForTesting inject(&fault);
      const auto status = prepare();
      Check(fault.triggered && fault.launched_in_group == before &&
                status.code() == (kind == WorkerLaunchFailureKind::kBadAlloc
                                     ? StatusCode::kOutOfMemory : StatusCode::kInternal) &&
                out.streams().data() == identity && domain->snapshot().live_capacity_bytes == retained &&
                !domain->snapshot().active_reservations && !domain->snapshot().active_cpu_participants,
            "Partial selection launch failed to unwind or preserve the previous frame");
    }
  options.selection_cpu_participants = 257;
  Check(prepare().code() == StatusCode::kInvalidArgument && out.streams().data() == identity,
        "Invalid selection participation changed the frame");
  out = {};
  Released(*domain);
  SelectionFixture fixture;
  for (size_t after : {0, 1, 2, 3, 4, 8, 16, 32, 64}) {
    auto records = fixture.Records();
    ReferenceWorkflowStats stats;
    ArmManagedHostAllocationFailureAfterForTest(after);
    auto status = WithCpu(domain, 4, [&] { return Select(records, Budget(*domain), &stats); });
    DisarmManagedHostAllocationFailureForTest();
    Check(status.ok() || status.code() == StatusCode::kOutOfMemory, "Allocation error escaped");
    records.clear();
    Released(*domain);
  }
}
void SharedDomain() {
  SelectionFixture f;
  auto domain = Domain(size_t{128} << 20, 3);
  std::array records{f.Records(), f.Records()};
  std::array<Status, 2> results;
  std::array<ReferenceWorkflowStats, 2> stats;
  std::barrier ready(2);
  const auto run = [&](size_t i) {
    ready.arrive_and_wait();
    results[i] = WithCpu(domain, 4, [&] { return Select(records[i], Budget(*domain), &stats[i]); });
  };
  std::thread worker(run, 0);
  run(1);
  worker.join();
  Ok(results[0]);
  Ok(results[1]);
  Same(records[0], records[1]);
  Check(domain->snapshot().peak_cpu_participants <= 3 &&
            domain->snapshot().peak_cpu_protected_slots <= 3,
        "Concurrent selectors exceeded the shared CPU domain");
  records = {};
  Released(*domain);
}
} // namespace
int main() try {
  Parity();
  LimitsAndMemory();
  Failures();
  SharedDomain();
  std::cout << "Parallel selection decisions, samples, trees, bytes, decode, limits and failures passed\n";
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
