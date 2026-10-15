// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/workflow.h"
#include "codestream/workflow_admission_scope.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"
#include "modular_prepared_entropy_cases.h"
#include "modular_reference_workflow_fixture.h"
#include <barrier>
#include <iostream>
#include <thread>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::thread_budget_internal;
using codestream_internal::CodestreamBuffer;
void Check(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
auto Domain(size_t cpus = 4, size_t memory = size_t{256} << 20) {
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({memory, cpus}, &domain));
  return domain;
}
const ResourceBudget &Budget(const ExecutionDomain &domain) {
  return codestream_internal::WorkflowAdmission::Budget(domain);
}
void Released(const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  Check(!s.live_capacity_bytes && !s.reserved_unbacked_bytes && !s.active_reservations &&
            !s.active_cpu_participants && !s.reserved_cpu_workers && !s.suspended_cpu_workers &&
            !s.waiting_cpu_callers && !s.waiting_requests,
        "Tokenization leaked CPU or storage");
}
struct Fixture {
  test::modular_workflow_fixture::Fixture input;
  ModularEncoderFrame frame;
  ModularStreamPlan layout;
  DecisionTree tree;
  std::vector<std::vector<ModularChannelView>> channels;
  std::vector<PreparedModularStreamView> streams;
  Fixture(Extent2D extent = {1025, 13}, size_t count = 3, unsigned bits = 8,
          bool squeeze = false) : input(test::modular_workflow_fixture::Make(extent, count, bits, 3)) {
    ModularCodingPolicy policy;
    if (squeeze) {
      policy.transforms.size = 6;
      for (size_t i = 0; i < 6; ++i)
        policy.transforms.entries[i] = {TransformKind::kSqueeze, 0, uint8_t(count), 1,
                                         i % 2 == 0, false};
    }
    Ok(ModularEncoderFrame::Prepare(input.view(), policy, &frame));
    ChannelShape shape;
    Ok(DescribeImage(frame.image(), &shape));
    ModularFrameGeometry geometry;
    Ok(ModularFrameGeometry::Create(extent, &geometry));
    Ok(BuildModularStreamPlan(geometry, shape.channels(), shape.metadata, &layout));
    tree.size = 3;
    tree.nodes[0] = {.property = 0, .split = 0, .left = 1, .right = 2};
    tree.nodes[1].predictor = Predictor::kWeighted;
    tree.nodes[2].predictor = Predictor::kGradient;
    channels.resize(layout.streams.size());
    streams.resize(channels.size());
    for (size_t i = 0; i < streams.size(); ++i) {
      const auto &entry = layout.streams[i];
      for (const auto &slice : std::span(layout.slices).subspan(entry.slice_begin, entry.slice_count)) {
        ModularChannelView view;
        Ok(BorrowChannelSlice(frame.image().view(slice.channel), slice.rect, &view));
        channels[i].push_back(view);
      }
      streams[i].channels = channels[i];
    }
    streams[0].transforms = policy.transforms;
  }
  PreparedModularStreamsView View(int scope) {
    for (size_t i = 0; i < streams.size(); ++i)
      streams[i].local_tree = !channels[i].empty() && (scope == 1 || (scope == 2 && i % 2))
                                  ? &tree : nullptr;
    return {input.view().extent, input.format, scope == 1 ? nullptr : &tree, streams};
  }
};
template <typename Function>
Status Admitted(PreparedModularStreamsView input, PreparedModularTokenizationOptions options,
                const std::shared_ptr<const ExecutionDomain> &domain, size_t cpus,
                bool complete, Function &&run, bool nested = false, size_t fail_after = SIZE_MAX,
                bool concurrent = false,
                PreparedModularEntropyOptions entropy = {PreparedModularEntropyPolicy::kPrefix}) {
  ResourceContextScope context({.domain = domain.get()});
  CpuExecutionScope cpu;
  if (auto s = cpu.Start(domain, cpus); !s.ok()) return s;
  EncodeScope threads(cpus);
  const auto work = [&]() -> Status {
    ModularWorkflowStoragePlan plan;
    ArmNextManagedHostAllocationFailureForTest();
    const auto planned = ComputePreparedModularStreamsStoragePlan(input, entropy, &plan, options);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(pending, "Tokenization planner allocated");
    if (!planned.ok()) return planned;
    const size_t bound = complete ? plan.working.peak_bytes : plan.preparation.peak_bytes;
    ResourceReservation reservation;
    if (auto s = Budget(*domain).TryReserve(bound, &reservation); !s.ok()) return s;
    ResourceContextScope resources({.reservation = &reservation,
                                    .resource_class = ResourceClass::kPreparation,
                                    .domain = domain.get()});
    if (fail_after != SIZE_MAX) ArmManagedHostAllocationFailureAfterForTest(fail_after);
    const auto status = run();
    DisarmManagedHostAllocationFailureForTest();
    Check(Budget(*domain).snapshot().peak_backing_bytes <= bound * (concurrent ? 2 : 1),
          "Tokenization exceeded its storage bound");
    return status;
  };
  if (nested) {
    ParallelScope scope(cpus, nullptr, CurrentResourceContext());
    return work();
  }
  return work();
}
Status Tokens(PreparedModularStreamsView input, PreparedModularTokenizationOptions options,
              const std::shared_ptr<const ExecutionDomain> &domain, PreparedModularStreamTokens *out,
              ModularProfile *profile = nullptr, bool nested = false, size_t fail_after = SIZE_MAX,
              bool concurrent = false, size_t cpus = 4) {
  ProfileSession session(profile);
  return Admitted(input, options, domain, cpus, false,
                   [&] { return TokenizePreparedModularStreams(input, out, options); },
                   nested, fail_after, concurrent);
}
void Same(const PreparedModularStreamTokens &a, const PreparedModularStreamTokens &b) {
  Check(std::ranges::equal(a.tree_tokens, b.tree_tokens) &&
            std::ranges::equal(a.global_tree_tokens, b.global_tree_tokens) &&
            a.streams.size() == b.streams.size() && a.local_tree_tokens.size() == b.local_tree_tokens.size(),
        "Tree tokens or canonical slots differ");
  for (size_t i = 0; i < a.streams.size(); ++i) {
    Check(a.streams[i].size() == b.streams[i].size() &&
              std::ranges::equal(a.local_tree_tokens[i], b.local_tree_tokens[i]),
          "Stream lengths or local headers differ");
    for (size_t t = 0; t < a.streams[i].size(); ++t)
      Check(a.streams[i][t] == b.streams[i][t], "Parallel sample tokens differ");
  }
}
void Parity() {
  size_t decoded = 0;
  struct Case { Extent2D extent; size_t channels; unsigned bits; bool squeeze; };
  for (auto c : {Case{{17, 19}, 1, 8, false}, {{1025, 13}, 3, 8, false},
                 {{2051, 17}, 4, 16, true}}) {
    Fixture f(c.extent, c.channels, c.bits, c.squeeze);
    for (int scope = 0; scope < 3; ++scope) {
      const auto view = f.View(scope);
      PreparedModularStreamTokens serial;
      Ok(TokenizePreparedModularStreams(view, &serial, {1}));
      for (size_t requested : {size_t{2}, size_t{4}, size_t{0}}) {
        auto domain = Domain();
        PreparedModularStreamTokens parallel;
        ModularProfile profile;
        Ok(Tokens(view, {requested}, domain, &parallel, &profile));
        Same(serial, parallel);
        Check(std::ranges::equal(serial.tokens, parallel.tokens), "Owned token order differs");
        const size_t jobs = std::ranges::count_if(serial.streams, [](auto s) { return s.size() != 0; });
        Check(profile.parallel_tokenization.stream_jobs == jobs &&
                  profile.parallel_tokenization.generated_tokens == serial.tokens.size() &&
                  !profile.parallel_tokenization.borrowed_tokens &&
                  profile.parallel_tokenization.peak_participants == std::min(jobs, requested ? requested : 4) &&
                  domain->snapshot().peak_cpu_participants <= 4,
              "Token counters or CPU participation differ");
        parallel = {};
        Released(*domain);
      }
      for (auto variant : test::kPreparedEntropyCases) {
        variant.options.cpu_participants = 1; // Token parallelism must be independent.
        CodestreamBuffer a, b;
        Ok(EncodePreparedModularStreams(view, variant.options, &a, {1}));
        auto domain = Domain();
        ModularProfile profile;
        {
          ProfileSession session(&profile);
          Ok(Admitted(view, {4}, domain, 4, true, [&] {
            return EncodePreparedModularStreams(view, variant.options, &b, {4});
          }, false, SIZE_MAX, false, variant.options));
        }
        Check(std::ranges::equal(a.view(), b.view()), "Parallel tokenization changed bytes");
        Check(!profile.parallel_entropy.parallel_calls &&
                  profile.parallel_tokenization.parallel_calls == (c.extent.width > 256),
              "Entropy and token allowances interfered");
#ifdef GJXL_REFERENCE_WORKFLOW_ORACLE
        Check(test::modular_reference::DecodeLossless(b.view()) == f.input.original,
              "Independent decoded samples differ");
        ++decoded;
#endif
        b = {};
        Released(*domain);
      }
    }
  }
  std::cout << decoded << " parallel files independently decoded\n";
}
void FeaturesAndBorrowedTokens() {
  Fixture f;
  const auto view = f.View(2);
  PreparedModularStreamTokens serial;
  Ok(TokenizePreparedModularStreams(view, &serial, {1}));
  std::vector<std::vector<std::vector<WeightedPredictorFeature>>> maps(f.streams.size());
  std::vector<std::vector<WeightedFeatureView>> features(f.streams.size());
  std::vector<std::vector<uint32_t>> values(f.streams.size());
  std::vector<std::vector<uint16_t>> contexts(f.streams.size());
  for (size_t i = 0; i < f.streams.size(); ++i) {
    auto &stream = f.streams[i];
    maps[i].resize(stream.channels.size());
    features[i].resize(stream.channels.size());
    for (size_t c = 0; c < stream.channels.size(); ++c) {
      const auto &channel = stream.channels[c];
      maps[i][c].resize(channel.descriptor.extent.width * channel.descriptor.extent.height);
      Ok(PrepareWeightedPredictorFeatures(channel, stream.weighted, maps[i][c]));
      features[i][c] = {channel, stream.weighted, maps[i][c]};
    }
    for (size_t t = 0; t < serial.streams[i].size(); ++t) {
      values[i].push_back(serial.streams[i][t].value);
      contexts[i].push_back(serial.streams[i][t].context);
    }
  }
  for (int cached : {1, 2}) {
    for (size_t i = 0; i < f.streams.size(); ++i) {
      if (cached == 1 && !features[i].empty()) features[i][0] = {};
      else if (cached == 2 && !features[i].empty())
        features[i][0] = {f.streams[i].channels[0], f.streams[i].weighted, maps[i][0]};
      f.streams[i].weighted_features = features[i];
    }
    for (int borrowed : {0, 1, 2}) {
      size_t borrowed_count = 0;
      for (size_t i = 0; i < f.streams.size(); ++i) {
        f.streams[i].completed_tokens.reset();
        if (serial.streams[i].size() && (borrowed == 2 || (borrowed == 1 && i % 2))) {
          const auto tokens = i % 2 ? EntropyTokenStreamView::Split(values[i], contexts[i]) : serial.streams[i];
          f.streams[i].completed_tokens = PreparedModularStreamTokenView{
              tokens, f.streams[i].channels, &f.tree, f.streams[i].weighted, f.layout.streams[i].id};
          borrowed_count += tokens.size();
        }
      }
      auto domain = Domain();
      PreparedModularStreamTokens result;
      ModularProfile profile;
      Ok(Tokens(view, {}, domain, &result, &profile));
      Same(serial, result);
      Check(result.tokens.size() + borrowed_count == serial.tokens.size() &&
                profile.parallel_tokenization.borrowed_tokens == borrowed_count &&
                profile.parallel_tokenization.generated_tokens == result.tokens.size() &&
                profile.parallel_tokenization.parallel_calls == (borrowed != 2),
            "Borrowed token backing copied or incorrectly scheduled");
      for (size_t i = 0; i < f.streams.size(); ++i)
        if (auto ready = f.streams[i].completed_tokens)
          Check(result.streams[i].tokens.data() == ready->tokens.tokens.data() &&
                    result.streams[i].values.data() == ready->tokens.values.data() &&
                    result.streams[i].contexts.data() == ready->tokens.contexts.data(),
                "Completed token views lost their borrowed backing");
      result = {};
      Released(*domain);
    }
  }
}
void LimitsAndFailures() {
  Fixture f;
  const auto view = f.View(1);
  PreparedModularStreamTokens expected;
  Ok(TokenizePreparedModularStreams(view, &expected, {1}));
  for (bool nested : {false, true}) {
    auto domain = Domain(nested ? 4 : 1);
    PreparedModularStreamTokens result;
    ModularProfile profile;
    Ok(Tokens(view, {}, domain, &result, &profile, nested));
    Same(expected, result);
    Check(profile.parallel_tokenization.peak_participants == 1 &&
              !profile.parallel_tokenization.parallel_calls && domain->snapshot().peak_cpu_participants == 1,
          "Nested/domain-limited tokenization spawned workers");
    result = {};
    Released(*domain);
  }
  for (size_t requested : {size_t{0}, size_t{4}}) {
    ModularProfile profile;
    ProfileSession session(&profile);
    PreparedModularStreamTokens result;
    Ok(TokenizePreparedModularStreams(view, &result, {requested}));
    Same(expected, result);
    Check(!profile.parallel_tokenization.parallel_calls, "Unadmitted tokenization spawned workers");
  }
  {
    auto domain = Domain();
    PreparedModularStreamTokens result;
    ModularProfile profile;
    Ok(Tokens(view, {4}, domain, &result, &profile, false, SIZE_MAX, false, 2));
    Same(expected, result);
    Check(profile.parallel_tokenization.peak_participants == 2 &&
              domain->snapshot().peak_cpu_participants <= 2,
          "Tokenization exceeded its per-image limit");
    result = {};
    Released(*domain);
  }
  for (size_t before : {size_t{0}, size_t{1}})
    for (auto kind : {WorkerLaunchFailureKind::kSystemError, WorkerLaunchFailureKind::kBadAlloc}) {
      auto domain = Domain();
      PreparedModularStreamTokens result;
      Ok(TokenizePreparedModularStreams(view, &result, {1})); // Prior result is caller-owned.
      const auto *saved = result.tokens.data();
      WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kModularPreparedTokenization, before, kind};
      WorkerLaunchFaultScopeForTesting inject(&fault);
      const auto status = Tokens(view, {}, domain, &result);
      Check(fault.triggered && fault.launched_in_group == before && result.tokens.data() == saved &&
                status.code() == (kind == WorkerLaunchFailureKind::kBadAlloc
                                     ? StatusCode::kOutOfMemory : StatusCode::kInternal),
            "Token worker launch failure escaped or published output");
      Same(expected, result);
      Released(*domain);
    }
  for (size_t after = 0; after < 14; ++after) {
    auto domain = Domain();
    PreparedModularStreamTokens result;
    Ok(TokenizePreparedModularStreams(view, &result, {1}));
    const auto *saved = result.tokens.data();
    const auto status = Tokens(view, {}, domain, &result, nullptr, false, after);
    Check(status.ok() || (status.code() == StatusCode::kOutOfMemory && result.tokens.data() == saved),
          "Token allocation failure escaped or published output");
    Same(expected, result);
    result = {};
    Released(*domain);
  }
  // Tree shape validates, but sample residuals cannot be represented. This
  // exercises a failure inside a running job, after disjoint output writes.
  f.tree.nodes[1].multiplier = f.tree.nodes[2].multiplier = 2;
  auto domain = Domain();
  PreparedModularStreamTokens result;
  const auto *saved = expected.tokens.data();
  Check(Tokens(view, {}, domain, &expected).code() == StatusCode::kInvalidArgument &&
            expected.tokens.data() == saved, "Worker sample error replaced token output");
  Released(*domain);
  const std::array<uint8_t, 3> sentinel{7, 8, 9};
  CodestreamBuffer output;
  Ok(CodestreamBuffer::CopyFrom(sentinel, &output));
  Check(Admitted(view, {}, domain, 4, true, [&] {
    return EncodePreparedModularStreams(view, EntropyCodingMode::kPrefix, &output);
  }).code() == StatusCode::kInvalidArgument && std::ranges::equal(output.view(), sentinel),
        "Worker sample error replaced encoded output");
  Released(*domain);
  Check(Tokens(view, {257}, domain, &result).code() == StatusCode::kInvalidArgument &&
            TokenizePreparedModularStreams(view, &result, {257}).code() == StatusCode::kInvalidArgument,
        "Invalid token participant limit accepted");
}
void SharedDomain() {
  Fixture f;
  const auto view = f.View(2);
  auto domain = Domain();
  std::barrier start(2);
  std::array<Status, 2> status;
  std::array<PreparedModularStreamTokens, 2> outputs;
  const auto run = [&](size_t i) {
    start.arrive_and_wait();
    status[i] = Tokens(view, {}, domain, &outputs[i], nullptr, false, SIZE_MAX, true);
  };
  std::thread worker(run, 0);
  run(1);
  worker.join();
  Ok(status[0]);
  Ok(status[1]);
  Same(outputs[0], outputs[1]);
  Check(domain->snapshot().peak_cpu_participants <= 4, "Tokenization exceeded shared CPU limit");
  outputs = {};
  Released(*domain);
}
} // namespace
int main() try {
  Parity();
  FeaturesAndBorrowedTokens();
  LimitsAndFailures();
  SharedDomain();
  std::cout << "Prepared token parity, borrowing, bounds, CPU limits and failure unwinding passed\n";
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
