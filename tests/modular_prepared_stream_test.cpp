// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/prepared_streams.h"
#include "modular_prepared_entropy_cases.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
using gjxl::codestream_internal::CodestreamBuffer;
void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Ok(Status s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
struct Fixture {
  std::array<int32_t, 9> pixels = {5, 1, 7, 8, 2, 6, 3, 9, 4};
  std::array<ModularChannelView, 1> channels = {ModularChannelView{pixels, {{3, 3}}, 3}};
  DecisionTree tree;
  std::array<PreparedModularStreamView, 3> streams{};
  Fixture() { streams[0].channels = channels; }
  PreparedModularStreamsView View(bool local) {
    streams[0].local_tree = local ? &tree : nullptr;
    return {{3, 3}, PackedModularFormat::kGray8, local ? nullptr : &tree, streams};
  }
};
void AllocationSweep(bool local, const test::PreparedEntropyCase &variant) {
  const auto &options = variant.options;
  Fixture f;
  auto view = f.View(local);
  ModularWorkflowStoragePlan plan;
  ArmNextManagedHostAllocationFailureForTest();
  auto status = ComputePreparedModularStreamsStoragePlan(view, options, &plan);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Check(status.ok() && pending, "Prepared stream planner allocated");
  CodestreamBuffer expected;
  Ok(EncodePreparedModularStreams(view, options, &expected));
  if (options.policy != PreparedModularEntropyPolicy::kDirectRateAns) {
    const auto mode = options.policy == PreparedModularEntropyPolicy::kPrefix
                          ? EntropyCodingMode::kPrefix
                          : EntropyCodingMode::kAns;
    ModularWorkflowStoragePlan old_plan;
    CodestreamBuffer old_bytes;
    Ok(ComputePreparedModularStreamsStoragePlan(view, mode, &old_plan));
    Ok(EncodePreparedModularStreams(view, mode, &old_bytes));
    Check(old_plan == plan && std::ranges::equal(old_bytes.view(), expected.view()),
          "Legacy prepared API changed plans or bytes");
  } else {
    auto alternate = options;
    alternate.selection = options.selection == AnsSelectionMode::kExact
                              ? AnsSelectionMode::kEstimated
                              : AnsSelectionMode::kExact;
    ModularWorkflowStoragePlan alternate_plan;
    Ok(ComputePreparedModularStreamsStoragePlan(view, alternate, &alternate_plan));
    Check(alternate_plan == plan, "Width estimation changed the conservative storage plan");
  }
  const std::array<uint8_t, 3> sentinel = {7, 8, 9};
  bool finished = false;
  size_t failures = 0;
  ResourceBudget budget(plan.working.peak_bytes);
  ResourceReservation reservation;
  Ok(budget.Reserve(plan.working.peak_bytes, &reservation));
  for (; failures < 10000; ++failures) {
    CodestreamBuffer output;
    Ok(CodestreamBuffer::CopyFrom(sentinel, &output));
    {
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      ArmManagedHostAllocationFailureAfterForTest(failures);
      const auto s = EncodePreparedModularStreams(view, options, &output);
      const bool pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (s.ok()) {
        Check(pending && std::ranges::equal(expected.view(), output.view()),
              "Recovery bytes differ");
        Check(output.size() <= plan.maximum_codestream_bytes &&
                  budget.snapshot().total.live_capacity_bytes +
                          budget.snapshot().total.idle_capacity_bytes <=
                      plan.output.retained_bytes,
              "Prepared file exceeded bound");
        finished = true;
      } else {
        Check(!pending && s.code() == StatusCode::kOutOfMemory &&
                  std::ranges::equal(output.view(), sentinel),
              "Failure published partial output");
        Check(budget.snapshot().total.backing_count == 0 &&
                  budget.snapshot().total.pending_count == 0,
              "Failed prepared encode leaked backing");
      }
    }
    if (finished)
      break;
  }
  Check(finished && budget.snapshot().total.backing_count == 0 &&
            budget.snapshot().peak_backing_bytes <= plan.working.peak_bytes,
        "Prepared encode storage sweep incomplete");
  std::cout << (local ? "local " : "global ") << variant.name << ' ' << failures
            << " allocation failures qualified\n";
  ResourceBudget tiny_budget(1);
  ResourceReservation tiny_reservation;
  Ok(tiny_budget.Reserve(1, &tiny_reservation));
  CodestreamBuffer output;
  Ok(CodestreamBuffer::CopyFrom(sentinel, &output));
  {
    ResourceContextScope scope({&tiny_reservation, ResourceClass::kPreparation});
    Check(EncodePreparedModularStreams(view, options, &output).code() == StatusCode::kOutOfMemory &&
              std::ranges::equal(output.view(), sentinel) &&
              tiny_budget.snapshot().total.backing_count == 0 &&
              tiny_budget.snapshot().total.pending_count == 0,
          "Underplanned encode published output or leaked backing");
  }
}
void Validation() {
  for (unsigned error = 0; error < 10; ++error) {
    Fixture f;
    auto view = f.View(false);
    if (error == 0)
      view.streams = view.streams.first(2);
    if (error == 1)
      f.channels[0].descriptor.extent.width = 4;
    if (error == 2)
      f.channels[0].descriptor.role = ChannelRole::kAlpha;
    if (error == 3)
      f.channels[0].stride = 2;
    if (error == 4)
      f.streams[0].weighted.coefficients[0] = 32;
    if (error == 5)
      view.global_tree = nullptr;
    if (error == 6)
      f.tree.size = 0;
    if (error == 7)
      f.streams[1].channels = f.channels;
    if (error == 8)
      f.streams[0].rct = 6;
    if (error == 9) {
      f.streams[0].transforms.size = 1;
      f.streams[0].transforms.entries[0].colors = 1025;
    }
    ModularWorkflowStoragePlan plan;
    plan.tokens = 73;
    const auto original = plan;
    ArmNextManagedHostAllocationFailureForTest();
    const auto s =
        ComputePreparedModularStreamsStoragePlan(view, EntropyCodingMode::kPrefix, &plan);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(!s.ok() && pending && plan == original,
          "Invalid prepared plan allocated or changed output");
  }
  Fixture f;
  auto view = f.View(true);
  PreparedModularStreamTokens tokens;
  Ok(TokenizePreparedModularStreams(view, &tokens));
  const auto first = tokens.tokens[0];
  PreparedModularStreamTokens moved = std::move(tokens);
  Check(moved.streams[0][0] == first && !moved.local_tree_tokens[0].empty(),
        "Moved stream tokens lost ownership");
  f.tree.nodes[0].predictor = Predictor::kZero;
  f.tree.nodes[0].multiplier = 2;
  Check(!TokenizePreparedModularStreams(view, &moved).ok() && moved.streams[0][0] == first,
        "Unrepresentable residual changed token output");
}
void PreparedTokenHandoff() {
  for (bool local : {false, true}) {
    Fixture f;
    auto view = f.View(local);
    PreparedModularStreamTokens cpu;
    Ok(TokenizePreparedModularStreams(view, &cpu));
    for (const auto &variant : test::kPreparedEntropyCases) {
      CodestreamBuffer expected, actual;
      Ok(EncodePreparedModularStreams(view, variant.options, &expected));
      auto borrowed = view;
      auto streams = f.streams;
      for (size_t i = 0; i < streams.size(); ++i)
        if (cpu.streams[i].size())
          streams[i].completed_tokens = PreparedModularStreamTokenView{
              cpu.streams[i], streams[i].channels,
              streams[i].local_tree ? streams[i].local_tree : view.global_tree,
              streams[i].weighted, i};
      borrowed.streams = streams;
      PreparedModularStreamTokens handoff;
      Ok(TokenizePreparedModularStreams(borrowed, &handoff));
      Check(handoff.tokens.empty() && handoff.streams[0].tokens.data() == cpu.tokens.data(),
            "Prepared token handoff copied or regenerated data tokens");
      Ok(EncodePreparedModularStreams(borrowed, variant.options, &actual));
      Check(std::ranges::equal(expected.view(), actual.view()),
            "Prepared token handoff changed bytes");
    }
    // The canonical DC slot has no tokens.
    f.streams[1].completed_tokens = PreparedModularStreamTokenView{
        cpu.streams[0], f.streams[1].channels, &f.tree, f.streams[1].weighted, 1};
    ModularWorkflowStoragePlan plan;
    ArmNextManagedHostAllocationFailureForTest();
    const auto status =
        ComputePreparedModularStreamsStoragePlan(view, EntropyCodingMode::kPrefix, &plan);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(status.code() == StatusCode::kInvalidArgument && pending,
          "Nonempty tokens for an empty stream were accepted");
  }
}
void InvalidOptions() {
  for (size_t error = 0; error < 6; ++error) {
    Fixture f;
    const auto view = f.View(false);
    PreparedModularEntropyOptions options{.policy = PreparedModularEntropyPolicy::kDirectRateAns};
    if (error == 0)
      options.policy = static_cast<PreparedModularEntropyPolicy>(99);
    if (error == 1)
      options.selection = static_cast<AnsSelectionMode>(99);
    if (error == 2)
      options.maximum_ans_clusters = 0;
    if (error == 3)
      options.maximum_ans_clusters = kMaximumAnsClusters + 1;
    if (error >= 4) {
      options.policy = error == 4 ? PreparedModularEntropyPolicy::kPrefix
                                  : PreparedModularEntropyPolicy::kAnsFromPrefix;
      options.maximum_ans_clusters = 64;
    }
    ModularWorkflowStoragePlan plan;
    plan.tokens = 73;
    const auto original = plan;
    CodestreamBuffer output;
    const std::array<uint8_t, 3> sentinel = {7, 8, 9};
    Ok(CodestreamBuffer::CopyFrom(sentinel, &output));
    ArmNextManagedHostAllocationFailureForTest();
    const auto p = ComputePreparedModularStreamsStoragePlan(view, options, &plan);
    const auto e = EncodePreparedModularStreams(view, options, &output);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(p.code() == StatusCode::kInvalidArgument && e.code() == StatusCode::kInvalidArgument &&
              pending && plan == original && std::ranges::equal(output.view(), sentinel),
          "Invalid entropy options allocated or published output");
  }
}
} // namespace
int main() try {
  Validation();
  InvalidOptions();
  PreparedTokenHandoff();
  for (bool local : {false, true})
    for (const auto &variant : test::kPreparedEntropyCases)
      AllocationSweep(local, variant);
  std::cout << "Prepared Modular stream contracts passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
