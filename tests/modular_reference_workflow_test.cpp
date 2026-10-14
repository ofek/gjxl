// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/reference_workflow.h"
#include "modular_reference_workflow_fixture.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
using gjxl::codestream_internal::CodestreamBuffer;
namespace reference = gjxl::test::modular_reference;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
using test::modular_workflow_fixture::Fixture;
using test::modular_workflow_fixture::Make;
void Released(const ResourceBudget &budget) {
  const auto s = budget.snapshot();
  Check(!s.committed_bytes() && !s.total.backing_count &&
            !s.total.pending_count && !s.open_reservations &&
            !s.waiting_requests,
        "Workflow leaked managed resources");
}
void Decode(const Fixture &f, const CodestreamBuffer &bytes) {
#ifdef GJXL_REFERENCE_WORKFLOW_ORACLE
  Check(reference::DecodeLossless(bytes.view()) == f.original,
        "Workflow exact decode differs");
#endif
}
void EncodePrepared(const ReferenceModularFrame &frame,
                    const ReferenceWorkflowOptions &options,
                    const ResourceBudget &budget, CodestreamBuffer *out) {
  ModularWorkflowStoragePlan plan;
  Ok(ComputePreparedModularStreamsStoragePlan(frame.view(), options.entropy,
                                              &plan));
  ResourceReservation reservation;
  Ok(budget.TryReserve(plan.working.peak_bytes, &reservation));
  ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
  Ok(EncodePreparedModularStreams(frame.view(), options.entropy, out));
}
size_t files = 0, faults = 0;
void Case(Fixture f, ReferenceWorkflowOptions options,
          const std::filesystem::path &directory) {
  ResourceBudget budget(size_t{2} << 30);
  ReferenceModularFrame frame;
  Ok(PrepareReferenceModularFrame(f.view(), options, budget, &frame));
  ModularFrameGeometry geometry;
  Ok(ModularFrameGeometry::Create(f.view().extent, &geometry));
  const bool local = options.buffering == ReferenceBuffering::kAutomatic &&
                     geometry.group_count() > 8;
  const auto stats = frame.stats();
  Check(stats.local_trees == local &&
            stats.learning_calls == (local ? geometry.group_count() + 1 : 1) &&
            stats.fits + stats.empty_learning_calls == stats.learning_calls &&
            (!local || stats.empty_learning_calls == 1),
        "Workflow selected the wrong learning scope");
  Check(stats.selection_calls ==
            (geometry.group_count() == 1 ? 1 : geometry.group_count() + !local),
        "Workflow repeated or omitted selection");
  for (const auto &s : frame.streams()) {
    if (!s.learning.stop_group)
      continue;
    Check(s.tree.nodes.size() == std::max(s.tree.size, kInlineTreeNodes),
          "Workflow retained unused tree backing");
    Check(s.learning.first_group == 0 && (!local || s.learning.stop_group == 1),
          "Local learning used the coding stream ID");
  }
  const auto *records = frame.streams().data();
  ReferenceModularFrame moved(std::move(frame));
  frame = std::move(moved);
  Check(frame.streams().data() == records,
        "Moving a frame changed referenced addresses");
  CodestreamBuffer expected;
  EncodePrepared(frame, options, budget, &expected);
  Decode(f, expected);
  // Input lifetime ends here. The prepared frame must own its samples.
  std::fill(f.bytes.begin(), f.bytes.end(), 0);
  CodestreamBuffer detached;
  EncodePrepared(frame, options, budget, &detached);
  Check(std::ranges::equal(expected.view(), detached.view()),
        "Prepared frame borrowed packed input");
  Decode(f, detached);
  std::ofstream file(directory /
                         ("workflow-" + std::to_string(files++) + ".jxl"),
                     std::ios::binary);
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.write(reinterpret_cast<const char *>(expected.view().data()),
             expected.size());
  file.close();
  frame = {};
  expected = {};
  detached = {};
  Released(budget);
}
void InvalidAndBusy() {
  const auto f = Make({5, 3}, 3, 8, 1);
  ResourceBudget budget(size_t{2} << 30);
  ReferenceModularFrame output;
  Ok(PrepareReferenceModularFrame(f.view(), {}, budget, &output));
  const auto *old = output.streams().data();
  const size_t baseline = budget.snapshot().committed_bytes();
  for (unsigned i = 0; i < 8; ++i) {
    auto input = f.view();
    ReferenceWorkflowOptions options;
    if (i == 0)
      options.buffering = static_cast<ReferenceBuffering>(255);
    if (i == 1)
      options.predictors = static_cast<ReferencePredictors>(255);
    if (i == 2)
      options.maximum_samples = 0;
    if (i == 3)
      options.maximum_samples = kHistogramMaximumSamples + 1;
    if (i == 4)
      options.entropy.maximum_ans_clusters = 0;
    if (i == 5)
      input.extent.width = 0;
    if (i == 6)
      input.row_stride = 0;
    if (i == 7)
      input.bytes = {};
    ArmNextManagedHostAllocationFailureForTest();
    const auto s =
        PrepareReferenceModularFrame(input, options, budget, &output);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(!s.ok() && pending && output.streams().data() == old &&
              budget.snapshot().committed_bytes() == baseline,
          "Invalid input allocated or changed output");
  }
  ResourceBudget tiny(1), busy(size_t{2} << 30);
  ResourceReservation hold;
  Ok(busy.TryReserve(size_t{2} << 30, &hold));
  Check(PrepareReferenceModularFrame(f.view(), {}, tiny, &output).code() ==
                StatusCode::kOutOfMemory &&
            PrepareReferenceModularFrame(f.view(), {}, busy, &output).code() ==
                StatusCode::kUnavailable &&
            output.streams().data() == old,
        "Admission did not fail atomically without waiting");
  ReferenceWorkflowOptions cap;
  cap.maximum_samples = 1;
  Check(!PrepareReferenceModularFrame(f.view(), cap, budget, &output).ok() &&
            output.streams().data() == old &&
            budget.snapshot().committed_bytes() == baseline,
        "Sample cap truncated or published an incomplete result");
  Check(!PrepareReferenceModularFrame(f.view(), {}, budget, nullptr).ok(),
        "Null prepare output accepted");
  Check(!EncodeReferenceModularImage(f.view(), {}, budget, nullptr).ok(),
        "Null encode output accepted");
  output = {};
  Released(budget);
  Released(tiny);
  hold.Reset();
  Released(busy);
}
void FaultSweep(const Fixture &f, bool prepare_only) {
  ResourceBudget budget(size_t{2} << 30);
  const auto old_input = Make({1, 1}, 1, 8, 0);
  ReferenceModularFrame prepared;
  Ok(PrepareReferenceModularFrame(old_input.view(), {}, budget, &prepared));
  const auto *old = prepared.streams().data();
  const auto old_tree = prepared.streams()[0].tree;
  const auto old_pixels = prepared.streams()[0].channels[0].backing;
  const auto baseline = budget.snapshot().committed_bytes();
  const std::array<uint8_t, 3> sentinel{7, 9, 11};
  CodestreamBuffer expected;
  Ok(EncodeReferenceModularImage(f.view(), {}, budget, &expected));
  Decode(f, expected);
  bool success = false;
  for (size_t i = 0; i < 10000; ++i) {
    CodestreamBuffer output;
    Ok(CodestreamBuffer::CopyFrom(sentinel, &output));
    ReferenceWorkflowStats stats;
    stats.fits = 777;
    const size_t retained = budget.snapshot().committed_bytes();
    ArmManagedHostAllocationFailureAfterForTest(i);
    const auto s =
        prepare_only
            ? PrepareReferenceModularFrame(f.view(), {}, budget, &prepared)
            : EncodeReferenceModularImage(f.view(), {}, budget, &output,
                                          &stats);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (s.ok()) {
      Check(pending, "An allocation failure was ignored");
      if (prepare_only)
        EncodePrepared(prepared, {}, budget, &output);
      Check(std::ranges::equal(expected.view(), output.view()),
            "Recovery changed codestream bytes");
      success = true;
      break;
    }
    ++faults;
    Check(!pending && s.code() == StatusCode::kOutOfMemory &&
              std::ranges::equal(output.view(), sentinel) &&
              stats.fits == 777 && prepared.streams().data() == old &&
              prepared.streams()[0].tree == old_tree &&
              prepared.streams()[0].channels[0].backing.data() ==
                  old_pixels.data() &&
              budget.snapshot().committed_bytes() == retained &&
              !budget.snapshot().total.pending_count,
          "Allocation failure changed output, stats, or retained capacity");
  }
  Check(success && baseline != 0, "Allocation failure sweep did not finish");
  prepared = {};
  expected = {};
  Released(budget);
}
void WrapperAlias() {
  const auto f = Make({17, 9}, 4, 16, 3, SampleByteOrder::kLittleEndian);
  ResourceBudget budget(size_t{2} << 30);
  CodestreamBuffer output;
  Ok(CodestreamBuffer::CopyFrom(f.bytes, &output));
  auto input = f.view();
  input.bytes = output.view();
  ReferenceWorkflowStats stats;
  Ok(EncodeReferenceModularImage(input, {}, budget, &output, &stats));
  Decode(f, output);
  Check(stats.learning_calls == 1 && stats.fits == 1,
        "Wrapper repeated training");
  output = {};
  Released(budget);
}
} // namespace
int main(int argc, char **argv) try {
  Check(argc == 2, "Expected artifact directory");
  const std::filesystem::path directory = argv[1];
  std::filesystem::create_directories(directory);
  InvalidAndBusy();
  WrapperAlias();
  for (size_t channels : {1, 3, 4})
    for (unsigned bits : {8, 16}) {
      for (unsigned shape = 0; shape < 4; ++shape) {
        const auto extent = shape == 0   ? Extent2D{1, 1}
                            : shape == 1 ? Extent2D{17, 9}
                            : shape == 2 ? Extent2D{2048, 1}
                                         : Extent2D{2049, 1};
        const auto order = shape % 2 ? SampleByteOrder::kLittleEndian
                                     : SampleByteOrder::kBigEndian;
        for (auto buffering :
             {ReferenceBuffering::kAutomatic, ReferenceBuffering::kBuffered})
          for (auto predictors :
               {ReferencePredictors::kBest, ReferencePredictors::kAll}) {
            ReferenceWorkflowOptions options;
            options.buffering = buffering;
            options.predictors = predictors;
            Case(Make(extent, channels, bits, shape, order), options,
                 directory);
          }
      }
    }
  // A two-dimensional partial group and a metadata-heavy buffered palette.
  Case(Make({257, 257}, 4, 8, 1), {}, directory);
  Case(Make({2049, 1}, 3, 16, 1), {ReferenceBuffering::kBuffered}, directory);
  for (bool prepare : {false, true}) {
    FaultSweep(Make({17, 9}, 4, 16, 1), prepare);
    FaultSweep(Make({2049, 1}, 1, 8, 0), prepare);
  }
  std::cout << "{\"files\":" << files << ",\"allocation_failures\":" << faults
#ifdef GJXL_REFERENCE_WORKFLOW_ORACLE
            << ",\"exact_decode\":true"
#else
            << ",\"exact_decode\":false"
#endif
            << "}\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
