// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codec/modular/tree_learning.h"
#include "codestream/modular/search.h"
#include "codestream/modular/workflow.h"
#include "gpu/ops/modular_token_storage_plan.h"
#include <iostream>
#include <stdexcept>
#ifdef GJXL_TRAINING_BUDGET_REFERENCE
#include "modular_reference.h"
#endif

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool v, const char *message) {
  if (!v)
    throw std::runtime_error(message);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
void Empty(const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  Check(!s.live_capacity_bytes && !s.reserved_unbacked_bytes && !s.active_reservations &&
            !s.active_cpu_participants,
        "Budget workflow leaked resources");
}
DecisionTree Balanced(size_t leaves) {
  DecisionTree tree;
  tree.nodes.resize(2 * leaves - 1);
  tree.size = 2 * leaves - 1;
  size_t next = 1;
  auto fill = [&](auto &&self, size_t node, size_t begin, size_t end) -> void {
    if (end - begin == 1)
      return;
    const size_t middle = (begin + end) / 2;
    const auto left = static_cast<TreeIndex>(next++), right = static_cast<TreeIndex>(next++);
    tree.nodes[node] = {
        .property = 3, .split = static_cast<int32_t>(middle - 1), .left = left, .right = right};
    self(self, left, middle, end);
    self(self, right, begin, middle);
  };
  fill(fill, 0, 0, leaves);
  return tree;
}
void ValidationAndCapacity() {
  HostStorageBound bound;
  for (auto options : {TreeLearningOptions{0, 16},
                       {4096, 0},
                       {kMaximumTrainingSamples + 1, 16},
                       {4096, kMaximumTreeLeaves + 1},
                       {SIZE_MAX, SIZE_MAX}}) {
    const auto before = bound;
    DecisionTree tree, saved = tree;
    const std::array<TrainingSample, 1> sample{};
    ArmNextManagedHostAllocationFailureForTest();
    Check(!ComputeTreeLearningStorageBound(&bound, options).ok() && bound == before &&
              !LearnEntropyTree(sample, &tree, options).ok() && tree == saved &&
              ManagedHostAllocationFailurePendingForTest(),
          "Invalid budget allocated or published");
    DisarmManagedHostAllocationFailureForTest();
  }
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeTreeLearningStorageBound(&bound, {kMaximumTrainingSamples, kMaximumTreeLeaves}));
  ModularWorkflowStoragePlan plan;
  Ok(ComputeModularSearchStoragePlan({257, 257}, PackedModularFormat::kGray16,
                                     EntropyCodingMode::kAns, &plan, 1,
                                     {kMaximumTrainingSamples, kMaximumTreeLeaves}));
  Check(ManagedHostAllocationFailurePendingForTest(), "Large budget planner allocated");
  DisarmManagedHostAllocationFailureForTest();
  auto tree = Balanced(kMaximumTreeLeaves);
  TreeLayout layout;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ValidateTree(tree, &layout));
  Check(layout.leaves == kMaximumTreeLeaves && ManagedHostAllocationFailurePendingForTest(),
        "Large validation allocated or narrowed leaf count");
  DisarmManagedHostAllocationFailureForTest();
  PredictionProperties properties{};
  Check(layout.context[Lookup(tree, properties)] == kMaximumTreeLeaves - 1,
        "Context index was narrowed");
  // A maximum-depth chain must validate without recursive stack growth.
  tree = {};
  tree.size = 2 * kMaximumTreeDepth - 1;
  tree.nodes.resize(tree.size);
  for (size_t i = 0; i + 1 < kMaximumTreeDepth; ++i)
    tree.nodes[2 * i] = {.property = 3,
                         .split = static_cast<int32_t>(i),
                         .left = static_cast<TreeIndex>(2 * i + 2),
                         .right = static_cast<TreeIndex>(2 * i + 1)};
  for (size_t i = 1; i < tree.size; i += 2)
    tree.nodes[i] = {};
  tree.nodes[tree.size - 1] = {};
  Ok(ValidateTree(tree, &layout));
  const auto saved_leaves = layout.leaves;
  tree.nodes[tree.size - 3].split = 0;
  Check(!ValidateTree(tree, &layout).ok() && layout.leaves == saved_leaves,
        "Deep contradictory split accepted or output changed");
  tree.nodes[tree.size - 3].split = static_cast<int32_t>(kMaximumTreeDepth - 2);
  tree.nodes.resize(tree.size + 2);
  tree.nodes[tree.size - 1] = {.property = 3,
                               .split = static_cast<int32_t>(kMaximumTreeDepth - 1),
                               .left = static_cast<TreeIndex>(tree.size),
                               .right = static_cast<TreeIndex>(tree.size + 1)};
  tree.size += 2;
  Check(!ValidateTree(tree, &layout).ok(), "Decoder-incompatible tree depth accepted");
  tree = Balanced(1024);
  ModularCodingPolicy oversized;
  oversized.tree = tree;
  ArmNextManagedHostAllocationFailureForTest();
  Check(ComputeModularWorkflowStoragePlan({17, 19}, PackedModularFormat::kGray8,
                                          EntropyCodingMode::kPrefix, oversized, &plan)
                    .code() == StatusCode::kUnsupported &&
            ManagedHostAllocationFailurePendingForTest(),
        "Image-dependent tree bound ignored");
  DisarmManagedHostAllocationFailureForTest();
  tree = Balanced(256);
  tree.nodes[0].left = 0;
  Check(!ValidateTree(tree, &layout).ok(), "Large cyclic tree accepted");
  tree = Balanced(64);
  tree.size = tree.nodes.size() + 2;
  Check(!ValidateTree(tree, &layout).ok(), "Unbacked nodes accepted");

  using namespace gjxl::modular_gpu_internal;
  ModularTokenizationOptions options;
  options.coding.tree = Balanced(16);
  ModularTokenEligibility eligibility;
  Ok(ResolveModularTokenEligibility(options, &eligibility));
  const auto previous = eligibility;
  options.coding.tree.nodes.resize(127);
  ArmNextManagedHostAllocationFailureForTest();
  const auto compact = options;
  Ok(ResolveModularTokenEligibility(compact, &eligibility));
  Check(compact.coding.tree.nodes.size() == kInlineTreeNodes && eligibility == previous &&
            ManagedHostAllocationFailurePendingForTest(),
        "Unused learning capacity escaped policy copy");
  DisarmManagedHostAllocationFailureForTest();
  options.coding.tree = Balanced(64);
  ArmNextManagedHostAllocationFailureForTest();
  Check(ResolveModularTokenEligibility(options, &eligibility).code() == StatusCode::kUnsupported &&
            eligibility == previous && ManagedHostAllocationFailurePendingForTest(),
        "Large CPU tree escaped GPU capacity or changed output");
  DisarmManagedHostAllocationFailureForTest();
}
void Learning() {
  // Both the exact uint16 histogram overflow boundary and the first sample
  // whose index cannot fit uint16 are exercised with full histogram populations.
  for (size_t count : {size_t{65536}, size_t{65537}}) {
    std::vector<TrainingSample> samples(count);
    for (auto &s : samples)
      s.residuals.fill(count == 65536 ? 0 : INT32_MIN);
    DecisionTree tree;
    Ok(LearnEntropyTree(samples, &tree, {count, 64}));
    Check(tree.size == 1 && tree.nodes[0].predictor == Predictor::kGradient,
          "Large constant histogram changed the predictor or split");
  }
  std::vector<TrainingSample> samples(32768);
  for (size_t i = 0; i < samples.size(); ++i) {
    const size_t region = i / 128;
    samples[i].properties[0] = static_cast<int64_t>(region);
    for (size_t p = 0; p < 14; ++p)
      samples[i].residuals[p] = static_cast<int32_t>((i * 977 + p * 131) % 65536);
    samples[i].residuals[region % 14] = int32_t{1} << (region % 25);
  }
  for (size_t leaves : {size_t{16}, size_t{64}, size_t{256}}) {
    DecisionTree tree, again;
    Ok(LearnEntropyTree(samples, &tree, {samples.size(), leaves}));
    Ok(LearnEntropyTree(samples, &again, {samples.size(), leaves}));
    TreeLayout layout;
    Ok(ValidateTree(tree, &layout));
    std::cout << "learned " << samples.size() << " samples, " << layout.leaves << " leaves\n";
    Check(tree == again && layout.leaves <= leaves &&
              layout.leaves >= std::min(leaves, size_t{129}),
          "Learning exceeded capacity, narrowed indices or was nondeterministic");
  }
  // Every new learner backing allocation: failure leaves a previously large
  // output unchanged; successful move publication retains only the node backing.
  samples.resize(64);
  const TreeLearningOptions options{65536, 64};
  HostStorageBound bound;
  Ok(ComputeTreeLearningStorageBound(&bound, options));
  size_t failures = 0;
  for (; failures < 16; ++failures) {
    auto tree = Balanced(64), before = tree;
    ResourceBudget budget(bound.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.peak_bytes, &reservation));
    Status status;
    bool pending;
    {
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      ArmManagedHostAllocationFailureAfterForTest(failures);
      status = LearnEntropyTree(samples, &tree, options);
      pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
    }
    Check(budget.snapshot().peak_backing_bytes <= bound.peak_bytes, "Learning exceeded admission");
    if (!status.ok())
      Check(!pending && status.code() == StatusCode::kOutOfMemory && tree == before,
            "Learning failure was not atomic");
    tree = {};
    Check(budget.snapshot().total.backing_count == 0, "Learning backing leaked");
    if (status.ok()) {
      Check(pending, "Successful learning consumed failure checkpoint");
      break;
    }
  }
  Check(failures == 6, "New learner allocation sweep did not cover all six backings");
  // Small input with the maximum capacity is also legal; no policy forces splits.
  DecisionTree tree;
  Ok(LearnEntropyTree(std::span(samples).first(1), &tree, {1, kMaximumTreeLeaves}));
  Check(tree.size == 1, "Tree budget forced useless leaves");
}
void QuantizedLearning() {
  std::vector<TrainingSample> samples(65537);
  for (size_t i = 0; i < samples.size(); ++i) {
    samples[i].properties[0] = i / 512;
    samples[i].properties[4] = i & 1 ? INT64_MIN : INT64_MAX;
    samples[i].residuals.fill(INT32_MIN);
    const size_t preferred =
        (i / 512) % 2 ? size_t(Predictor::kWeighted) : size_t(Predictor::kGradient);
    samples[i].residuals[preferred] = static_cast<int32_t>((i / 512) * 2);
  }
  const TreeLearningOptions options{65537, 64, TreeLearningMode::kQuantized};
  DecisionTree learned, again;
  Ok(LearnEntropyTree(samples, &learned, options, samples.size() * 10));
  Ok(LearnEntropyTree(samples, &again, options, samples.size() * 10));
  Check(learned == again && learned.size == 127, "Quantized capacity or determinism differs");
  for (size_t i = 0; i < learned.size; ++i)
    if (learned.nodes[i].property < 0)
      Check(learned.nodes[i].predictor == Predictor::kWeighted ||
                learned.nodes[i].predictor == Predictor::kGradient,
            "Unexpected quantized predictor");
  const auto saved = learned;
  ArmNextManagedHostAllocationFailureForTest();
  auto invalid = options;
  invalid.mode = static_cast<TreeLearningMode>(99);
  HostStorageBound bound;
  Check(!LearnEntropyTree(samples, &learned, options, samples.size() - 1).ok() &&
            !LearnEntropyTree(samples, &learned, invalid).ok() &&
            !ComputeTreeLearningStorageBound(&bound, invalid).ok() && learned == saved &&
            ManagedHostAllocationFailurePendingForTest(),
        "Invalid quantized learning allocated");
  DisarmManagedHostAllocationFailureForTest();
  Ok(ComputeTreeLearningStorageBound(&bound, options));
  samples.resize(64);
  size_t failures = 0;
  for (; failures < 16; ++failures) {
    ResourceBudget budget(bound.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.peak_bytes, &reservation));
    Status status;
    bool pending;
    DecisionTree tree = saved;
    {
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      ArmManagedHostAllocationFailureAfterForTest(failures);
      status = LearnEntropyTree(samples, &tree, options);
      pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
    }
    Check(budget.snapshot().peak_backing_bytes <= bound.peak_bytes, "Quantized admission exceeded");
    if (!status.ok())
      Check(!pending && tree == saved && status.code() == StatusCode::kOutOfMemory,
            "Quantized allocation failure was not atomic");
    tree = {};
    Check(budget.snapshot().total.backing_count == 0, "Quantized backing leaked");
    if (status.ok()) {
      Check(pending, "Failure sweep missed a backing");
      break;
    }
  }
  Check(failures == 7, "Quantized learner allocation count changed");
  std::cout << "Quantized learner: 65537 samples, 64 leaves, seven allocation failures checked\n";
}
void Encodes(TreeLearningMode learner = TreeLearningMode::kLegacy) {
  const Extent2D extent{257, 65};
  std::vector<uint8_t> bytes(extent.width * extent.height * 2);
  for (size_t i = 0; i < bytes.size() / 2; ++i) {
    const auto value = static_cast<uint16_t>(i * 71 + (i % 17) * 511);
    bytes[2 * i] = static_cast<uint8_t>(value);
    bytes[2 * i + 1] = static_cast<uint8_t>(value >> 8);
  }
  PackedModularImageView view{bytes, extent, 2 * extent.width, PackedModularFormat::kGray16};
  for (bool search : {false, true})
    for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns})
      for (size_t leaves : {size_t{64}, size_t{256}, size_t{1024}}) {
        if ((learner == TreeLearningMode::kQuantized && (!search || leaves != 64)) ||
            (search && leaves == 1024))
          continue;
        std::cout << "encode search " << search << " mode " << static_cast<int>(mode) << " leaves "
                  << leaves << std::endl;
        ModularEncodingOptions options;
        options.entropy = mode;
        options.search = search;
        options.learning = {65536, leaves, learner};
        if (!search)
          options.coding.tree = Balanced(leaves);
        ModularWorkflowStoragePlan plan;
        ArmNextManagedHostAllocationFailureForTest();
        Ok(search ? ComputeModularSearchStoragePlan(extent, view.format, mode, &plan, 1,
                                                    options.learning)
                  : ComputeModularWorkflowStoragePlan(extent, view.format, mode, options.coding,
                                                      &plan));
        Check(ManagedHostAllocationFailurePendingForTest(), "Encode planner allocated");
        DisarmManagedHostAllocationFailureForTest();
        Ok(ExecutionDomain::Create(
            {.managed_memory_bytes = plan.working.peak_bytes, .cpu_participant_limit = 1},
            &options.execution_domain));
        std::vector<uint8_t> encoded, again;
        Ok(EncodeModularImage(view, options, &encoded));
        Empty(*options.execution_domain);
        Check(options.execution_domain->snapshot().peak_backing_bytes <= plan.working.peak_bytes &&
                  encoded.size() <= plan.maximum_codestream_bytes,
              "Encode exceeded its storage plan");
#ifdef GJXL_TRAINING_BUDGET_REFERENCE
        const auto decoded = test::modular_reference::DecodeLossless(encoded);
        Check(decoded.width == extent.width && decoded.height == extent.height &&
                  decoded.bits == 16 && decoded.channels == 1 &&
                  decoded.samples.size() * 2 == bytes.size(),
              "Large tree decode metadata differs");
        for (size_t i = 0; i < decoded.samples.size(); ++i)
          Check(decoded.samples[i] == (bytes[2 * i] | (uint16_t{bytes[2 * i + 1]} << 8)),
                "Large tree decode sample differs");
#endif
        // Parallel tokenization/emission must preserve complete bytes.
        options.cpu_thread_count = 4;
        Ok(search ? ComputeModularSearchStoragePlan(extent, view.format, mode, &plan, 4,
                                                    options.learning)
                  : ComputeModularWorkflowStoragePlan(extent, view.format, mode, options.coding,
                                                      &plan, 4));
        Ok(ExecutionDomain::Create(
            {.managed_memory_bytes = plan.working.peak_bytes, .cpu_participant_limit = 4},
            &options.execution_domain));
        Ok(EncodeModularImage(view, options, &again));
        Check(encoded == again, "Large tree serial/parallel bytes differ");
        Empty(*options.execution_domain);
        std::shared_ptr<const ExecutionDomain> too_small;
        Ok(ExecutionDomain::Create({.managed_memory_bytes = plan.working.peak_bytes - 1},
                                   &too_small));
        options.execution_domain = too_small;
        ArmNextManagedHostAllocationFailureForTest();
        Check(!EncodeModularImage(view, options, &again).ok() && encoded == again &&
                  ManagedHostAllocationFailurePendingForTest(),
              "Failed admission allocated or published");
        DisarmManagedHostAllocationFailureForTest();
        Empty(*too_small);
      }
}
void TrainingPolicies(TreeLearningMode learner = TreeLearningMode::kLegacy) {
  const Extent2D extent{257, 65};
  std::vector<uint8_t> bytes(extent.width * extent.height * 4);
  for (size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<uint8_t>((i / 4) % 31 + i % 4);
  PackedModularImageView view{bytes, extent, extent.width * 4, PackedModularFormat::kRgba8};
  for (size_t kind = 0; kind < 3; ++kind) {
    ModularCodingPolicy base;
    if (kind) {
      base.transforms.size = 1;
      base.transforms.entries[0] =
          kind == 1 ? TransformDescriptor{TransformKind::kPalette, 0, 4, 31}
                    : TransformDescriptor{TransformKind::kSqueeze, 0, 4, 1, true, false};
    }
    ModularEncoderFrame frame;
    Ok(ModularEncoderFrame::Prepare(view, base, &frame));
    ChannelShape shape;
    Ok(DescribeImage(frame.image(), &shape));
    ModularFrameGeometry geometry;
    Ok(ModularFrameGeometry::Create(extent, &geometry));
    ModularStreamPlan layout;
    Ok(BuildModularStreamPlan(geometry, shape.channels(), shape.metadata, &layout));
    ModularCodingPolicy single, split, old, new_single, new_split, learned;
    Ok(LearnModularPolicies(frame, layout, base, &single, &split, &old));
    TrainingCoverage coverage, again;
    Ok(LearnModularPolicies(frame, layout, base, &new_single, &new_split, &learned,
                            {65537, 64, learner}, &coverage));
    auto saved = learned;
    Ok(LearnModularPolicies(frame, layout, base, &new_single, &new_split, &learned,
                            {65537, 64, learner}, &again));
    Check(coverage == again && saved == learned && coverage.samples <= 65537 &&
              coverage.samples <= coverage.population &&
              coverage.slices == coverage.sampled_slices &&
              coverage.slice_floor == (learner == TreeLearningMode::kQuantized ? 1024 : 1),
          "Sampling coverage was not deterministic and bounded");
    for (size_t budget : {size_t{1}, size_t{17}, size_t{4096}}) {
      ModularCodingPolicy a, b, tree;
      TrainingCoverage small;
      Ok(LearnModularPolicies(frame, layout, base, &a, &b, &tree, {budget, 16, learner}, &small));
      Check(small.samples == std::min(budget, small.population) && a == single && b == split &&
                small.sampled_slices <= small.slices &&
                (budget < small.slices || small.sampled_slices == small.slices),
            "Small sampling budget exceeded coverage or changed retained candidates");
    }
    Check(single == new_single && split == new_split && learned.transforms == base.transforms,
          "New entropy budget changed retained candidates or transforms");
    std::vector<uint8_t> output;
    Ok(EncodeModularImage(view, {.entropy = EntropyCodingMode::kAns, .coding = learned}, &output));
#ifdef GJXL_TRAINING_BUDGET_REFERENCE
    const auto decoded = test::modular_reference::DecodeLossless(output);
    Check(decoded.samples.size() == bytes.size() && decoded.channels == 4 && decoded.bits == 8,
          "Transformed learned tree metadata differs");
    for (size_t i = 0; i < bytes.size(); ++i)
      Check(decoded.samples[i] == bytes[i], "Transformed learned tree samples differ");
#endif
  }
}
void AllocationFailures(TreeLearningMode learner = TreeLearningMode::kLegacy) {
  const std::array<uint8_t, 12> bytes{0, 3, 1, 4, 5, 9, 2, 6, 7, 0, 8, 3};
  const PackedModularImageView view{bytes, {3, 4}, 3, PackedModularFormat::kGray8};
  for (bool search : {false, true})
    for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
      if (learner == TreeLearningMode::kQuantized && !search)
        continue;
      ModularEncodingOptions options;
      options.entropy = mode;
      options.search = search;
      options.learning = {65537, 64, learner};
      if (!search)
        options.coding.tree = Balanced(64);
      ModularWorkflowStoragePlan plan;
      Ok(search ? ComputeModularSearchStoragePlan(view.extent, view.format, mode, &plan, 1,
                                                  options.learning)
                : ComputeModularWorkflowStoragePlan(view.extent, view.format, mode, options.coding,
                                                    &plan));
      Ok(ExecutionDomain::Create(
          {.managed_memory_bytes = plan.working.peak_bytes, .cpu_participant_limit = 1},
          &options.execution_domain));
      std::vector<uint8_t> expected;
      size_t allocations;
      {
        ModularProfile profile;
        ProfileSession session(&profile);
        constexpr size_t sentinel = SIZE_MAX - 1;
        ArmManagedHostAllocationFailureAfterForTest(sentinel);
        Ok(EncodeModularImage(view, options, &expected));
        allocations = sentinel - detail::managed_host_backings_before_failure;
        DisarmManagedHostAllocationFailureForTest();
      }
      Empty(*options.execution_domain);
      Check(allocations > 6 && allocations < 75000, "Unexpected large-budget allocation count");
      for (size_t fail = 0; fail < allocations; ++fail) {
        std::vector<uint8_t> output{7, 8, 9};
        {
          ModularProfile profile;
          ProfileSession session(&profile);
          ArmManagedHostAllocationFailureAfterForTest(fail);
          const auto status = EncodeModularImage(view, options, &output);
          const bool pending = ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
          Check(!pending && status.code() == StatusCode::kOutOfMemory &&
                    output == std::vector<uint8_t>({7, 8, 9}),
                "Large-budget failure published or escaped");
        }
        Empty(*options.execution_domain);
      }
      std::vector<uint8_t> again;
      Ok(EncodeModularImage(view, options, &again));
      Check(expected == again, "Large-budget failure recovery changed bytes");
      Empty(*options.execution_domain);
      std::cout << "allocation failures search " << search << " mode " << static_cast<int>(mode)
                << ": " << allocations << std::endl;
    }
}
} // namespace
int main(int argc, char **argv) try {
  if (argc == 2 && std::string_view(argv[1]) == "--coverage") {
    QuantizedLearning();
    TrainingPolicies(TreeLearningMode::kQuantized);
    Encodes(TreeLearningMode::kQuantized);
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--coverage-allocations") {
    AllocationFailures(TreeLearningMode::kQuantized);
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--allocations") {
    AllocationFailures();
    return 0;
  }
  Check(argc == 1, "Unknown training budget test argument");
  ValidationAndCapacity();
  Learning();
  TrainingPolicies();
  Encodes();
  std::cout << "Modular configurable training budget checks passed\n";
} catch (const std::exception &e) {
  DisarmManagedHostAllocationFailureForTest();
  std::cerr << e.what() << '\n';
  return 1;
}
