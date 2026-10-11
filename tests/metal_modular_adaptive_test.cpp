// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/frame_encoder.h"
#include "codestream/modular/gpu_token_handoff.h"
#include "core/cpu_execution.h"
#include "core/thread_budget.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/metal/metal_modular_adaptive_tokens.h"
#include "gpu/ops/modular_adaptive_layout.h"
#if GJXL_MODULAR_ADAPTIVE_REFERENCE
#include "modular_reference.h"
#endif
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;
size_t cases = 0, checked_tokens = 0, checked_features = 0, checked_diagnostics = 0;
size_t checked_files = 0, decoded_samples = 0, different_ordinals = 0;
std::filesystem::path artifacts;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
GpuAdaptiveModularTokenGeneration &Operation(GpuBackend &gpu) {
  auto *op = QueryGpuAdaptiveModularTokenGeneration(gpu);
  Check(op != nullptr, "Adaptive token capability missing");
  return *op;
}
constexpr PackedModularFormat formats[] = {
    PackedModularFormat::kGray8,  PackedModularFormat::kRgb8,  PackedModularFormat::kRgba8,
    PackedModularFormat::kGray16, PackedModularFormat::kRgb16, PackedModularFormat::kRgba16};
struct Fixture {
  Extent2D extent;
  PackedModularFormat format;
  SampleByteOrder endian;
  size_t channels, bytes, stride;
  std::vector<uint8_t> packed;
  std::vector<uint16_t> samples;
  Fixture(PackedModularFormat f, Extent2D e = {17, 19}, bool palette = false,
          SampleByteOrder order = SampleByteOrder::kBigEndian)
      : extent(e), format(f), endian(order) {
    ModularInputProfile p;
    Ok(ResolveModularInput(e, f, &p));
    channels = p.channel_count;
    bytes = p.bytes_per_sample;
    stride = e.width * channels * bytes + 3;
    packed.resize(1 + (e.height - 1) * stride + e.width * channels * bytes, 0xad);
    samples.resize(e.width * e.height * channels);
    const uint32_t maximum = bytes == 1 ? 255 : 65535;
    const uint32_t edge[] = {0, 1, 2, maximum / 2, maximum - 1, maximum};
    uint32_t random = 0x912ac678;
    for (size_t y = 0; y < e.height; ++y)
      for (size_t x = 0; x < e.width; ++x)
        for (size_t c = 0, divisor = 1; c < channels; ++c, divisor *= 6) {
          random ^= random << 13;
          random ^= random >> 17;
          random ^= random << 5;
          const size_t pixel = y * e.width + x;
          uint32_t value = pixel < 216 ? edge[(pixel / divisor) % 6] : random & maximum;
          if (palette)
            value = (((x + 5 * y) % 16) * (19 + 6 * c)) & maximum;
          if (!palette && c == 3 && pixel % 3 == 0)
            value = 0;
          samples[pixel * channels + c] = static_cast<uint16_t>(value);
          const size_t i = 1 + y * stride + (x * channels + c) * bytes;
          packed[i] = static_cast<uint8_t>(
              bytes == 2 && order == SampleByteOrder::kBigEndian ? value >> 8 : value);
          if (bytes == 2)
            packed[i + 1] =
                static_cast<uint8_t>(order == SampleByteOrder::kBigEndian ? value : value >> 8);
        }
  }
  PackedModularImageView view() const {
    return {std::span(packed).subspan(1), extent, stride, format, endian};
  }
};
struct Prepared {
  ModularEncoderFrame frame;
  ChannelShape shape;
  ModularStreamPlan layout;
  ManagedVector<AdaptiveModularTokenSlice> slices;
  std::unique_ptr<PreparedGpuAdaptiveModularFeatures> features;
  Prepared() = default;
  void Plan() {
    Ok(DescribeImage(frame.image(), &shape));
    ModularFrameGeometry geometry;
    Ok(ModularFrameGeometry::Create(frame.metadata().extent, &geometry));
    Ok(BuildModularStreamPlan(geometry, shape.channels(), shape.metadata, &layout));
    Ok(BuildAdaptiveModularSlices(frame, &slices));
    // Independent canonical CPU planner: include empty streams and verify that
    // property 0 is the stream ordinal, not the physical transformed channel.
    size_t offset = 0, index = 0;
    std::array<size_t, kMaximumTransformedChannels> plane{};
    size_t plane_end = 0;
    for (size_t c = 0; c < shape.count; ++c) {
      plane[c] = (plane_end + 255) & ~size_t{255};
      plane_end = plane[c] + shape.descriptors[c].extent.width *
                                 shape.descriptors[c].extent.height * sizeof(int32_t);
    }
    for (const auto &stream : layout.streams)
      for (size_t ordinal = 0; ordinal < stream.slice_count; ++ordinal) {
        const auto &cpu = layout.slices[stream.slice_begin + ordinal];
        const auto &slice = slices.at(index++);
        const size_t stride = shape.descriptors[cpu.channel].extent.width;
        const AdaptiveModularTokenSlice expected{offset,
                                                 plane[cpu.channel] / sizeof(int32_t) +
                                                     cpu.rect.y * stride + cpu.rect.x,
                                                 stride,
                                                 static_cast<uint32_t>(stream.id),
                                                 static_cast<uint32_t>(ordinal),
                                                 static_cast<uint32_t>(cpu.rect.extent.width),
                                                 static_cast<uint32_t>(cpu.rect.extent.height),
                                                 {0, 0}};
        Check(slice == expected, "Adaptive descriptor differs from canonical CPU stream plan");
        different_ordinals += ordinal != cpu.channel;
        offset += cpu.rect.extent.width * cpu.rect.extent.height;
      }
    Check(index == slices.size(), "Adaptive descriptor count differs");
  }
  Prepared(GpuBackend &gpu, const Fixture &f, const ModularCodingPolicy &p, bool weighted = true) {
    Ok(ModularEncoderFrame::Prepare(f.view(), p, &frame));
    Plan();
    const auto submissions = gpu.stats().committed_submissions;
    Ok(Operation(gpu).PrepareAdaptiveModularFeatures(frame, slices, p.weighted, weighted,
                                                     &features));
    Check(gpu.stats().committed_submissions == submissions,
          "CPU feature bridge unexpectedly submitted GPU work");
  }
};
ModularCodingPolicy Split(unsigned property) {
  ModularCodingPolicy p;
  p.tree.size = 3;
  p.tree.nodes[0] = {.property = static_cast<int32_t>(property), .split = 0, .left = 2, .right = 1};
  p.tree.nodes[1].predictor = Predictor::kAverage4;
  p.tree.nodes[2].predictor = Predictor::kWeighted;
  p.tree.nodes[1].offset = -27;
  p.tree.nodes[2].offset = 19;
  return p;
}
DecisionTree Balanced(size_t leaves, unsigned property = 3) {
  DecisionTree tree;
  tree.size = 2 * leaves - 1;
  tree.nodes.resize(tree.size);
  const auto fill = [&](auto &&self, size_t i, int32_t lo, int32_t hi) -> void {
    if (lo == hi) {
      tree.nodes[i].predictor = static_cast<Predictor>((i - leaves + 1) % 14);
      return;
    }
    const auto mid = lo + (hi - lo) / 2;
    tree.nodes[i] = {.property = static_cast<int32_t>(property),
                     .split = mid,
                     .left = static_cast<TreeIndex>(2 * i + 1),
                     .right = static_cast<TreeIndex>(2 * i + 2)};
    self(self, 2 * i + 1, mid + 1, hi);
    self(self, 2 * i + 2, lo, mid);
  };
  fill(fill, 0, 0, static_cast<int32_t>(leaves - 1));
  TreeLayout layout;
  Ok(ValidateTree(tree, &layout));
  return tree;
}
void Equal(const PreparedModularTokens &cpu, const PreparedGpuModularTokens &gpu) {
  ModularTokenHostView view;
  Ok(gpu.BorrowHostView(&view));
  Check(view.values.size() == cpu.tokens.size() && view.contexts.size() == cpu.tokens.size() &&
            gpu.context_count() == cpu.context_count,
        "Adaptive token shape differs");
  for (size_t i = 0; i < cpu.tokens.size(); ++i)
    if (view.values[i] != cpu.tokens[i].value || view.contexts[i] != cpu.tokens[i].context)
      throw std::runtime_error("Adaptive token differs at " + std::to_string(i));
  checked_tokens += cpu.tokens.size();
}
void Tokens(GpuBackend &gpu, const Prepared &f, const ModularCodingPolicy &policy) {
  PreparedModularTokens cpu;
  Ok(TokenizeModular(f.frame, f.layout, policy, &cpu));
  std::unique_ptr<PreparedGpuModularTokens> tokens;
  const auto before = gpu.stats();
  Ok(Operation(gpu).GenerateAdaptiveModularTokens(*f.features, policy.tree, &tokens));
  Check(gpu.stats().committed_submissions == before.committed_submissions + 1,
        "Adaptive production did not submit exactly once");
  Equal(cpu, *tokens);
  ++cases;
}
void Diagnostics(GpuBackend &gpu, const Prepared &f, const ModularCodingPolicy &policy) {
  ManagedVector<metal_internal::AdaptiveModularDiagnostic> trace;
  Ok(metal_internal::DiagnoseMetalAdaptiveModularTokens(gpu, *f.features, policy.tree, &trace));
  std::span<const AdaptiveModularWeightedFeature> features;
  Ok(f.features->BorrowWeightedHostView(&features));
  Check(features.size() == f.features->token_count() && trace.size() == features.size(),
        "Adaptive diagnostic/feature shape differs");
  TreeLayout tree;
  Ok(ValidateTree(policy.tree, &tree));
  size_t slice_index = 0;
  for (const auto &stream : f.layout.streams)
    for (size_t ordinal = 0; ordinal < stream.slice_count; ++ordinal) {
      const auto &slice = f.layout.slices[stream.slice_begin + ordinal];
      const auto &descriptor = f.slices[slice_index++];
      ModularChannelView view;
      Ok(BorrowChannelSlice(f.frame.image().view(slice.channel), slice.rect, &view));
      WeightedPredictor<ResourceClass::kPreparation> wp(slice.rect.extent.width, policy.weighted);
      for (size_t y = 0; y < slice.rect.extent.height; ++y) {
        int64_t previous = 0;
        for (size_t x = 0; x < slice.rect.extent.width; ++x) {
          const auto n = Neighbors(view, x, y);
          const auto w = wp.Predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
          const size_t i = descriptor.token_offset + y * descriptor.width + x;
          Check(features[i] == AdaptiveModularWeightedFeature{w.first, w.second},
                "Weighted feature bridge differs from scalar recurrence");
          const auto properties = Properties(n, ordinal, stream.id, x, y, previous, w.second);
          previous = properties[9];
          const auto &actual = trace[i];
          for (size_t p = 0; p < kPropertyCount; ++p)
            Check(actual.properties[p] == properties[p], "Adaptive property differs");
          for (unsigned p = 0; p < 14; ++p)
            Check(actual.predictions[p] == Predict(static_cast<Predictor>(p), n, w.first),
                  "Adaptive predictor differs");
          const size_t leaf = Lookup(policy.tree, properties);
          const auto &node = policy.tree.nodes[leaf];
          const auto prediction = Predict(node.predictor, n, w.first);
          const auto residual =
              (int64_t{view.Row(y)[x]} - prediction - node.offset) / node.multiplier;
          Check(actual.leaf == leaf && actual.context == tree.context[leaf] &&
                    actual.sample == view.Row(y)[x] && actual.selected_prediction == prediction &&
                    actual.residual == residual &&
                    actual.value == PackSigned(static_cast<int32_t>(residual)),
                "Adaptive diagnostic leaf/residual differs");
          Check(wp.Update(view.Row(y)[x], x, y), "Scalar weighted range failure");
          ++checked_features;
          ++checked_diagnostics;
        }
      }
    }
}
void Consume(GpuBackend &gpu, const Fixture &source, const Prepared &f,
             const ModularCodingPolicy &policy) {
  for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    PreparedModularTokens cpu, split;
    Ok(TokenizeModular(f.frame, f.layout, policy, &cpu));
    std::unique_ptr<PreparedGpuModularTokens> owner;
    Ok(Operation(gpu).GenerateAdaptiveModularTokens(*f.features, policy.tree, &owner));
    auto *identity = owner.get();
    Ok(AdoptAdaptiveGpuModularTokens(f.layout, policy, std::move(owner), &split));
    Check(!owner && split.gpu_backing.get() == identity && split.tokens.empty(),
          "Adaptive handoff copied or lost completed backing");
    ModularWorkflowStoragePlan storage;
    Ok(ComputeModularWorkflowStoragePlan(source.extent, source.format, mode, policy, &storage));
    codestream_internal::CodestreamBuffer expected, actual;
    Ok(EncodeModularFrame(f.frame.metadata(), f.layout, std::move(cpu), mode, storage, &expected));
    Ok(EncodeModularFrame(f.frame.metadata(), f.layout, std::move(split), mode, storage, &actual));
    Check(std::ranges::equal(actual, expected), "Adaptive CPU continuation changed complete bytes");
    std::ofstream file(artifacts / ("case-" + std::to_string(checked_files++) + ".jxl"),
                       std::ios::binary);
    file.write(reinterpret_cast<const char *>(actual.data()), actual.size());
    Check(bool(file), "Could not retain adaptive token codestream");
#if GJXL_MODULAR_ADAPTIVE_REFERENCE
    namespace ref = gjxl::test::modular_reference;
    const ref::IntegerImage original{source.extent.width, source.extent.height,
                                     static_cast<uint32_t>(source.channels),
                                     static_cast<uint32_t>(source.bytes * 8), source.samples};
    Check(ref::DecodeLossless(actual.view()) == original, "Independent adaptive decode differs");
    decoded_samples += source.samples.size();
#endif
  }
}
void Matrix(GpuBackend &gpu) {
  for (auto format : formats) {
    Fixture source(format);
    auto policy = Split(15);
    policy.rct = source.channels >= 3 ? 12 : 0;
    policy.weighted.coefficients = {17, 9, 6, 8, 5, 1, 2};
    policy.weighted.weights = {12, 9, 10, 8};
    Prepared f(gpu, source, policy);
    Diagnostics(gpu, f, policy);
    Consume(gpu, source, f, policy);
    for (unsigned predictor = 0; predictor < 14; ++predictor) {
      auto p = policy;
      p.tree = {};
      p.tree.nodes[0].predictor = static_cast<Predictor>(predictor);
      Tokens(gpu, f, p);
    }
    for (unsigned property = 0; property < 16; ++property) {
      auto p = policy;
      p.tree = Split(property).tree;
      Tokens(gpu, f, p);
    }
  }
  for (auto extent : {Extent2D{1, 1}, {1, 257}, {257, 1}, {255, 256}, {256, 256}, {257, 259}}) {
    Fixture source(PackedModularFormat::kRgba16, extent, false, SampleByteOrder::kLittleEndian);
    auto p = Split(15);
    p.rct = 41;
    Prepared f(gpu, source, p);
    Tokens(gpu, f, p);
    Diagnostics(gpu, f, p);
  }
  // Palette metadata precedes grouped channels. Both Squeeze insertion orders
  // change physical channel identities; six reductions also exercise DC groups.
  for (auto format : {PackedModularFormat::kGray8, PackedModularFormat::kRgba16})
    for (unsigned transform = 0; transform < 5; ++transform) {
      Fixture source(format, transform == 4 ? Extent2D{2057, 17} : Extent2D{513, 259},
                     transform < 2);
      auto p = Split(0);
      p.rct = source.channels >= 3 ? 6 : 0;
      if (transform < 2) {
        p.transforms.size = transform ? 3 : 1;
        p.transforms.entries[0] = {TransformKind::kPalette, 0,
                                   static_cast<uint8_t>(source.channels), 16};
        p.transforms.entries[1] = {TransformKind::kSqueeze, 1, 1, 1, true, false};
        p.transforms.entries[2] = {TransformKind::kSqueeze, 1, 1, 1, false, true};
      } else {
        p.transforms.size = transform == 4 ? 6 : 1;
        for (size_t i = 0; i < p.transforms.size; ++i)
          p.transforms.entries[i] = {TransformKind::kSqueeze,
                                     0,
                                     static_cast<uint8_t>(source.channels),
                                     1,
                                     transform == 4 ? i % 2 == 0 : transform == 2,
                                     transform == 3};
      }
      Prepared f(gpu, source, p);
      Tokens(gpu, f, p);
      Diagnostics(gpu, f, p);
      Consume(gpu, source, f, p);
    }
  Check(different_ordinals != 0, "No transformed stream-local channel case exercised");
}
void LargeTrees(GpuBackend &gpu) {
  Fixture source(PackedModularFormat::kGray16, {512, 512});
  ModularCodingPolicy policy;
  Prepared f(gpu, source, policy);
  policy.tree = Balanced(256);
  Tokens(gpu, f, policy);
  Consume(gpu, source, f, policy);
  policy.tree = Balanced(kMaximumTreeLeaves);
  Tokens(gpu, f, policy);
  // Deep valid chain, unlike the balanced maximum-size fixture.
  constexpr size_t leaves = 1024;
  policy.tree.size = 2 * leaves - 1;
  policy.tree.nodes.resize(policy.tree.size);
  for (size_t i = 0; i < leaves - 1; ++i)
    policy.tree.nodes[i] = {.property = 15,
                            .split = static_cast<int32_t>(i) - 512,
                            .left = static_cast<TreeIndex>(i + 1),
                            .right = static_cast<TreeIndex>(leaves + i)};
  for (size_t i = leaves - 1; i < policy.tree.size; ++i)
    policy.tree.nodes[i] = {.predictor = Predictor::kWeighted};
  Tokens(gpu, f, policy);
}
void WideArithmetic(GpuBackend &gpu) {
  // Arithmetic-only synthetic transformed planes. These are not claimed to be
  // legal unpacked Gray16 sources and are never serialized or decoder-tested.
  const auto prepare = [&](bool even, Prepared *out) {
    ModularInputProfile profile;
    Ok(ResolveModularInput({5, 5}, PackedModularFormat::kGray16, &profile));
    ModularImage image;
    Ok(ModularImage::Create(profile.channels(), 0, &image));
    const int32_t values[] = {INT32_MIN, INT32_MAX - (even ? 1 : 0), -2, 2, 0};
    for (size_t i = 0; i < image.samples(0).size(); ++i)
      image.samples(0)[i] = values[(i * 3 + i / 5) % 5];
    // Force Average4 above int32 at (2,2), as well as wide differences/abs.
    image.samples(0)[2] = INT32_MIN;
    for (size_t i : {7u, 8u, 9u, 10u, 11u})
      image.samples(0)[i] = INT32_MAX - (even ? 1 : 0);
    Ok(ModularEncoderFrame::AdoptPreparedRct({5, 5}, PackedModularFormat::kGray16, 0,
                                             std::move(image), &out->frame));
    out->Plan();
    Ok(Operation(gpu).PrepareAdaptiveModularFeatures(out->frame, out->slices, {}, false,
                                                     &out->features));
  };
  Prepared f;
  prepare(false, &f);
  ModularCodingPolicy policy;
  policy.tree.nodes[0].predictor = Predictor::kZero;
  Tokens(gpu, f, policy);
  ManagedVector<metal_internal::AdaptiveModularDiagnostic> trace;
  Ok(metal_internal::DiagnoseMetalAdaptiveModularTokens(gpu, *f.features, policy.tree, &trace));
  const auto view = f.frame.image().view(0);
  bool wide_prediction = false, wide_property = false;
  for (size_t y = 0; y < 5; ++y) {
    int64_t previous = 0;
    for (size_t x = 0; x < 5; ++x) {
      const auto n = Neighbors(view, x, y);
      const auto properties = Properties(n, 0, 0, x, y, previous, 0);
      previous = properties[9];
      const auto &actual = trace[y * 5 + x];
      for (size_t p = 0; p < kPropertyCount; ++p) {
        Check(actual.properties[p] == properties[p], "Wide adaptive property was narrowed");
        wide_property |= properties[p] < INT32_MIN || properties[p] > INT32_MAX;
      }
      for (unsigned p = 0; p < 14; ++p) {
        const auto prediction = Predict(static_cast<Predictor>(p), n, 0);
        Check(actual.predictions[p] == prediction, "Wide adaptive predictor was narrowed");
        wide_prediction |= prediction < INT32_MIN || prediction > INT32_MAX;
      }
      ++checked_diagnostics;
    }
  }
  Check(wide_prediction && wide_property, "Wide arithmetic fixture did not exceed int32");
  std::unique_ptr<PreparedGpuModularTokens> old;
  Ok(Operation(gpu).GenerateAdaptiveModularTokens(*f.features, policy.tree, &old));
  const auto *identity = old.get();
  for (unsigned p = 1; p < 14; ++p) {
    if (p == 6)
      continue;
    policy.tree.nodes[0].predictor = static_cast<Predictor>(p);
    PreparedModularTokens cpu;
    const auto expected = TokenizeModular(f.frame, f.layout, policy, &cpu);
    const auto actual =
        Operation(gpu).GenerateAdaptiveModularTokens(*f.features, policy.tree, &old);
    Check(actual.code() == expected.code(), "Wide residual representability status differs");
    if (actual.ok()) {
      Equal(cpu, *old);
      identity = old.get();
    } else
      Check(old.get() == identity, "Wide residual failure replaced prior owner");
  }
  policy.tree.nodes[0].predictor = Predictor::kZero;
  policy.tree.nodes[0].offset = -1;
  Check(Operation(gpu).GenerateAdaptiveModularTokens(*f.features, policy.tree, &old).code() ==
                StatusCode::kInvalidArgument &&
            old.get() == identity,
        "Signed leaf-offset overflow was published");
  const auto *features = f.features.get();
  const auto live = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  Check(Operation(gpu)
                    .PrepareAdaptiveModularFeatures(f.frame, f.slices, {}, true, &f.features)
                    .code() == StatusCode::kUnsupported &&
            f.features.get() == features &&
            DefaultResourceBudget().snapshot().total.live_capacity_bytes == live,
        "Weighted recurrence overflow changed completed features");
  Prepared even;
  prepare(true, &even);
  policy.tree.nodes[0].offset = 2;
  policy.tree.nodes[0].multiplier = 2;
  Tokens(gpu, even, policy);
}
void Participation(GpuBackend &gpu) {
  using namespace thread_budget_internal;
  Fixture source(PackedModularFormat::kRgb8, {3, 2});
  ModularCodingPolicy policy;
  Prepared f(gpu, source, policy);
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({.cpu_participant_limit = 1}, &domain));
  struct Observation {
    size_t entered = 0, left = 0;
  } observation;
  {
    CpuExecutionScope cpu;
    Ok(cpu.Start(domain, 1));
    current_cpu_execution.observer = {
        &observation, +[](void *p) noexcept { ++static_cast<Observation *>(p)->entered; },
        +[](void *p) noexcept { ++static_cast<Observation *>(p)->left; }};
    std::unique_ptr<PreparedGpuModularTokens> tokens;
    for (bool failure : {false, true}) {
      if (failure)
        Ok(ArmNextMetalSubmissionFailureForTest(gpu, false, true));
      const auto status =
          Operation(gpu).GenerateAdaptiveModularTokens(*f.features, policy.tree, &tokens);
      Check(status.code() == (failure ? StatusCode::kDeviceError : StatusCode::kOk) &&
                observation.entered == (failure ? 2u : 1u) &&
                observation.left == observation.entered && HasCpuParticipation(),
            "Adaptive wait did not suspend and resume CPU participation");
    }
  }
  Check(domain->snapshot().active_cpu_participants == 0, "Adaptive wait leaked CPU participation");
}
void LifetimeAndAdmission(GpuBackend &gpu) {
  Fixture source(PackedModularFormat::kRgba16, {257, 3});
  ModularCodingPolicy policy;
  Prepared f(gpu, source, policy);
  // Feature ownership is independent of the borrowed CPU frame and descriptors.
  PreparedModularTokens detached_cpu;
  std::unique_ptr<PreparedGpuAdaptiveModularFeatures> detached;
  {
    Prepared temporary(gpu, source, policy);
    Ok(TokenizeModular(temporary.frame, temporary.layout, policy, &detached_cpu));
    detached = std::move(temporary.features);
  }
  std::unique_ptr<PreparedGpuModularTokens> detached_tokens;
  Ok(Operation(gpu).GenerateAdaptiveModularTokens(*detached, policy.tree, &detached_tokens));
  Equal(detached_cpu, *detached_tokens);
  AdaptiveModularTokenStoragePlan plan;
  Ok(ComputeAdaptiveModularTokenStoragePlan(f.shape.channels(), f.shape.metadata, f.slices.size(),
                                            policy.tree.size, true, 1, &plan));
  const size_t cap = std::max(plan.bridge_peak_bytes, plan.generation_peak_bytes);
  ResourceBudget budget(cap);
  ResourceReservation reservation;
  Ok(budget.TryReserve(cap, &reservation));
  {
    ResourceContextScope resources({&reservation, ResourceClass::kPreparation});
    std::unique_ptr<PreparedGpuAdaptiveModularFeatures> features;
    Ok(Operation(gpu).PrepareAdaptiveModularFeatures(f.frame, f.slices, policy.weighted, true,
                                                     &features));
    Check(budget.snapshot().total.live_capacity_bytes == plan.features.retained_bytes,
          "Adaptive feature owner retained staging or unplanned storage");
    std::unique_ptr<PreparedGpuModularTokens> tokens;
    Ok(Operation(gpu).GenerateAdaptiveModularTokens(*features, policy.tree, &tokens));
    Check(budget.snapshot().total.live_capacity_bytes == plan.retained_bytes &&
              budget.snapshot().total.pending_count == 0 &&
              budget.snapshot().peak_backing_bytes <= cap,
          "Adaptive generation exceeded plan or retained dispatch scratch");
    features.reset();
    Check(budget.snapshot().total.live_capacity_bytes == plan.output_bytes,
          "Completed token owner unnecessarily retained feature backing");
    PreparedModularTokens cpu;
    // CPU reference outside this deliberately operation-only reservation.
    {
      ResourceContextScope reference({nullptr, ResourceClass::kPreparation});
      Ok(TokenizeModular(f.frame, f.layout, policy, &cpu));
    }
    Equal(cpu, *tokens);
  }
  Check(budget.snapshot().total.live_capacity_bytes == 0, "Adaptive owners leaked storage");
  size_t maximum_width = 0;
  for (const auto &slice : f.slices)
    maximum_width = std::max(maximum_width, size_t{slice.width});
  const size_t bridge_exact =
      plan.features.retained_bytes + (maximum_width + 2) * 2 * 5 * sizeof(uint32_t);
  for (size_t available : {bridge_exact - 1, bridge_exact}) {
    ResourceBudget b(available);
    ResourceReservation r;
    Ok(b.TryReserve(available, &r));
    ResourceContextScope scope({&r, ResourceClass::kPreparation});
    std::unique_ptr<PreparedGpuAdaptiveModularFeatures> features;
    const auto status = Operation(gpu).PrepareAdaptiveModularFeatures(
        f.frame, f.slices, policy.weighted, true, &features);
    Check(available == bridge_exact ? status.ok() : status.resource_plan_exceeded(),
          "Adaptive feature bridge exact allocation boundary differs");
    features.reset();
    Check(b.snapshot().total.live_capacity_bytes == 0 && b.snapshot().total.pending_count == 0,
          "Adaptive feature bridge admission failure leaked storage");
  }
  // Direct initialization has an exact smaller allocation peak for generation.
  const size_t exact = plan.output_bytes + plan.dispatch_bytes;
  for (size_t available : {exact - 1, exact}) {
    ResourceBudget b(available);
    ResourceReservation r;
    Ok(b.TryReserve(available, &r));
    ResourceContextScope scope({&r, ResourceClass::kPreparation});
    std::unique_ptr<PreparedGpuModularTokens> tokens;
    const auto status =
        Operation(gpu).GenerateAdaptiveModularTokens(*f.features, policy.tree, &tokens);
    Check(available == exact ? status.ok() : status.resource_plan_exceeded(),
          "Adaptive exact allocation boundary differs");
    tokens.reset();
    Check(b.snapshot().total.live_capacity_bytes == 0 && b.snapshot().total.pending_count == 0,
          "Adaptive admission failure leaked storage");
  }
}
void Failures(GpuBackend &gpu, GpuBackend &foreign) {
  Fixture source(PackedModularFormat::kRgba16, {257, 3});
  auto policy = Split(15);
  Prepared f(gpu, source, policy);
  auto &op = Operation(gpu);
  const auto *feature_identity = f.features.get();
  const size_t live = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  const auto before = gpu.stats();
  for (unsigned error = 0; error < 9; ++error) {
    auto slices = f.slices;
    auto parameters = policy.weighted;
    switch (error) {
    case 0:
      ++slices[0].plane_offset_elements;
      break;
    case 1:
      ++slices[0].token_offset;
      break;
    case 2:
      ++slices[0].property_channel;
      break;
    case 3:
      ++slices[0].stream_id;
      break;
    case 4:
      slices[0].reserved[0] = 1;
      break;
    case 5:
      slices.pop_back();
      break;
    case 6:
      std::swap(slices[0], slices[1]);
      break;
    case 7:
      parameters.coefficients[0] = 32;
      break;
    default:
      break;
    }
    const auto status = op.PrepareAdaptiveModularFeatures(f.frame, slices, parameters, true,
                                                          error == 8 ? nullptr : &f.features);
    Check(!status.ok() && f.features.get() == feature_identity,
          "Malformed adaptive feature request changed output");
  }
  Check(gpu.stats().successful_allocations == before.successful_allocations &&
            gpu.stats().committed_submissions == before.committed_submissions,
        "Rejected adaptive feature request allocated or submitted");
  // Exhaust every managed host allocation in the CPU weighted bridge.
  size_t failed = 0;
  for (size_t at = 0; at < 100; ++at) {
    std::unique_ptr<PreparedGpuAdaptiveModularFeatures> trial;
    ArmManagedHostAllocationFailureAfterForTest(at);
    const auto status =
        op.PrepareAdaptiveModularFeatures(f.frame, f.slices, policy.weighted, true, &trial);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (pending) {
      Ok(status);
      break;
    }
    ++failed;
    Check(status.code() == StatusCode::kOutOfMemory && !trial &&
              DefaultResourceBudget().snapshot().total.live_capacity_bytes == live,
          "Adaptive CPU bridge allocation failure leaked/published backing");
  }
  Check(failed > 5 && failed < 99, "Adaptive bridge allocation sweep did not complete");
  Ok(ArmNextMetalAllocationFailureForTest(gpu));
  Check(op.PrepareAdaptiveModularFeatures(f.frame, f.slices, policy.weighted, true, &f.features)
                    .code() == StatusCode::kOutOfMemory &&
            f.features.get() == feature_identity &&
            DefaultResourceBudget().snapshot().total.live_capacity_bytes == live,
        "Adaptive feature device allocation failure was not atomic");
  std::unique_ptr<PreparedGpuModularTokens> old;
  Ok(op.GenerateAdaptiveModularTokens(*f.features, policy.tree, &old));
  const auto *identity = old.get();
  const auto retained = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  for (unsigned fault = 0; fault < 3; ++fault) {
    if (fault == 0)
      Ok(ArmNextMetalAllocationFailureForTest(gpu));
    else
      Ok(ArmNextMetalSubmissionFailureForTest(gpu, fault == 1, fault == 2));
    const auto status = op.GenerateAdaptiveModularTokens(*f.features, policy.tree, &old);
    Check(status.code() == (fault == 0   ? StatusCode::kOutOfMemory
                            : fault == 1 ? StatusCode::kSubmissionFailed
                                         : StatusCode::kDeviceError) &&
              old.get() == identity &&
              DefaultResourceBudget().snapshot().total.live_capacity_bytes == retained &&
              DefaultResourceBudget().snapshot().total.pending_count == 0,
          "Adaptive token failure changed old output or leaked backing");
  }
  const auto stats = gpu.stats();
  struct ForeignImplementation final : PreparedGpuAdaptiveModularFeatures {
    explicit ForeignImplementation(const PreparedGpuAdaptiveModularFeatures &native)
        : native(native) {}
    const PreparedGpuAdaptiveModularFeatures &native;
    BackendId backend_id() const noexcept override { return native.backend_id(); }
    const codec_internal::ImageMetadata &metadata() const noexcept override {
      return native.metadata();
    }
    size_t token_count() const noexcept override { return native.token_count(); }
    std::span<const AdaptiveModularTokenSlice> slices() const noexcept override {
      return native.slices();
    }
    bool has_weighted_features() const noexcept override { return native.has_weighted_features(); }
    Status
    BorrowWeightedHostView(std::span<const AdaptiveModularWeightedFeature> *out) const override {
      return native.BorrowWeightedHostView(out);
    }
  } fake(*f.features);
  Check(!op.GenerateAdaptiveModularTokens(fake, policy.tree, &old).ok() && old.get() == identity,
        "Nonnative adaptive feature implementation was accepted");
  auto invalid = policy.tree;
  invalid.nodes[0].left = 0;
  Check(!op.GenerateAdaptiveModularTokens(*f.features, invalid, &old).ok() && old.get() == identity,
        "Adaptive invalid tree was accepted");
  Check(!Operation(foreign).GenerateAdaptiveModularTokens(*f.features, policy.tree, &old).ok() &&
            old.get() == identity,
        "Foreign adaptive features were accepted");
  Check(!op.GenerateAdaptiveModularTokens(*f.features, policy.tree, nullptr).ok(),
        "Null adaptive output accepted");
  Check(gpu.stats().successful_allocations == stats.successful_allocations &&
            gpu.stats().committed_submissions == stats.committed_submissions,
        "Rejected adaptive token request allocated/submitted");
  Prepared plain(gpu, source, policy, false);
  std::span<const AdaptiveModularWeightedFeature> empty;
  Ok(plain.features->BorrowWeightedHostView(&empty));
  Check(empty.empty(), "Unweighted owner exposed weighted features");
  Check(op.GenerateAdaptiveModularTokens(*plain.features, policy.tree, &old).code() ==
                StatusCode::kUnsupported &&
            old.get() == identity,
        "Missing weighted map was not rejected");
  ModularCodingPolicy unweighted;
  Tokens(gpu, plain, unweighted);
  auto too_large = Balanced(1024);
  Check(op.GenerateAdaptiveModularTokens(*plain.features, too_large, &old).code() ==
                StatusCode::kUnsupported &&
            old.get() == identity,
        "Image-dependent tree limit was not enforced");
  auto indivisible = unweighted.tree;
  indivisible.nodes[0].predictor = Predictor::kZero;
  indivisible.nodes[0].multiplier = 2;
  Check(op.GenerateAdaptiveModularTokens(*plain.features, indivisible, &old).code() ==
                StatusCode::kInvalidArgument &&
            old.get() == identity,
        "Inexact leaf residual was published");
  ManagedVector<metal_internal::AdaptiveModularDiagnostic> diagnostic(1);
  const auto *diagnostic_identity = diagnostic.data();
  const auto diagnostic_live = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  ArmNextManagedHostAllocationFailureForTest();
  const auto diagnostic_status = metal_internal::DiagnoseMetalAdaptiveModularTokens(
      gpu, *f.features, policy.tree, &diagnostic);
  const bool pending_diagnostic = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Check(diagnostic_status.code() == StatusCode::kOutOfMemory && !pending_diagnostic &&
            diagnostic.data() == diagnostic_identity &&
            DefaultResourceBudget().snapshot().total.live_capacity_bytes == diagnostic_live,
        "Adaptive diagnostic allocation failure changed output or leaked storage");
  // Handoff must preserve both owners on each dynamic header/view allocation
  // failure.
  ModularCodingPolicy large;
  large.tree = Balanced(256);
  std::unique_ptr<PreparedGpuModularTokens> backing;
  Ok(op.GenerateAdaptiveModularTokens(*f.features, large.tree, &backing));
  auto *backing_identity = backing.get();
  PreparedModularTokens output;
  Ok(TokenizeModular(f.frame, f.layout, policy, &output));
  const auto *old_tokens = output.tokens.data();
  size_t handoff_failures = 0;
  for (size_t at = 0; at < 10; ++at) {
    const auto charged = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
    ArmManagedHostAllocationFailureAfterForTest(at);
    const auto status = AdoptAdaptiveGpuModularTokens(f.layout, large, std::move(backing), &output);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (pending) {
      Ok(status);
      break;
    }
    ++handoff_failures;
    Check(status.code() == StatusCode::kOutOfMemory && backing.get() == backing_identity &&
              output.tokens.data() == old_tokens &&
              DefaultResourceBudget().snapshot().total.live_capacity_bytes == charged,
          "Adaptive handoff allocation failure consumed or replaced an owner");
  }
  Check(handoff_failures >= 3 && !backing && output.gpu_backing.get() == backing_identity,
        "Adaptive handoff allocation sweep did not complete");
}
} // namespace
int main(int argc, char **argv) try {
  artifacts = argc > 1 ? argv[1] : "metal-modular-adaptive";
  std::filesystem::create_directories(artifacts);
  {
    std::unique_ptr<GpuBackend> gpu, embedded;
    Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu));
    Ok(CreateEmbeddedMetalBackend({}, &embedded));
    std::cout << "Adaptive Modular device: " << gpu->name() << std::endl;
    Matrix(*gpu);
    LargeTrees(*gpu);
    WideArithmetic(*gpu);
    Participation(*gpu);
    LifetimeAndAdmission(*gpu);
    Failures(*gpu, *embedded);
    Fixture source(PackedModularFormat::kRgb8);
    auto p = Split(15);
    Prepared f(*embedded, source, p);
    Tokens(*embedded, f, p);
  }
  Check(DefaultResourceBudget().snapshot().total.live_capacity_bytes == 0 &&
            DefaultResourceBudget().snapshot().total.pending_count == 0,
        "Adaptive qualification leaked managed backing");
  std::cout << cases << " cases, " << checked_tokens << " exact tokens, " << checked_features
            << " weighted features, " << checked_diagnostics << " diagnostics, " << checked_files
            << " identical files, " << decoded_samples << " independently decoded samples\n";
  return 0;
} catch (const std::exception &error) {
  DisarmManagedHostAllocationFailureForTest();
  std::cerr << "Adaptive Modular qualification failed: " << error.what() << '\n';
  return 1;
}
