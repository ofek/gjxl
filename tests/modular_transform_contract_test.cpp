// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/transform/transform.h"
#include "codestream/modular/search.h"
#include "codestream/modular/stream_encoder.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
TransformDescriptor Rct(uint8_t begin, uint8_t type) {
  TransformDescriptor t{TransformKind::kRct, begin, 3};
  t.rct_type = type;
  return t;
}
bool Same(const ModularImage &a, const ModularImage &b) {
  if (a.channel_count() != b.channel_count() || a.metadata_channels() != b.metadata_channels())
    return false;
  for (size_t c = 0; c < a.channel_count(); ++c)
    if (a.view(c).descriptor != b.view(c).descriptor ||
        !std::ranges::equal(a.view(c).backing, b.view(c).backing))
      return false;
  return true;
}
ModularImage Sentinel() {
  const ChannelDescriptor shape{{1, 1}};
  ModularImage image;
  Ok(ModularImage::Create({&shape, 1}, 0, &image));
  image.samples(0)[0] = 73;
  return image;
}
void Contracts() {
  const std::array<ChannelDescriptor, 4> shape = {
      ChannelDescriptor{{17, 19}}, {{17, 19}}, {{17, 19}}, {{17, 19}, 0, 0, ChannelRole::kAlpha}};
  ModularImage input;
  Ok(ModularImage::Create(shape, 0, &input));
  for (size_t c = 0; c < 4; ++c)
    for (size_t i = 0; i < 17 * 19; ++i)
      input.samples(c)[i] = static_cast<int32_t>(i * (2 * c + 1)) - 701;
  TransformSequence seq;
  seq.size = 4;
  seq.entries[0] = {TransformKind::kPalette, 0, 1, 1024};
  seq.entries[1] = {TransformKind::kPalette, 2, 1, 1024};
  seq.entries[2] = Rct(2, 33);
  seq.entries[3] = {TransformKind::kSqueeze, 2, 3, 1, true, false};
  std::array<ChannelShape, kMaximumTransforms + 1> shapes{};
  ArmNextManagedHostAllocationFailureForTest();
  const auto status = PlanTransforms(shape, 0, seq, &shapes);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Check(status.ok() && pending, "Valid planning allocated");
  HostStorageBound bound;
  for (size_t i = 0; i <= seq.size; ++i) {
    HostStorageBound stage;
    Ok(ComputeModularImageStorageBound(shapes[i].channels(), shapes[i].metadata, &stage));
    Check(bound.Add(stage), "Transform bound overflow");
  }
  ModularImage expected, restored;
  Ok(ForwardTransforms(input, seq, &expected));
  Ok(InverseTransforms(expected, shape, 0, seq, &restored));
  Check(Same(input, restored), "Ordered native roundtrip differs");
  ModularImage alias;
  Ok(ForwardTransforms(input, {}, &alias));
  ArmNextManagedHostAllocationFailureForTest();
  const auto alias_status = ForwardTransforms(alias, seq, &alias);
  DisarmManagedHostAllocationFailureForTest();
  Check(alias_status.code() == StatusCode::kOutOfMemory && Same(alias, input),
        "Failed aliased transform changed source/output");
  Ok(ForwardTransforms(alias, seq, &alias));
  Check(Same(alias, expected), "Aliased forward differs");
  Ok(InverseTransforms(alias, shape, 0, seq, &alias));
  Check(Same(alias, input), "Aliased inverse differs");
  // Every allocation in both directions is injected under the exact sum of
  // stage bounds. Failure must retain the old output and release partial stages.
  for (bool inverse : {false, true}) {
    ResourceBudget budget(bound.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.peak_bytes, &reservation));
    bool finished = false;
    for (size_t fail = 0; fail < 100; ++fail) {
      auto out = Sentinel();
      const auto original = Sentinel();
      {
        ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
        ArmManagedHostAllocationFailureAfterForTest(fail);
        const auto s = inverse ? InverseTransforms(expected, shape, 0, seq, &out)
                               : ForwardTransforms(input, seq, &out);
        const bool pending = ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        if (s.ok()) {
          Check(pending && Same(out, inverse ? input : expected), "Successful fault sweep differs");
          finished = true;
        } else {
          Check(!pending && s.code() == StatusCode::kOutOfMemory && Same(out, original),
                "Transform allocation failure changed output");
          Check(budget.snapshot().total.backing_count == 0 &&
                    budget.snapshot().total.pending_count == 0, "Partial transform leaked backing");
        }
      }
      if (finished)
        break;
    }
    Check(finished && Same(input, restored), "Fault sweep incomplete or input changed");
    Check(budget.snapshot().peak_backing_bytes <= bound.peak_bytes, "Transform exceeded its bound");
    Check(budget.snapshot().total.backing_count == 0, "Successful transform leaked backing");
  }
  for (unsigned error = 0; error < 7; ++error) {
    auto invalid = seq;
    auto descriptors = shape;
    if (error == 0) invalid.entries[2].rct_type = 42;
    if (error == 1) invalid.entries[2].count = 2;
    if (error == 2) invalid.entries[2].begin = 1; // metadata after two palettes
    if (error == 3) invalid.entries[2].begin = 4; // only two remaining channels
    if (error == 4) invalid.entries[0].colors = 1025;
    if (error == 5) descriptors[2].hshift = 1;
    if (error == 6) descriptors[2].extent.width = 16;
    auto unchanged = shapes;
    ArmNextManagedHostAllocationFailureForTest();
    const auto s = PlanTransforms(descriptors, 0, invalid, &shapes);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(!s.ok() && pending, "Invalid transform planning allocated or succeeded");
    for (size_t i = 0; i < shapes.size(); ++i)
      Check(shapes[i].count == unchanged[i].count && shapes[i].metadata == unchanged[i].metadata &&
                shapes[i].descriptors == unchanged[i].descriptors, "Failed plan changed output");
  }
  // Wide RCT arithmetic rejects overflow instead of narrowing modulo 2^32.
  const std::array<ChannelDescriptor, 3> tiny = {ChannelDescriptor{{1, 1}}, {{1, 1}}, {{1, 1}}};
  ModularImage extremes;
  Ok(ModularImage::Create(tiny, 0, &extremes));
  seq.size = 1;
  seq.entries[0] = Rct(0, 1);
  extremes.samples(0)[0] = INT32_MIN;
  extremes.samples(2)[0] = INT32_MAX;
  auto out = Sentinel();
  const auto original = Sentinel();
  Check(!ForwardTransforms(extremes, seq, &out).ok() && Same(out, original),
        "Forward RCT overflow was not atomic");
  extremes.samples(0)[0] = INT32_MAX;
  extremes.samples(2)[0] = 1;
  Check(!InverseTransforms(extremes, tiny, 0, seq, &out).ok() && Same(out, original),
        "Inverse RCT overflow was not atomic");
}
void HeadersAndSearch() {
  ModularCodingPolicy p;
  p.rct = 41;
  p.transforms.size = kMaximumTransforms;
  for (auto &t : p.transforms.entries)
    t = {TransformKind::kPalette, 60, 2, 1024};
  p.weighted.coefficients.fill(31);
  p.weighted.weights.fill(15);
  BitWriter writer;
  Ok(writer.WriteBits(3, 5));
  auto sentinel = [&] { Check(writer.bits_written() == 3 && writer.padded_bytes()[0] == 5,
                              "Header failure changed output"); };
  ArmNextManagedHostAllocationFailureForTest();
  const auto s = WriteCodingStreamHeader(p.weighted, p.rct, p.transforms, &writer);
  DisarmManagedHostAllocationFailureForTest();
  Check(s.code() == StatusCode::kOutOfMemory, "Header allocation fault not exercised");
  sentinel();
  Check(!writer.WithMaxBits(3, [&] {
    return WriteCodingStreamHeader(p.weighted, p.rct, p.transforms, &writer);
  }).ok(), "Header escaped parent allotment");
  sentinel();
  for (auto invalid : {Rct(0, 42), TransformDescriptor{TransformKind::kRct, 0, 2},
                      TransformDescriptor{TransformKind::kPalette, 0, 1, 1025}}) {
    p.transforms.entries[0] = invalid;
    ArmNextManagedHostAllocationFailureForTest();
    const auto s = WriteCodingStreamHeader(p.weighted, p.rct, p.transforms, &writer);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(!s.ok() && pending, "Invalid header allocated or succeeded");
    sentinel();
  }
  std::array<uint8_t, 514> bytes{};
  for (size_t i = 0; i < 257; ++i) {
    bytes[2 * i] = static_cast<uint8_t>(i);
    bytes[2 * i + 1] = static_cast<uint8_t>(i >> 8);
  }
  uint16_t colors = 73;
  Ok(ProbePalette({bytes, {256, 1}, 514, PackedModularFormat::kGray16}, &colors));
  Check(colors == 256, "Legacy palette probe lost capacity");
  Ok(ProbePalette({bytes, {257, 1}, 514, PackedModularFormat::kGray16}, &colors));
  Check(colors == 0, "Prescribed palette expansion changed automatic search");
  std::array<ModularCodingPolicy, 4> candidates;
  size_t count = 73;
  Check(!BuildTransformCandidates({257, 1}, PackedModularFormat::kGray16, 257, &candidates, &count)
             .ok() && count == 73, "Unqualified palette search expansion accepted");
}
} // namespace
int main() try {
  Contracts();
  HeadersAndSearch();
  std::cout << "Ordered transform storage, failure and search contracts passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
