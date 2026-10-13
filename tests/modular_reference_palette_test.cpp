// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/histogram_sample_preparation.h"
#include "codec/modular/selection_cost.h"
#include "codec/modular/transform/reference_palette.h"
#include "codestream/modular/prepared_streams.h"
#include "codestream/modular/workflow.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifdef GJXL_PALETTE_REFERENCE
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/enc_palette.h"
#include "lib/jxl/modular/transform/palette.h"
#include "lib/jxl/modular/transform/transform.h"
#include "modular_reference.h"
#endif

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
using Color = std::array<int32_t, 4>;
size_t cases = 0, accepted = 0, empty = 0, enlarged = 0, failures = 0, files = 0, peak = 0;
std::filesystem::path artifacts;
void Check(bool ok, const char *m) {
  if (!ok)
    throw std::runtime_error(m);
}
void Ok(Status s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
Color Cube(size_t k, uint8_t bits) {
  Color v{};
  const uint32_t maximum = (uint32_t{1} << bits) - 1;
  for (size_t c = 0, scale = 1; c < 3; ++c, scale *= k < 64 ? 4 : 5) {
    const auto digit = (k < 64 ? k : k - 64) / scale % (k < 64 ? 4 : 5);
    v[c] = digit * uint64_t{maximum} / 4 + (k < 64 ? (1 << std::max(0, int{bits} - 3)) : 0);
  }
  return v;
}
struct Fixture {
  std::vector<std::vector<int32_t>> backing;
  std::vector<ModularChannelView> views;
  size_t metadata = 0;
  ReferencePaletteOptions options;
};
Fixture Make(const std::vector<Color> &pixels, uint8_t count, uint8_t bits, uint16_t limit,
             bool ordered, bool embedded, size_t width = 0) {
  Fixture f;
  f.metadata = embedded ? 2 : 0;
  const uint8_t begin = embedded ? 3 : 0;
  f.options = {begin, count, bits, limit, ordered};
  const size_t channels = begin + count + (embedded ? 1 : 0);
  const size_t w = width ? width : pixels.size(), h = pixels.size() / w;
  Check(w * h == pixels.size(), "Bad generated shape");
  f.backing.resize(channels);
  f.views.resize(channels);
  for (size_t c = 0; c < channels; ++c) {
    const bool selected = c >= begin && c < size_t{begin} + count;
    const Extent2D extent = selected ? Extent2D{w, h} : c == 0 ? Extent2D{0, 3} : Extent2D{7, 2};
    const size_t stride = extent.width ? extent.width + 5 : 0;
    f.backing[c].assign(extent.width ? (extent.height - 1) * stride + extent.width + 2 : 0, -991);
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        f.backing[c][y * stride + x] = selected ? pixels[y * w + x][c - begin] : int32_t(c + x + y);
    ChannelDescriptor d{extent, uint8_t(selected && embedded ? 2 : 0),
                        uint8_t(selected && embedded ? 1 : 0),
                        c < f.metadata ? ChannelRole::kMetadata : ChannelRole::kColor};
    if (selected && count == 4 && c == size_t{begin} + 3)
      d.role = ChannelRole::kAlpha;
    f.views[c] = {f.backing[c], d, stride};
  }
  return f;
}
ModularImage Own(std::span<const ModularChannelView> views, size_t metadata) {
  std::vector<ChannelDescriptor> descriptors;
  for (auto v : views)
    descriptors.push_back(v.descriptor);
  ModularImage image;
  Ok(ModularImage::Create(descriptors, metadata, &image));
  for (size_t c = 0; c < views.size(); ++c) {
    const auto [w, h] = views[c].descriptor.extent;
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x)
        image.samples(c)[y * w + x] = views[c].Row(y)[x];
  }
  return image;
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
#ifdef GJXL_PALETTE_REFERENCE
void Reference(const Fixture &f, const ReferencePaletteCandidate &candidate) {
  JxlMemoryManager memory;
  Check(bool(jxl::MemoryManagerInit(&memory, nullptr)), "Reference allocator failed");
  jxl::Image image(&memory);
  image.w = f.views[f.options.begin].descriptor.extent.width;
  image.h = f.views[f.options.begin].descriptor.extent.height;
  image.bitdepth = f.options.bit_depth;
  image.nb_meta_channels = f.metadata;
  for (const auto v : f.views) {
    const auto [w, h] = v.descriptor.extent;
    auto result = jxl::Channel::Create(&memory, w, h);
    Check(result.ok(), "Reference channel failed");
    auto ch = std::move(result).value_();
    ch.hshift = v.descriptor.role == ChannelRole::kMetadata ? -1 : v.descriptor.hshift;
    ch.vshift = v.descriptor.role == ChannelRole::kMetadata ? -1 : v.descriptor.vshift;
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x)
        ch.Row(y)[x] = v.Row(y)[x];
    image.channel.push_back(std::move(ch));
  }
  uint32_t colors = f.options.maximum_colors, deltas = 0;
  jxl::Predictor predictor = jxl::Predictor::Weighted;
  const bool applied =
      bool(jxl::FwdPalette(image, f.options.begin, f.options.begin + f.options.count - 1, colors,
                           deltas, f.options.ordered, false, predictor, {}));
  Check(applied == candidate.applied, "Reference acceptance differs");
  if (applied) {
    Check(colors == candidate.transform.colors && deltas == 0 && predictor == jxl::Predictor::Zero,
          "Reference palette width/predictor differs");
    Check(image.channel.size() == candidate.image.channel_count() &&
              image.nb_meta_channels == candidate.image.metadata_channels(),
          "Reference shape differs");
    for (size_t c = 0; c < image.channel.size(); ++c) {
      const auto v = candidate.image.view(c);
      const auto &r = image.channel[c];
      Check(v.descriptor.extent == Extent2D{r.w, r.h}, "Reference extent differs");
      if (c >= image.nb_meta_channels)
        Check(v.descriptor.hshift == r.hshift && v.descriptor.vshift == r.vshift,
              "Reference shifts differ");
      for (size_t y = 0; y < r.h; ++y)
        for (size_t x = 0; x < r.w; ++x)
          Check(v.Row(y)[x] == r.Row(y)[x], "Reference dictionary/index/preserved sample differs");
    }
    Check(bool(jxl::InvPalette(image, f.options.begin, colors, 0, predictor, {}, nullptr)),
          "Reference inverse failed");
  }
  Check(image.channel.size() == f.views.size() && image.nb_meta_channels == f.metadata,
        "Reference inverse/rejection shape differs");
  for (size_t c = 0; c < f.views.size(); ++c) {
    const auto [w, h] = f.views[c].descriptor.extent;
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x)
        Check(image.channel[c].Row(y)[x] == f.views[c].Row(y)[x],
              "Reference reconstruction differs");
  }
}
#endif
void Case(const Fixture &f, bool fault = false, int expected_width = -1) {
  const auto unchanged = f.backing;
  HostStorageBound bound{19, 23};
  ArmNextManagedHostAllocationFailureForTest();
  auto status = ComputeReferencePaletteStorageBound(f.views, f.metadata, f.options, &bound);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Check(status.ok() && pending && bound.retained_bytes == bound.peak_bytes,
        "Palette planner failed/allocated");
  ResourceBudget budget(std::max(size_t{1}, bound.peak_bytes));
  ResourceReservation reservation;
  Ok(budget.Reserve(std::max(size_t{1}, bound.peak_bytes), &reservation));
  ReferencePaletteCandidate candidate;
  {
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    Ok(TryBuildReferencePalette(f.views, f.metadata, f.options, &candidate));
  }
  Check(budget.snapshot().peak_backing_bytes <= bound.peak_bytes, "Palette exceeded bound");
  peak = std::max(peak, budget.snapshot().peak_backing_bytes);
  if (expected_width >= 0)
    Check(candidate.applied && candidate.transform.colors == expected_width,
          "Golden palette width differs");
#ifdef GJXL_PALETTE_REFERENCE
  Reference(f, candidate);
#endif
  if (candidate.applied) {
    ++accepted;
    empty += candidate.transform.colors == 0;
    enlarged += candidate.transform.colors > f.options.maximum_colors;
    auto original = Own(f.views, f.metadata);
    ChannelShape shape;
    Ok(DescribeImage(original, &shape));
    TransformSequence sequence;
    sequence.size = 1;
    sequence.entries[0] = candidate.transform;
    ModularImage again, restored;
    Ok(ForwardTransforms(original, sequence, &again));
    Check(Same(candidate.image, again), "Prescribed forward differs from discovered palette");
    Ok(InverseTransforms(candidate.image, shape.channels(), shape.metadata, sequence, &restored));
    Check(Same(original, restored), "Native reconstruction differs");
    // Empty dictionaries must survive following transforms and copies.
    if (candidate.transform.colors == 0) {
      const size_t begin = candidate.transform.begin + 1;
      if (candidate.image.view(begin).descriptor.extent.width > 1) {
        sequence.entries[sequence.size++] = {TransformKind::kSqueeze, static_cast<uint8_t>(begin),
                                             1};
        Ok(ForwardTransforms(original, sequence, &again));
        Ok(InverseTransforms(again, shape.channels(), shape.metadata, sequence, &restored));
        Check(Same(original, restored), "Empty metadata composition differs");
      }
    }
    if (fault) {
      bool completed = false;
      for (size_t fail = 0; fail < 70; ++fail) {
        ResourceBudget fault_budget(bound.peak_bytes);
        ResourceReservation fault_reservation;
        Ok(fault_budget.Reserve(bound.peak_bytes, &fault_reservation));
        ReferencePaletteCandidate output;
        output.applied = true;
        output.transform.colors = 73;
        output.image = Own(f.views, f.metadata);
        const auto before = output.transform;
        {
          ResourceContextScope scope({&fault_reservation, ResourceClass::kPreparation});
          ArmManagedHostAllocationFailureAfterForTest(fail);
          status = TryBuildReferencePalette(f.views, f.metadata, f.options, &output);
          const bool unused = ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
          if (status.ok()) {
            Check(unused && output.applied && Same(output.image, candidate.image),
                  "Fault recovery differs");
            completed = true;
          } else {
            Check(!unused && status.code() == StatusCode::kOutOfMemory && output.applied &&
                      output.transform == before && Same(output.image, original),
                  "Fault changed output");
            Check(fault_budget.snapshot().total.backing_count == 0 &&
                      fault_budget.snapshot().total.pending_count == 0,
                  "Partial palette leaked");
            ++failures;
          }
        }
        output.image = {};
        Check(fault_budget.snapshot().total.backing_count == 0 &&
                  fault_budget.snapshot().total.pending_count == 0,
              "Palette result leaked backing");
        Check(fault_budget.snapshot().peak_backing_bytes <= bound.peak_bytes,
              "Fault exceeded bound");
        if (completed)
          break;
      }
      Check(completed, "Fault sweep did not finish");
    }
  } else {
    Check(candidate.image.channel_count() == 0, "Rejected candidate retained image");
    ArmNextManagedHostAllocationFailureForTest();
    status = TryBuildReferencePalette(f.views, f.metadata, f.options, &candidate);
    const bool unused = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(status.ok() && unused && !candidate.applied, "Rejected palette allocated");
  }
  Check(f.backing == unchanged, "Palette mutated borrowed input");
  candidate.image = {};
  Check(budget.snapshot().total.backing_count == 0 && budget.snapshot().total.pending_count == 0,
        "Palette candidate leaked backing");
  ++cases;
}
void Generated() {
  for (uint8_t bits : {1, 2, 3, 4, 7, 8, 12, 16, 24})
    for (uint8_t count = 1; count <= 4; ++count)
      for (bool ordered : {false, true})
        for (size_t scenario = 0; scenario < 12; ++scenario) {
          std::vector<Color> pixels;
          uint16_t limit = scenario % 4 == 0 ? 2 : scenario % 4 == 1 ? 16 : 1024;
          if (scenario == 5 || scenario == 8)
            limit = 1024;
          const size_t n = scenario == 0 ? 1 : scenario == 1 ? 2 : scenario == 2 ? 16 : 97;
          for (size_t k = 0; k < n; ++k) {
            Color color{};
            if (scenario < 3 || scenario == 10 || scenario == 11)
              color = Cube(k % 189, bits);
            else
              for (size_t c = 0; c < count; ++c)
                color[c] = int32_t((k * (53 + c * 19) + 13 + c) % 65521) - 31000;
            if (scenario == 5)
              color[3] = -1; // Equal luma*alpha keys.
            if (scenario == 6)
              color[0] = k % 2 ? INT32_MIN : INT32_MAX;
            size_t frequency = scenario == 10   ? 10
                               : scenario == 11 ? 11
                               : scenario == 7  ? 4
                               : scenario == 8  ? 5
                                                : 1;
            for (size_t repeat = 0; repeat < frequency; ++repeat)
              pixels.push_back(color);
          }
          if (scenario == 3)
            limit = 0;
          if (scenario == 4)
            limit = 1;
          // Deterministic shuffle stresses first-occurrence ordering and sort
          // ties independently of color value and frequency runs.
          if (scenario >= 5) {
            uint32_t random = 0x71339a17u;
            for (size_t i = pixels.size(); i > 1; --i) {
              random = random * 1664525u + 1013904223u;
              std::swap(pixels[i - 1], pixels[random % i]);
            }
          }
          try {
            Case(Make(pixels, count, bits, limit, ordered, scenario % 2,
                      pixels.size() % 5 == 0 ? 5 : 0));
          } catch (...) {
            std::cerr << "bits=" << unsigned(bits) << " count=" << unsigned(count)
                      << " ordered=" << ordered << " scenario=" << scenario << '\n';
            throw;
          }
        }
  // All 189 distinct implicit RGB colors at the exact promotion threshold,
  // plus the full requested nonimplicit budget: 1024 + 189 explicit entries.
  for (uint8_t bits : {8, 16}) {
    std::vector<Color> all;
    for (size_t k = 0; k < 1024; ++k)
      all.push_back({int32_t(k % 256), int32_t(k / 256), 1, 0});
    for (size_t k = 0; k < 189; ++k)
      for (size_t repeat = 0; repeat < 11; ++repeat)
        all.push_back(Cube(k, bits));
    Case(Make(all, 3, bits, 1024, true, true), true, 1213);
    std::vector<Color> implicit;
    for (size_t k = 0; k < 189; ++k)
      implicit.push_back(Cube(k, bits));
    Case(Make(implicit, 3, bits, 2, true, true, 7), true, 0);
    std::vector<Color> frequent;
    for (size_t k = 0; k < 189; ++k)
      for (size_t repeat = 0; repeat < 11; ++repeat)
        frequent.push_back(Cube(k, bits));
    for (uint8_t count : {2, 3, 4})
      Case(Make(frequent, count, bits, 2, true, true, 7), true, 189);
    Case(Make({Color{31, 31, 31, 31}}, 1, bits, 2, true, true), true, 1);
  }
}
void Contracts() {
  const auto good = Make({Color{7, 9, 11, 13}, Color{13, 17, 19, 23}}, 3, 8, 2, true, false);
  ReferencePaletteCandidate output;
  Ok(TryBuildReferencePalette(good.views, 0, good.options, &output));
  auto before = Own(std::array{output.image.view(0), output.image.view(1)}, 1);
  const auto transform = output.transform;
  for (size_t error = 0; error < 15; ++error) {
    auto options = good.options;
    auto views = good.views;
    size_t metadata = 0;
    if (error == 0)
      options.bit_depth = 0;
    if (error == 1)
      options.bit_depth = 25;
    if (error == 2)
      options.count = 0;
    if (error == 3)
      options.count = 5;
    if (error == 4)
      options.begin = 3;
    if (error == 5)
      options.maximum_colors = 1025;
    if (error == 6)
      metadata = 1;
    if (error == 7)
      views[2].stride = 1;
    if (error == 8)
      views[2].backing = views[2].backing.first(1);
    if (error == 9)
      views[2].descriptor.hshift = 1;
    if (error == 10)
      views[2].descriptor.extent.height = 0;
    if (error == 11)
      views[2].descriptor.extent.height = SIZE_MAX;
    if (error == 12)
      views[2].descriptor.extent.width = 1;
    if (error == 13)
      views.clear();
    if (error == 14)
      views.resize(65, good.views[0]);
    HostStorageBound bound{19, 23};
    ArmNextManagedHostAllocationFailureForTest();
    const auto plan = ComputeReferencePaletteStorageBound(views, metadata, options, &bound);
    const auto build = TryBuildReferencePalette(views, metadata, options, &output);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(plan.code() == StatusCode::kInvalidArgument &&
              build.code() == StatusCode::kInvalidArgument && pending &&
              bound == HostStorageBound{19, 23} && output.applied &&
              output.transform == transform && Same(output.image, before),
          "Invalid palette changed output");
  }
  Check(!TryBuildReferencePalette(good.views, 0, good.options, nullptr).ok() &&
            !ComputeReferencePaletteStorageBound(good.views, 0, good.options, nullptr).ok(),
        "Null output accepted");
  ResourceBudget tiny(1);
  ResourceReservation reservation;
  Ok(tiny.Reserve(1, &reservation));
  {
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    const auto status = TryBuildReferencePalette(good.views, 0, good.options, &output);
    Check(status.code() == StatusCode::kOutOfMemory && Same(output.image, before) &&
              tiny.snapshot().total.backing_count == 0,
          "Underbudget palette changed output");
  }
}
void EmptyDictionaryContracts() {
  ModularChannelView empty_view{{}, {{0, 3}, 0, 0, ChannelRole::kMetadata}, 0};
  Ok(empty_view.Validate());
  Check(empty_view.Row(0) == nullptr && empty_view.Row(2) == nullptr,
        "Empty row addressing differs");
  ModularChannelView slice;
  Ok(BorrowChannelSlice(empty_view, {0, 0, {0, 3}}, &slice));
  for (const ModularRect rect : {ModularRect{1, 0, {0, 3}}, {0, 1, {0, 3}}, {0, 0, {0, 2}}})
    Check(!BorrowChannelSlice(empty_view, rect, &slice).ok(),
          "Invalid empty metadata slice accepted");
  int32_t sample = 3;
  for (size_t error = 0; error < 8; ++error) {
    auto bad = empty_view;
    if (error == 0)
      bad.descriptor.role = ChannelRole::kColor;
    if (error == 1)
      bad.descriptor.extent.height = 0;
    if (error == 2)
      bad.descriptor.extent.height = 5;
    if (error == 3)
      bad.descriptor.hshift = 1;
    if (error == 4)
      bad.descriptor.vshift = 1;
    if (error == 5)
      bad.stride = 1;
    if (error == 6)
      bad.backing = {&sample, 1};
    if (error == 7)
      bad.descriptor.extent = {1, 0};
    Check(!bad.Validate().ok(), "Malformed empty metadata accepted");
  }
  std::vector<Color> pixels;
  for (size_t k = 0; k < 189; ++k)
    pixels.push_back(Cube(k, 8));
  auto fixture = Make(pixels, 3, 8, 2, true, false, 7);
  ReferencePaletteCandidate candidate;
  Ok(TryBuildReferencePalette(fixture.views, 0, fixture.options, &candidate));
  Check(candidate.applied && candidate.transform.colors == 0, "Empty dictionary fixture differs");
  const std::array views{candidate.image.view(0), candidate.image.view(1)};
  const HistogramTrainingImage training{{7, 27}, views, 1};
  // Static sample columns contain quantized bins, not raw channel IDs. A unit
  // multiplier box forces a cut at channel 0 without changing residuals, so
  // bin 1 explicitly distinguishes the retained index-plane ID from 0.
  HistogramMultiplierRange channel_box;
  channel_box.range = {{{1, 2}, {0, 1}}};
  HistogramSamplingOptions sampling;
  sampling.multipliers = {&channel_box, 1};
  PreparedHistogramSamples prepared;
  Ok(PrepareHistogramSamples({&training, 1}, &prepared, sampling));
  Check(prepared.stats().scanned_pixels == 189 && prepared.stats().channels == 1 &&
            prepared.samples().total_samples != 0 &&
            std::ranges::all_of(prepared.samples().static_values[0],
                                [](uint32_t c) { return c == 1; }),
        "Empty palette shifted training channel IDs/population");
  ModularTransformCost full_cost, index_cost;
  Ok(EstimateModularTransformCost(views, &full_cost));
  Ok(EstimateModularTransformCost(std::span(views).subspan(1), &index_cost));
  Check(full_cost == index_cost, "Empty dictionary changed gradient estimate");
  ModularWeightedCost full_wp, index_wp;
  Ok(EstimateModularWeightedCost(views, 0, &full_wp));
  Ok(EstimateModularWeightedCost(std::span(views).subspan(1), 0, &index_wp));
  Check(full_wp == index_wp, "Empty dictionary changed WP estimate");
  TransformSequence sequence;
  sequence.size = 1;
  sequence.entries[0] = candidate.transform;
  auto original = Own(fixture.views, 0);
  ChannelShape shape;
  Ok(DescribeImage(original, &shape));
  for (int32_t bad_index : {-1, 189, INT32_MAX}) {
    ModularImage invalid, output;
    Ok(ForwardTransforms(candidate.image, {}, &invalid));
    Ok(ForwardTransforms(original, {}, &output));
    invalid.samples(1)[0] = bad_index;
    Check(!InverseTransforms(invalid, shape.channels(), 0, sequence, &output).ok() &&
              Same(output, original),
          "Invalid implicit index changed inverse output");
  }
  // A successful publication may replace the very image that supplied its
  // borrowed channels. All reads must finish before destroying that image.
  auto aliased = Own(fixture.views, 0);
  ReferencePaletteCandidate replacement;
  replacement.image = std::move(aliased);
  std::array<ModularChannelView, 3> borrowed{replacement.image.view(0), replacement.image.view(1),
                                             replacement.image.view(2)};
  Ok(TryBuildReferencePalette(borrowed, 0, fixture.options, &replacement));
  Check(replacement.applied && Same(replacement.image, candidate.image),
        "Aliased palette publication differs");
}
#ifdef GJXL_PALETTE_REFERENCE
void SaveFile(std::span<const uint8_t> bytes) {
  if (!artifacts.empty()) {
    std::ofstream out(artifacts / ("case-" + std::to_string(files) + ".jxl"), std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    Check(bool(out), "Palette artifact write failed");
  }
  ++files;
}
void FileCase(uint8_t bits, uint8_t count, size_t scenario, bool local_palette) {
  namespace ref = gjxl::test::modular_reference;
  std::vector<Color> pixels;
  if (scenario == 0 || scenario == 1) {
    for (size_t repeat = 0; repeat < (scenario == 0 ? 1 : 2); ++repeat)
      for (size_t k = 0; k < 189; ++k)
        pixels.push_back(Cube(k, bits));
  } else if (scenario == 2) {
    for (size_t k = 0; k < 1024; ++k)
      pixels.push_back({int32_t(k % 256), int32_t(k / 256), 1, 0});
    for (size_t k = 0; k < 189; ++k)
      for (size_t repeat = 0; repeat < 11; ++repeat)
        pixels.push_back(Cube(k, bits));
  } else {
    for (size_t k = 0; k < 378; ++k)
      pixels.push_back({31, 31, 31, 31});
  }
  const auto format =
      count == 1   ? (bits == 8 ? PackedModularFormat::kGray8 : PackedModularFormat::kGray16)
      : count == 3 ? (bits == 8 ? PackedModularFormat::kRgb8 : PackedModularFormat::kRgb16)
                   : (bits == 8 ? PackedModularFormat::kRgba8 : PackedModularFormat::kRgba16);
  const Extent2D extent{pixels.size(), 1};
  const size_t row = pixels.size() * count * (bits / 8);
  std::vector<uint8_t> bytes(row);
  ref::IntegerImage expected{extent.width, extent.height, count, bits, {}};
  for (size_t i = 0; i < pixels.size(); ++i)
    for (size_t c = 0; c < count; ++c) {
      const uint16_t sample = pixels[i][c];
      expected.samples.push_back(sample);
      bytes[(i * count + c) * (bits / 8)] = sample;
      if (bits == 16)
        bytes[(i * count + c) * 2 + 1] = sample >> 8;
    }
  const PackedModularImageView packed{bytes, extent, row, format};
  ModularEncoderFrame frame;
  Ok(ModularEncoderFrame::Prepare(packed, &frame));
  std::vector<ModularChannelView> source;
  for (size_t c = 0; c < count; ++c)
    source.push_back(frame.image().view(c));
  ReferencePaletteCandidate global;
  ModularCodingPolicy policy;
  if (!local_palette) {
    Ok(TryBuildReferencePalette(source, 0, {0, count, bits, 1024, true}, &global));
    Check(global.applied, "Global file palette rejected");
    policy.transforms.size = 1;
    policy.transforms.entries[0] = global.transform;
    for (const auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
      ModularWorkflowStoragePlan plan;
      Ok(ComputeModularWorkflowStoragePlan(extent, format, mode, policy, &plan));
      std::shared_ptr<const ExecutionDomain> domain;
      Ok(ExecutionDomain::Create(
          {.managed_memory_bytes = plan.working.peak_bytes, .cpu_participant_limit = 1}, &domain));
      std::vector<uint8_t> encoded;
      Ok(EncodeModularImage(packed, {domain, mode, policy}, &encoded));
      Check(ref::DecodeLossless(encoded) == expected, "Whole-frame palette decode differs");
      SaveFile(encoded);
    }
    auto wrong = policy;
    wrong.transforms.entries[0].palette_bit_depth = bits == 8 ? 16 : 8;
    ModularWorkflowStoragePlan plan;
    Check(
        !ComputeModularWorkflowStoragePlan(extent, format, EntropyCodingMode::kPrefix, wrong, &plan)
             .ok(),
        "Whole-frame planner accepted wrong palette depth");
    ModularEncoderFrame invalid;
    Check(!ModularEncoderFrame::Prepare(packed, wrong, &invalid).ok(),
          "Frame accepted wrong palette depth");
  }
  const auto &image = local_palette ? frame.image() : global.image;
  ChannelShape shape;
  Ok(DescribeImage(image, &shape));
  ModularFrameGeometry geometry;
  Ok(ModularFrameGeometry::Create(extent, &geometry));
  ModularStreamPlan layout;
  Ok(BuildModularStreamPlan(geometry, shape.channels(), shape.metadata, &layout));
  std::vector<ReferencePaletteCandidate> local(layout.streams.size());
  std::vector<std::vector<ModularChannelView>> views(layout.streams.size());
  std::vector<PreparedModularStreamView> streams(layout.streams.size());
  DecisionTree tree;
  // Channel-ID split exercises retention of zero-width dictionary IDs.
  tree.size = 3;
  tree.nodes[0] = {.property = 0, .split = 0, .left = 1, .right = 2};
  tree.nodes[1].predictor = Predictor::kWeighted;
  tree.nodes[2].predictor = Predictor::kGradient;
  for (size_t i = 0; i < streams.size(); ++i) {
    for (const auto &slice :
         std::span(layout.slices)
             .subspan(layout.streams[i].slice_begin, layout.streams[i].slice_count)) {
      ModularChannelView view;
      Ok(BorrowChannelSlice(image.view(slice.channel), slice.rect, &view));
      views[i].push_back(view);
    }
    if (i == 0)
      streams[i].transforms = policy.transforms;
    if (i && local_palette && !views[i].empty()) {
      Ok(TryBuildReferencePalette(views[i], 0, {0, count, bits, 1024, true}, &local[i]));
      Check(local[i].applied, "Local file palette rejected");
      streams[i].transforms.size = 1;
      streams[i].transforms.entries[0] = local[i].transform;
      views[i].clear();
      for (size_t c = 0; c < local[i].image.channel_count(); ++c)
        views[i].push_back(local[i].image.view(c));
    }
    streams[i].channels = views[i];
  }
  for (bool local_trees : {false, true}) {
    for (auto &stream : streams) {
      size_t pixels_in_stream = 0;
      for (auto ch : stream.channels)
        pixels_in_stream += ch.descriptor.extent.width * ch.descriptor.extent.height;
      stream.local_tree = local_trees && pixels_in_stream ? &tree : nullptr;
    }
    const PreparedModularStreamsView input{extent, format, local_trees ? nullptr : &tree, streams};
    for (auto entropy :
         {PreparedModularEntropyPolicy::kPrefix, PreparedModularEntropyPolicy::kAnsFromPrefix,
          PreparedModularEntropyPolicy::kDirectRateAns}) {
      const PreparedModularEntropyOptions options{entropy, AnsSelectionMode::kEstimated};
      ModularWorkflowStoragePlan plan;
      Ok(ComputePreparedModularStreamsStoragePlan(input, options, &plan));
      ResourceBudget budget(plan.working.peak_bytes);
      ResourceReservation reservation;
      Ok(budget.Reserve(plan.working.peak_bytes, &reservation));
      codestream_internal::CodestreamBuffer encoded;
      {
        ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
        Ok(EncodePreparedModularStreams(input, options, &encoded));
      }
      Check(budget.snapshot().peak_backing_bytes <= plan.working.peak_bytes &&
                encoded.size() <= plan.maximum_codestream_bytes,
            "Palette file exceeded bound");
      Check(ref::DecodeLossless(encoded.view()) == expected, "Prepared palette decode differs");
      SaveFile(encoded.view());
    }
  }
  for (size_t i = 0; i < streams.size(); ++i)
    if (streams[i].transforms.size) {
      auto &depth = streams[i].transforms.entries[0].palette_bit_depth;
      depth = bits == 8 ? 16 : 8;
      const PreparedModularStreamsView bad{extent, format, &tree, streams};
      ModularWorkflowStoragePlan plan;
      Check(!ComputePreparedModularStreamsStoragePlan(bad, EntropyCodingMode::kPrefix, &plan).ok(),
            "Prepared planner accepted wrong palette depth");
      depth = bits;
    }
}
void Files() {
  for (uint8_t bits : {8, 16}) {
    for (uint8_t count : {3, 4})
      for (size_t scenario = 0; scenario < 3; ++scenario)
        for (bool local : {false, true}) {
          if (local && scenario == 0)
            continue; // Small image entirely in global stream.
          try {
            FileCase(bits, count, scenario, local);
          } catch (...) {
            std::cerr << "file bits=" << unsigned(bits) << " count=" << unsigned(count)
                      << " scenario=" << scenario << " local=" << local << '\n';
            throw;
          }
        }
    for (bool local : {false, true})
      FileCase(bits, 1, 3, local);
  }
}
#endif
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc > 1) {
      artifacts = argv[1];
      std::filesystem::create_directories(artifacts);
    }
    Generated();
    Contracts();
    EmptyDictionaryContracts();
#ifdef GJXL_PALETTE_REFERENCE
    Files();
#endif
    std::cout << "palette cases=" << cases << " accepted=" << accepted << " empty=" << empty
              << " enlarged=" << enlarged << " allocation_failures=" << failures
              << " max_managed_peak=" << peak << " files=" << files << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
