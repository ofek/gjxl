// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/reference_selection.h"
#include "codestream/modular/prepared_streams.h"
#include <algorithm>
#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifdef GJXL_SELECTION_REFERENCE
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/transform/transform.h"
#include "modular_reference.h"
#include "modular_selection_reference.h"
#endif

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
size_t cases = 0, faults = 0, files = 0, peak = 0;
size_t palette_built = 0, palette_accepted = 0, palette_ties = 0,
       palette_rejected_cost = 0;
size_t rct_candidates = 0, wp_candidates = 0, borrowed_results = 0,
       wide_palettes = 0;
size_t shared_baseline_accepts = 0;
std::array<size_t, 42> rct_winners{};
std::array<size_t, 5> wp_winners{};
std::filesystem::path artifacts;
std::ofstream ledger;
void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
struct Fixture {
  std::vector<std::vector<int32_t>> backing;
  std::vector<ModularChannelView> views;
  size_t metadata = 0;
  ReferenceSelectionOptions options;
};
Fixture Make(size_t count, uint8_t bits, size_t pattern, Extent2D extent,
             size_t meta_kind = 0) {
  Fixture f;
  f.metadata = meta_kind ? 1 : 0;
  f.options.bit_depth = bits;
  f.backing.resize(count + f.metadata);
  const uint32_t mask = (1u << bits) - 1;
  for (size_t c = 0; c < count + f.metadata; ++c) {
    const bool meta = c < f.metadata;
    const auto size =
        meta ? (meta_kind == 1 ? Extent2D{0, 3} : Extent2D{7, 3}) : extent;
    const size_t stride = size.width ? size.width + 5 : 0;
    f.backing[c].assign(
        size.width ? (size.height - 1) * stride + size.width + 2 : 0, -1777);
    const size_t channel = meta ? 0 : c - f.metadata;
    uint32_t random = 0x9178359u + channel * 1231231u;
    for (size_t y = 0; y < size.height; ++y)
      for (size_t x = 0; x < size.width; ++x) {
        const size_t i = y * size.width + x;
        random = random * 1664525u + 1013904223u;
        uint32_t value = 0;
        switch (pattern) {
        case 0:
          value = 0;
          break;
        case 1:
          value = 17 + channel * 13;
          break;
        case 2:
          value = ((x / 3 + y / 2) % 3) * (mask / 3) + channel;
          break;
        case 3:
          value = random >> 10;
          break;
        case 4:
          value = (x * 331 + y * 127) % (mask + 1) + channel * 3;
          break;
        case 5:
          value = x + y * (channel + 1);
          break;
        case 6:
          value = (i * 7 % 13) * (mask / 13) + channel;
          break;
        case 7:
          value = channel == 3 ? random >> 10 : (i % 4) * (mask / 4) + channel;
          break;
        case 8:
          value = ((i * (3 + channel * 2)) % 17) * (mask / 17);
          break;
        case 9:
          value = ((i / (channel == 0   ? 1
                         : channel == 1 ? 4
                                        : 16)) %
                   4) *
                      mask / 4 +
                  (1u << std::max(0, int{bits} - 3));
          break;
        case 10:
          value = (x * 3 + y * 7) ^ (channel ? 11 : 0);
          break;
        case 11:
          value = (i % 2) * mask;
          break;
        case 12:
          value = (i % 2048) * 31;
          break;
        case 13:
          value = ((random >> 10) % 2048) * 31;
          break;
        default:
          throw std::runtime_error("Unknown selection fixture");
        }
        f.backing[c][y * stride + x] = meta ? int32_t(i) - 9 : value & mask;
      }
    ChannelDescriptor descriptor{
        size, 0, 0, meta ? ChannelRole::kMetadata : ChannelRole::kColor};
    if (!meta && channel == 3)
      descriptor.role = ChannelRole::kAlpha;
    f.views.push_back({f.backing[c], descriptor, stride});
  }
  return f;
}
std::vector<ModularChannelView> Views(const ModularImage &image) {
  std::vector<ModularChannelView> views;
  for (size_t c = 0; c < image.channel_count(); ++c)
    views.push_back(image.view(c));
  return views;
}
ModularImage Own(std::span<const ModularChannelView> views, size_t metadata) {
  std::vector<ChannelDescriptor> descriptors;
  for (auto view : views)
    descriptors.push_back(view.descriptor);
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
void Same(std::span<const ModularChannelView> a,
          std::span<const ModularChannelView> b) {
  Check(a.size() == b.size(), "Channel counts differ");
  for (size_t c = 0; c < a.size(); ++c) {
    Check(a[c].descriptor == b[c].descriptor, "Channel descriptors differ");
    const auto [w, h] = a[c].descriptor.extent;
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x)
        Check(a[c].Row(y)[x] == b[c].Row(y)[x], "Samples differ");
  }
}
std::vector<ModularChannelView>
Selected(std::span<const ModularChannelView> input,
         const ReferenceSelectionResult &selected) {
  return selected.image.channel_count()
             ? Views(selected.image)
             : std::vector<ModularChannelView>(input.begin(), input.end());
}
uint64_t PixelHash(std::span<const ModularChannelView> views) {
  uint64_t hash = 14695981039346656037ull;
  for (auto view : views) {
    const auto [w, h] = view.descriptor.extent;
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x)
        for (size_t b = 0; b < 4; ++b) {
          hash ^= (uint32_t(view.Row(y)[x]) >> (b * 8)) & 255;
          hash *= 1099511628211ull;
        }
  }
  return hash;
}
#ifdef GJXL_SELECTION_REFERENCE
void Reference(std::span<const ModularChannelView> input, size_t metadata,
               const ReferenceSelectionOptions &options,
               const ReferenceSelectionResult &selected) {
  JxlMemoryManager memory;
  Check(bool(jxl::MemoryManagerInit(&memory, nullptr)),
        "Reference memory manager failed");
  jxl::Image image(&memory);
  image.w = input.back().descriptor.extent.width;
  image.h = input.back().descriptor.extent.height;
  image.bitdepth = options.bit_depth;
  image.nb_meta_channels = metadata;
  for (auto view : input) {
    const auto [w, h] = view.descriptor.extent;
    auto channel = jxl::Channel::Create(&memory, w, h);
    Check(channel.ok(), "Reference channel allocation failed");
    auto ch = std::move(channel).value_();
    ch.hshift = view.descriptor.role == ChannelRole::kMetadata
                    ? -1
                    : view.descriptor.hshift;
    ch.vshift = view.descriptor.role == ChannelRole::kMetadata
                    ? -1
                    : view.descriptor.vshift;
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x)
        ch.Row(y)[x] = view.Row(y)[x];
    image.channel.push_back(std::move(ch));
  }
  jxl::gjxl_selection_reference::Trace trace;
  Check(bool(jxl::gjxl_selection_reference::Select(
            image, options.palettes != ReferencePaletteStage::kNone,
            options.palettes == ReferencePaletteStage::kGlobal ? 95.f : 80.f,
            options.palette_colors, options.select_rct, options.select_wp,
            &trace)),
        "Reference selection failed");
  Check(trace.gradient.size() == selected.trace.gradient_count &&
            trace.weighted.size() == selected.trace.wp_count &&
            trace.wp_mode == selected.wp_mode,
        "Reference cost call counts/WP winner differ");
  for (size_t i = 0; i < trace.gradient.size(); ++i)
    if (trace.gradient[i] != selected.trace.gradient_costs[i]) {
      std::cerr << "gradient call " << i
                << ": native=" << selected.trace.gradient_costs[i]
                << " reference=" << trace.gradient[i] << '\n';
      throw std::runtime_error("Reference gradient cost sequence differs");
    }
  for (size_t i = 0; i < trace.weighted.size(); ++i)
    Check(trace.weighted[i] == selected.trace.wp_costs[i],
          "Reference WP costs differ");
  Check(image.transform.size() == selected.transforms.size,
        "Reference transform count differs");
  for (size_t i = 0; i < image.transform.size(); ++i) {
    const auto &r = image.transform[i];
    const auto &n = selected.transforms.entries[i];
    Check(r.begin_c == n.begin, "Reference transform start differs");
    if (n.kind == TransformKind::kPalette)
      Check(r.id == jxl::TransformId::kPalette && r.num_c == n.count &&
                r.nb_colors == n.colors && r.nb_deltas == 0 &&
                r.predictor == jxl::Predictor::Zero &&
                r.ordered_palette == n.palette_ordered,
            "Reference palette descriptor differs");
    else
      Check(r.id == jxl::TransformId::kRCT && r.rct_type == n.rct_type,
            "Reference RCT winner differs");
  }
  const auto views = Selected(input, selected);
  Check(image.channel.size() == views.size() &&
            image.nb_meta_channels == (selected.image.channel_count()
                                           ? selected.image.metadata_channels()
                                           : metadata),
        "Reference selected shape differs");
  for (size_t c = 0; c < views.size(); ++c) {
    const auto &r = image.channel[c];
    const auto n = views[c];
    Check(n.descriptor.extent == Extent2D{r.w, r.h},
          "Reference selected extent differs");
    if (c >= image.nb_meta_channels)
      Check(n.descriptor.hshift == r.hshift && n.descriptor.vshift == r.vshift,
            "Reference selected shifts differ");
    for (size_t y = 0; y < r.h; ++y)
      for (size_t x = 0; x < r.w; ++x)
        Check(n.Row(y)[x] == r.Row(y)[x], "Reference selected pixels differ");
  }
}
#endif
void Case(const Fixture &f) {
  const auto unchanged = f.backing;
  HostStorageBound bound{19, 23};
  ArmNextManagedHostAllocationFailureForTest();
  const auto planned = ComputeReferenceSelectionStorageBound(
      f.views, f.metadata, f.options, &bound);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Check(planned.ok() && pending, "Selector planner failed or allocated");
  ResourceBudget budget(std::max(size_t{1}, bound.peak_bytes));
  ResourceReservation reservation;
  Ok(budget.Reserve(std::max(size_t{1}, bound.peak_bytes), &reservation));
  ReferenceSelectionResult result;
  {
    ResourceContextScope scope({&reservation, ResourceClass::kInput});
    Ok(SelectReferenceModularEncoding(f.views, f.metadata, f.options, &result));
  }
  const auto snapshot = budget.snapshot();
  Check(
      snapshot.peak_backing_bytes <= bound.peak_bytes &&
          snapshot.total.live_capacity_bytes <= bound.retained_bytes &&
          snapshot.classes[size_t(ResourceClass::kInput)].backing_count == 0 &&
          snapshot.classes[size_t(ResourceClass::kPreparation)].backing_count ==
              snapshot.total.backing_count,
      "Selector exceeded its storage bound or used wrong ownership");
  peak = std::max(peak, snapshot.peak_backing_bytes);
  const auto views = Selected(f.views, result);
#ifdef GJXL_SELECTION_REFERENCE
  Reference(f.views, f.metadata, f.options, result);
#endif
  if (result.transforms.size) {
    auto original = Own(f.views, f.metadata);
    ChannelShape shape;
    Ok(DescribeImage(original, &shape));
    ModularImage forward, inverse;
    Ok(ForwardTransforms(original, result.transforms, &forward));
    Same(Views(forward), views);
    Ok(InverseTransforms(result.image, shape.channels(), f.metadata,
                         result.transforms, &inverse));
    Same(Views(inverse), f.views);
  } else {
    Check(result.image.channel_count() == 0 &&
              snapshot.total.backing_count == 0,
          "Identity selection unnecessarily owns samples");
    ++borrowed_results;
  }
  Check(f.backing == unchanged, "Selection modified borrowed input/padding");
  float previous_single = std::numeric_limits<float>::max();
  for (size_t i = 0; i < result.trace.palette_count; ++i) {
    const auto &decision = result.trace.palettes[i];
    palette_built += decision.built;
    palette_accepted += decision.accepted;
    palette_ties += decision.built && decision.before == decision.after;
    palette_rejected_cost += decision.built && !decision.accepted;
    Check(!decision.built ||
              decision.accepted == (decision.after <= decision.before),
          "Palette tie/acceptance rule differs");
    if (decision.count == 1 && decision.accepted) {
      shared_baseline_accepts += decision.after > previous_single;
      previous_single = decision.after;
    }
  }
  for (size_t i = 0; i < result.transforms.size; ++i)
    wide_palettes +=
        result.transforms.entries[i].kind == TransformKind::kPalette &&
        result.transforms.entries[i].colors > 1024;
  rct_candidates += result.trace.rct_count;
  wp_candidates += result.trace.wp_count;
  ++rct_winners[result.trace.rct];
  ++wp_winners[result.wp_mode];
  if (ledger.is_open()) {
    ledger << "{\"case\":" << cases << ",\"pixel_hash\":\"" << PixelHash(views)
           << "\",\"rct\":" << unsigned(result.trace.rct)
           << ",\"wp\":" << unsigned(result.wp_mode) << ",\"transforms\":[";
    for (size_t i = 0; i < result.transforms.size; ++i) {
      const auto &t = result.transforms.entries[i];
      ledger << (i ? "," : "") << '[' << unsigned(t.kind) << ','
             << unsigned(t.begin) << ',' << unsigned(t.count) << ',' << t.colors
             << ',' << unsigned(t.rct_type) << ',' << t.palette_ordered << ']';
    }
    ledger << "],\"gradient_bits\":[";
    for (size_t i = 0; i < result.trace.gradient_count; ++i)
      ledger << (i ? "," : "")
             << std::bit_cast<uint32_t>(result.trace.gradient_costs[i]);
    ledger << "],\"wp_bits\":[";
    for (size_t i = 0; i < result.trace.wp_count; ++i)
      ledger << (i ? "," : "")
             << std::bit_cast<uint32_t>(result.trace.wp_costs[i]);
    ledger << "],\"palettes\":[";
    for (size_t i = 0; i < result.trace.palette_count; ++i) {
      const auto &p = result.trace.palettes[i];
      ledger << (i ? "," : "") << '[' << unsigned(p.begin) << ','
             << unsigned(p.count) << ',' << p.maximum_colors << ',' << p.built
             << ',' << p.accepted << ',' << std::bit_cast<uint32_t>(p.before)
             << ',' << std::bit_cast<uint32_t>(p.after) << ']';
    }
    ledger << "],\"managed_peak\":" << snapshot.peak_backing_bytes << "}\n";
  }
  result.image = {};
  Check(budget.snapshot().total.backing_count == 0 &&
            budget.snapshot().total.pending_count == 0,
        "Selector leaked backing");
  ++cases;
}
void Faults(const Fixture &f, bool alias) {
  ReferenceSelectionResult expected;
  Ok(SelectReferenceModularEncoding(f.views, f.metadata, f.options, &expected));
  HostStorageBound bound;
  Ok(ComputeReferenceSelectionStorageBound(f.views, f.metadata, f.options,
                                           &bound));
  for (size_t fail = 0; fail < 512; ++fail) {
    ReferenceSelectionResult result;
    result.image = Own(f.views, f.metadata);
    const auto old_views = Views(result.image);
    const auto old_hash = PixelHash(old_views);
    const auto *old_pointer = old_views.back().backing.data();
    result.wp_mode = 4;
    result.trace.rct = 41;
    result.transforms.size = 1;
    const auto old_transforms = result.transforms;
    const auto old_trace = result.trace;
    ResourceBudget budget(std::max(size_t{1}, bound.peak_bytes));
    ResourceReservation reservation;
    Ok(budget.Reserve(std::max(size_t{1}, bound.peak_bytes), &reservation));
    Status status;
    {
      ResourceContextScope scope({&reservation, ResourceClass::kInput});
      ArmManagedHostClassAllocationFailureAfterForTest(
          ResourceClass::kPreparation, fail);
      status = SelectReferenceModularEncoding(alias ? old_views : f.views,
                                              f.metadata, f.options, &result);
      DisarmManagedHostAllocationFailureForTest();
    }
    Check(budget.snapshot().peak_backing_bytes <= bound.peak_bytes,
          "Fault exceeded selector bound");
    if (status.ok()) {
      Check(result.trace == expected.trace &&
                result.transforms == expected.transforms &&
                result.wp_mode == expected.wp_mode,
            "Allocation recovery changed selection");
      if (result.image.channel_count())
        Same(Views(result.image), Selected(f.views, expected));
      result.image = {};
      Check(budget.snapshot().total.backing_count == 0,
            "Recovered selector leaked backing");
      return;
    }
    Check(status.code() == StatusCode::kOutOfMemory && result.wp_mode == 4 &&
              result.trace == old_trace && result.transforms == old_transforms &&
              result.image.view(result.image.channel_count() - 1)
                      .backing.data() == old_pointer &&
              PixelHash(Views(result.image)) == old_hash &&
              budget.snapshot().total.backing_count == 0 &&
              budget.snapshot().total.pending_count == 0,
          "Allocation failure changed output or leaked backing");
    ++faults;
  }
  throw std::runtime_error("Selector allocation fault sweep did not terminate");
}
void Generated() {
  size_t id = 0;
  for (uint8_t bits : {1, 8, 16})
    for (size_t channels : {1, 2, 3, 4})
      for (auto stage :
           {ReferencePaletteStage::kNone, ReferencePaletteStage::kGlobal,
            ReferencePaletteStage::kLocal})
        for (size_t pattern = 0; pattern < 12; ++pattern, ++id) {
          const Extent2D extent =
              pattern < 2 ? Extent2D{3, 1} : Extent2D{33, 17};
          auto f = Make(channels, bits, pattern, extent, id % 3);
          f.options.palettes = stage;
          f.options.palette_colors = id % 7 == 0   ? 0
                                     : id % 7 == 1 ? -16
                                     : id % 7 == 2 ? 2
                                                   : 1024;
          try {
            Case(f);
          } catch (...) {
            std::cerr << "bits=" << unsigned(bits) << " channels=" << channels
                      << " stage=" << unsigned(stage) << " pattern=" << pattern
                      << " meta=" << f.metadata << " case=" << id << '\n';
            throw;
          }
        }
  for (size_t pattern : {2, 3, 4, 6, 7, 8, 9, 10, 11})
    for (int colors : {0, 1, 2, 16, -1024, 1024}) {
      auto f = Make(4, 8, pattern, {65, 33});
      f.options.palette_colors = colors;
      Case(f);
    }
  for (auto stage :
       {ReferencePaletteStage::kGlobal, ReferencePaletteStage::kLocal}) {
    auto f = Make(1, 16, 12, {2048, 33});
    f.options.palettes = stage;
    Case(f);
    auto noise = Make(1, 16, 13, {2048, 33});
    noise.options.palettes = stage;
    Case(noise);
  }
  // Source and palette representation both cost six bits in the qualified
  // reference estimator. This nonuniform proposal is accepted on equality.
  auto tie = Make(3, 8, 0, {2, 1});
  tie.backing[0][0] = 1;
  tie.backing[1][1] = 3;
  tie.backing[2][0] = 1;
  Case(tie);
  // The second compaction is slightly worse than the preceding accepted
  // image, but still better than the shared baseline. Updating the baseline
  // after each single-channel acceptance would incorrectly reject it.
  auto baseline = Make(2, 16, 13, {2048, 33});
  baseline.options.palette_colors = 0;
  for (size_t y = 0; y < 33; ++y)
    for (size_t x = 0; x < 2048; ++x)
      baseline.backing[1][y * baseline.views[1].stride + x] = x * 31;
  Case(baseline);
  Case(Make(0, 8, 0, {3, 1}, 1));
  Case(Make(0, 8, 0, {3, 1}, 2));
  // Shifted/unequal alpha rejects RGBA but permits RGB trials and RCT.
  auto alpha = Make(4, 8, 7, {33, 17});
  alpha.views.back().descriptor.hshift = 1;
  Case(alpha);
  auto disabled = Make(3, 8, 4, {33, 17});
  disabled.options = {ReferencePaletteStage::kNone, 8, 1024, false, false};
  Case(disabled);
  for (size_t pattern : {2, 3, 4, 6, 7, 8}) {
    auto f = Make(4, 8, pattern, {33, 17});
    if (pattern == 8)
      f.options.palette_colors = 0;
    Faults(f, false);
  }
  Faults(Make(3, 8, 4, {33, 17}), true);
  auto identity = Make(3, 8, 0, {3, 1});
  Faults(identity, true);
  identity.options = {ReferencePaletteStage::kNone, 8, 1024, false, false};
  Faults(identity, true);
}
void Contracts() {
  auto f = Make(3, 8, 4, {17, 3});
  ReferenceSelectionResult result;
  result.wp_mode = 4;
  result.trace.rct = 41;
  for (size_t error = 0; error < 13; ++error) {
    auto views = f.views;
    auto options = f.options;
    size_t metadata = 0;
    if (error == 0)
      options.bit_depth = 0;
    if (error == 1)
      options.bit_depth = 17;
    if (error == 2)
      options.palettes = ReferencePaletteStage(99);
    if (error == 3)
      options.palette_colors = 1025;
    if (error == 4)
      options.palette_colors = -1025;
    if (error == 5)
      metadata = 4;
    if (error == 6)
      metadata = 1;
    if (error == 7)
      views.back().stride = 0;
    if (error == 8)
      views.back().backing = views.back().backing.first(1);
    if (error == 9)
      views.back().descriptor.hshift = 1;
    if (error == 10)
      views.back().descriptor.extent.height = SIZE_MAX;
    if (error == 11)
      views.clear();
    if (error == 12)
      views.resize(5, f.views[0]);
    HostStorageBound bound{19, 23};
    ArmNextManagedHostAllocationFailureForTest();
    const auto plan =
        ComputeReferenceSelectionStorageBound(views, metadata, options, &bound);
    const auto status =
        SelectReferenceModularEncoding(views, metadata, options, &result);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(plan.code() == StatusCode::kInvalidArgument &&
              status.code() == plan.code() && pending &&
              bound == HostStorageBound{19, 23} && result.wp_mode == 4 &&
              result.trace.rct == 41,
          "Invalid selector shape/options allocated or changed output");
  }
  for (int32_t invalid : {-1, 65536}) {
    f.backing[2][0] = invalid;
    HostStorageBound bound;
    ArmNextManagedHostAllocationFailureForTest();
    const auto plan =
        ComputeReferenceSelectionStorageBound(f.views, 0, f.options, &bound);
    const auto status =
        SelectReferenceModularEncoding(f.views, 0, f.options, &result);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(plan.ok() && status.code() == StatusCode::kInvalidArgument &&
              pending && result.wp_mode == 4 && result.trace.rct == 41,
          "Planner read pixels or invalid samples changed output");
  }
  f.backing[2][0] = 0;
  // Partial output overlap must not invalidate input when identity is chosen.
  {
    ReferenceSelectionResult overlapping;
    overlapping.image = Own(f.views, 0);
    auto borrowed = Views(overlapping.image);
    borrowed.back().backing = borrowed.back().backing.subspan(1);
    borrowed.back().descriptor.extent.height = 1;
    borrowed.back().descriptor.extent.width = 1;
    const auto hash = PixelHash(Views(overlapping.image));
    auto options = f.options;
    options.select_rct = false;
    Check(SelectReferenceModularEncoding(borrowed, 0, options, &overlapping)
                      .code() == StatusCode::kInvalidArgument &&
              PixelHash(Views(overlapping.image)) == hash,
          "Partial overlap was not rejected atomically");
  }
  Check(
      !SelectReferenceModularEncoding(f.views, 0, f.options, nullptr).ok() &&
          !ComputeReferenceSelectionStorageBound(f.views, 0, f.options, nullptr)
               .ok(),
      "Null selector output accepted");
  ResourceBudget tiny(1);
  ResourceReservation reservation;
  Ok(tiny.Reserve(1, &reservation));
  {
    ResourceContextScope scope({&reservation, ResourceClass::kInput});
    Check(
        SelectReferenceModularEncoding(f.views, 0, f.options, &result).code() ==
                StatusCode::kOutOfMemory &&
            result.wp_mode == 4 && result.trace.rct == 41 &&
            tiny.snapshot().total.backing_count == 0,
        "Underbudget selector changed output");
  }
}
#ifdef GJXL_SELECTION_REFERENCE
void FileCase(uint8_t bits, size_t count, size_t pattern, bool local) {
  namespace ref = gjxl::test::modular_reference;
  const Extent2D extent = local ? Extent2D{513, 37} : Extent2D{65, 33};
  auto f = Make(count, bits, pattern, extent);
  f.options.select_rct = f.options.select_wp = !local;
  if (pattern == 8)
    f.options.palette_colors = 0;
  ReferenceSelectionResult global;
  Ok(SelectReferenceModularEncoding(f.views, 0, f.options, &global));
  Reference(f.views, 0, f.options, global);
  auto original = Own(f.views, 0);
  const auto &image = global.image.channel_count() ? global.image : original;
  ChannelShape shape;
  Ok(DescribeImage(image, &shape));
  ModularFrameGeometry geometry;
  Ok(ModularFrameGeometry::Create(extent, &geometry));
  ModularStreamPlan layout;
  Ok(BuildModularStreamPlan(geometry, shape.channels(), shape.metadata,
                            &layout));
  std::vector<ReferenceSelectionResult> choices(layout.streams.size());
  std::vector<std::vector<ModularChannelView>> views(layout.streams.size());
  std::vector<PreparedModularStreamView> streams(layout.streams.size());
  for (size_t i = 0; i < streams.size(); ++i) {
    for (auto slice : std::span(layout.slices)
                          .subspan(layout.streams[i].slice_begin,
                                   layout.streams[i].slice_count)) {
      ModularChannelView view;
      Ok(BorrowChannelSlice(image.view(slice.channel), slice.rect, &view));
      views[i].push_back(view);
    }
    streams[i].weighted = kSelectionWeightedPresets[global.wp_mode];
    if (i == 0)
      streams[i].transforms = global.transforms;
    if (i && local && !views[i].empty()) {
      auto options = f.options;
      options.palettes = ReferencePaletteStage::kLocal;
      options.select_rct = options.select_wp = true;
      Ok(SelectReferenceModularEncoding(views[i], 0, options, &choices[i]));
      Reference(views[i], 0, options, choices[i]);
      streams[i].transforms = choices[i].transforms;
      streams[i].weighted = kSelectionWeightedPresets[choices[i].wp_mode];
      views[i] = Selected(views[i], choices[i]);
    }
    streams[i].channels = views[i];
  }
  ref::IntegerImage expected{
      extent.width, extent.height, uint32_t(count), bits, {}};
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < extent.width; ++x)
      for (size_t c = 0; c < count; ++c)
        expected.samples.push_back(f.views[c].Row(y)[x]);
  const auto format = count == 1   ? (bits == 8 ? PackedModularFormat::kGray8
                                                : PackedModularFormat::kGray16)
                      : count == 3 ? (bits == 8 ? PackedModularFormat::kRgb8
                                                : PackedModularFormat::kRgb16)
                                   : (bits == 8 ? PackedModularFormat::kRgba8
                                                : PackedModularFormat::kRgba16);
  DecisionTree tree;
  tree.nodes[0].predictor = Predictor::kWeighted;
  for (bool local_tree : {false, true}) {
    for (auto &stream : streams) {
      const bool samples = std::ranges::any_of(stream.channels, [](auto ch) {
        return !ch.descriptor.extent.empty();
      });
      stream.local_tree = local_tree && samples ? &tree : nullptr;
    }
    PreparedModularStreamsView input{extent, format,
                                     local_tree ? nullptr : &tree, streams};
    for (auto entropy : {PreparedModularEntropyPolicy::kPrefix,
                         PreparedModularEntropyPolicy::kAnsFromPrefix,
                         PreparedModularEntropyPolicy::kDirectRateAns}) {
      PreparedModularEntropyOptions options{entropy,
                                            AnsSelectionMode::kEstimated};
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
            "Selected file exceeded bound");
      Check(ref::DecodeLossless(encoded.view()) == expected,
            "Selected file decode differs");
      if (!artifacts.empty()) {
        std::ofstream file(artifacts /
                               ("selected-" + std::to_string(files) + ".jxl"),
                           std::ios::binary);
        file.write(reinterpret_cast<const char *>(encoded.view().data()),
                   encoded.size());
        Check(bool(file), "Writing selected codestream failed");
      }
      ++files;
    }
  }
}
#endif
void Summary(std::ostream &out) {
  out << "{\"cases\":" << cases << ",\"faults\":" << faults
      << ",\"files\":" << files << ",\"managed_peak\":" << peak
      << ",\"palette_built\":" << palette_built
      << ",\"palette_accepted\":" << palette_accepted
      << ",\"palette_ties\":" << palette_ties
      << ",\"palette_rejected_cost\":" << palette_rejected_cost
      << ",\"borrowed_results\":" << borrowed_results
      << ",\"wide_palettes\":" << wide_palettes
      << ",\"rct_candidates\":" << rct_candidates
      << ",\"shared_baseline_accepts\":" << shared_baseline_accepts
      << ",\"wp_candidates\":" << wp_candidates << ",\"rct_winners\":[";
  for (size_t i = 0; i < rct_winners.size(); ++i)
    out << (i ? "," : "") << rct_winners[i];
  out << "],\"wp_winners\":[";
  for (size_t i = 0; i < wp_winners.size(); ++i)
    out << (i ? "," : "") << wp_winners[i];
  out << "]}\n";
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc > 1) {
      artifacts = argv[1];
      std::filesystem::create_directories(artifacts);
      ledger.open(artifacts / "decisions.jsonl");
      Check(bool(ledger), "Opening selection ledger failed");
    }
    Generated();
    Contracts();
#ifdef GJXL_SELECTION_REFERENCE
    for (uint8_t bits : {8, 16})
      for (size_t channels : {1, 3, 4})
        for (size_t pattern : {2, 4, 8})
          for (bool local : {false, true})
            FileCase(bits, channels, pattern, local);
#endif
    if (!(palette_ties && palette_rejected_cost && wide_palettes &&
          borrowed_results && shared_baseline_accepts))
      Summary(std::cerr);
    Check(palette_ties && palette_rejected_cost && wide_palettes &&
              borrowed_results && shared_baseline_accepts,
          "Missing palette tie/rejection/wide/borrowed coverage");
    Summary(std::cout);
    if (!artifacts.empty()) {
      std::ofstream summary(artifacts / "summary.json");
      Summary(summary);
      Check(bool(summary), "Writing selection summary failed");
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
