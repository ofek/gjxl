// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/prepared_streams.h"
#include "codestream/modular/workflow.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/enc_ma.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/pack_signed.h"
#include "modular_reference.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
namespace ref = gjxl::test::modular_reference;
void Check(bool ok, const char *m) {
  if (!ok)
    throw std::runtime_error(m);
}
void Ok(Status s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
TransformDescriptor Rct(uint8_t begin, uint8_t type) {
  TransformDescriptor t{TransformKind::kRct, begin, 3};
  t.rct_type = type;
  return t;
}
WeightedPredictorParameters Weighted(size_t mode) {
  jxl::weighted::Header header;
  jxl::weighted::PredictorMode(mode, &header);
  WeightedPredictorParameters p;
  p.coefficients = {static_cast<uint8_t>(header.p1C),  static_cast<uint8_t>(header.p2C),
                    static_cast<uint8_t>(header.p3Ca), static_cast<uint8_t>(header.p3Cb),
                    static_cast<uint8_t>(header.p3Cc), static_cast<uint8_t>(header.p3Cd),
                    static_cast<uint8_t>(header.p3Ce)};
  std::copy(std::begin(header.w), std::end(header.w), p.weights.begin());
  return p;
}
jxl::weighted::Header ReferenceHeader(const WeightedPredictorParameters &p) {
  jxl::weighted::Header h;
  h.p1C = p.coefficients[0];
  h.p2C = p.coefficients[1];
  h.p3Ca = p.coefficients[2];
  h.p3Cb = p.coefficients[3];
  h.p3Cc = p.coefficients[4];
  h.p3Cd = p.coefficients[5];
  h.p3Ce = p.coefficients[6];
  std::copy(p.weights.begin(), p.weights.end(), h.w);
  return h;
}
void CheckTokens(const PreparedModularStreamsView &input, const PreparedModularStreamTokens &tokens,
                 const ModularStreamPlan &plan) {
  JxlMemoryManager memory;
  Check(bool(jxl::MemoryManagerInit(&memory, nullptr)), "Reference allocation setup failed");
  for (size_t i = 0; i < input.streams.size(); ++i) {
    const auto &stream = input.streams[i];
    if (stream.channels.empty())
      continue;
    const auto &tree = *(stream.local_tree ? stream.local_tree : input.global_tree);
    jxl::Tree raw, decoded;
    for (size_t n = 0; n < tree.size; ++n) {
      const auto &v = tree.nodes[n];
      raw.emplace_back(v.property, v.split, v.left, v.right,
                       static_cast<jxl::Predictor>(v.predictor), v.offset, v.multiplier);
    }
    std::vector<jxl::Token> tree_tokens;
    Check(bool(jxl::TokenizeTree(raw, &tree_tokens, &decoded)),
          "Reference tree serialization failed");
    auto actual = stream.local_tree ? tokens.local_tree_tokens[i] : tokens.global_tree_tokens;
    Check(actual.size() == tree_tokens.size(), "Scoped tree token count differs");
    for (size_t n = 0; n < actual.size(); ++n)
      Check(actual[n].value == tree_tokens[n].value && actual[n].context == tree_tokens[n].context,
            "Scoped tree tokens differ");
    const auto header = ReferenceHeader(stream.weighted);
    size_t next = 0;
    for (size_t c = 0; c < stream.channels.size(); ++c) {
      const auto &view = stream.channels[c];
      const auto [w, h] = view.descriptor.extent;
      auto created = jxl::Channel::Create(&memory, w, h),
           refs_created = jxl::Channel::Create(&memory, 0, w);
      Check(created.ok() && refs_created.ok(), "Reference channel allocation failed");
      auto channel = std::move(created).value_(), refs = std::move(refs_created).value_();
      for (size_t y = 0; y < h; ++y)
        std::copy_n(view.Row(y), w, channel.Row(y));
      jxl::weighted::State state(header, w, h);
      jxl::Properties properties(16);
      for (size_t y = 0; y < h; ++y) {
        jxl::InitPropsRow(&properties,
                          {static_cast<int32_t>(c), static_cast<int32_t>(plan.streams[i].id)}, y);
        for (size_t x = 0; x < w; ++x) {
          std::array<int64_t, 14> predictions;
          jxl::PredictLearnAll(&properties, w, channel.Row(y) + x, channel.plane.PixelsPerRow(), x,
                               y, refs, &state, predictions.data());
          size_t leaf = 0;
          while (decoded[leaf].property != -1) {
            const auto &n = decoded[leaf];
            leaf = properties[n.property] > n.splitval ? n.lchild : n.rchild;
          }
          const auto &n = decoded[leaf];
          const int64_t residual = int64_t{channel.Row(y)[x]} -
                                   predictions[static_cast<size_t>(n.predictor)] -
                                   n.predictor_offset;
          Check(residual % n.multiplier == 0, "Reference residual not representable");
          Check(next < tokens.streams[i].size() && tokens.streams[i][next].context == n.lchild &&
                    tokens.streams[i][next].value ==
                        jxl::PackSigned(static_cast<int32_t>(residual / n.multiplier)),
                "Per-stream sample token differs from pinned reference");
          ++next;
          state.UpdateErrors(channel.Row(y)[x], x, y, w);
        }
      }
    }
    Check(next == tokens.streams[i].size(), "Per-stream token population differs");
  }
}
size_t cases = 0, files = 0;
std::filesystem::path artifacts;
void Case(PackedModularFormat format, size_t scenario, size_t scope) {
  constexpr std::array extents = {Extent2D{17, 19},   Extent2D{257, 3},   Extent2D{513, 257},
                                  Extent2D{257, 257}, Extent2D{2057, 17}, Extent2D{3, 257}};
  const auto extent = extents[scenario];
  ModularInputProfile profile;
  Ok(ResolveModularInput(extent, format, &profile));
  ref::IntegerImage original{extent.width,
                             extent.height,
                             static_cast<uint32_t>(profile.channel_count),
                             profile.metadata.bits,
                             {}};
  const size_t row = extent.width * profile.channel_count * profile.bytes_per_sample;
  std::vector<uint8_t> bytes(row * extent.height);
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < extent.width; ++x)
      for (size_t c = 0; c < profile.channel_count; ++c) {
        const size_t v = (x + 7 * y) % 64;
        uint16_t sample = (v * (2 * c + 1) * (profile.metadata.bits == 16 ? 991 : 1) + c * 19) &
                          (profile.metadata.bits == 16 ? 65535 : 255);
        if (c == 3 && v % 3 == 0)
          sample = 0;
        original.samples.push_back(sample);
        size_t pos = y * row + (x * profile.channel_count + c) * profile.bytes_per_sample;
        bytes[pos] = sample;
        if (profile.bytes_per_sample == 2)
          bytes[pos + 1] = sample >> 8;
      }
  PackedModularImageView packed{bytes, extent, row, format};
  ModularCodingPolicy global_policy;
  if (scenario == 3) {
    global_policy.transforms.size = 1;
    global_policy.transforms.entries[0] = {TransformKind::kPalette, 0, 1, 1024};
  }
  if (scenario == 4) {
    global_policy.transforms.size = 6;
    for (size_t i = 0; i < 6; ++i)
      global_policy.transforms.entries[i] = {TransformKind::kSqueeze,
                                             0,
                                             static_cast<uint8_t>(profile.channel_count),
                                             1,
                                             i % 2 == 0,
                                             false};
  }
  ModularEncoderFrame full;
  Ok(ModularEncoderFrame::Prepare(packed, global_policy, &full));
  ChannelShape full_shape;
  Ok(DescribeImage(full.image(), &full_shape));
  ModularFrameGeometry geometry;
  Ok(ModularFrameGeometry::Create(extent, &geometry));
  ModularStreamPlan layout;
  Ok(BuildModularStreamPlan(geometry, full_shape.channels(), full_shape.metadata, &layout));
  std::array<DecisionTree, 3> trees;
  for (size_t i = 1; i < trees.size(); ++i) {
    auto &tree = trees[i];
    tree.size = 3;
    tree.nodes[0] = {.property = static_cast<int32_t>(i - 1),
                     .split = static_cast<int32_t>(i == 1 ? 0 : 22),
                     .left = 1,
                     .right = 2};
    tree.nodes[1].predictor = Predictor::kWeighted;
    tree.nodes[2].predictor = i == 1 ? Predictor::kGradient : Predictor::kAverage4;
  }
  std::vector<ModularImage> owned(layout.streams.size());
  std::vector<std::vector<ModularChannelView>> views(layout.streams.size());
  std::vector<PreparedModularStreamView> streams(layout.streams.size());
  for (size_t i = 0; i < streams.size(); ++i) {
    const auto &placed = layout.streams[i];
    auto &stream = streams[i];
    auto slices = std::span(layout.slices).subspan(placed.slice_begin, placed.slice_count);
    if (i == 0) {
      stream.rct = global_policy.rct;
      stream.transforms = global_policy.transforms;
      for (const auto &slice : slices) {
        ModularChannelView view;
        Ok(BorrowChannelSlice(full.image().view(slice.channel), slice.rect, &view));
        views[i].push_back(view);
      }
    } else if (!slices.empty()) {
      std::vector<ChannelDescriptor> shape;
      for (const auto &slice : slices) {
        auto ch = full.image().view(slice.channel).descriptor;
        ch.extent = slice.rect.extent;
        shape.push_back(ch);
      }
      ModularImage tile;
      Ok(ModularImage::Create(shape, 0, &tile));
      for (size_t c = 0; c < slices.size(); ++c) {
        ModularChannelView view;
        Ok(BorrowChannelSlice(full.image().view(slices[c].channel), slices[c].rect, &view));
        for (size_t y = 0; y < view.descriptor.extent.height; ++y)
          std::copy_n(view.Row(y), view.descriptor.extent.width,
                      tile.samples(c).data() + y * view.descriptor.extent.width);
      }
      if (scenario == 1 || scenario == 3 || scenario == 4) {
        bool equal = shape.size() >= 3;
        for (size_t c = 1; c < std::min(size_t{3}, shape.size()); ++c)
          equal &= shape[c].extent == shape[0].extent && shape[c].hshift == shape[0].hshift &&
                   shape[c].vshift == shape[0].vshift;
        if (equal) {
          stream.rct = std::array<uint8_t, 3>{8, 0, 40}[i % 3];
          TransformSequence leading;
          leading.size = 1;
          leading.entries[0] = Rct(0, stream.rct);
          Ok(ForwardTransforms(tile, leading, &tile));
        }
      }
      if (scenario == 2 || scenario == 5) {
        stream.transforms.size = 1;
        const uint8_t count =
            scenario == 5 ? static_cast<uint8_t>(std::min(size_t{3}, shape.size())) : 1;
        stream.transforms.entries[0] = {TransformKind::kPalette, 0, count, 1024};
        if (scenario == 2 && shape.size() >= 3) {
          stream.transforms.size = 2;
          stream.transforms.entries[1] = Rct(1, 33);
        }
      }
      Ok(ForwardTransforms(tile, stream.transforms, &owned[i]));
      for (size_t c = 0; c < owned[i].channel_count(); ++c)
        views[i].push_back(owned[i].view(c));
    }
    stream.channels = views[i];
    if (i == 0 || !slices.empty()) {
      if (scenario)
        stream.weighted = Weighted(i % 5);
      if (scope == 1 || (scope == 2 && i % 2 == 0))
        stream.local_tree = &trees[(i % 2) + 1];
    }
  }
  PreparedModularStreamsView input{extent, format, scope == 1 ? nullptr : &trees[scenario ? 1 : 0],
                                   streams};
  {
    PreparedModularStreamTokens tokens;
    Ok(TokenizePreparedModularStreams(input, &tokens));
    CheckTokens(input, tokens, layout);
  }
  for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    ModularWorkflowStoragePlan plan;
    Ok(ComputePreparedModularStreamsStoragePlan(input, mode, &plan));
    ResourceBudget budget(plan.working.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(plan.working.peak_bytes, &reservation));
    gjxl::codestream_internal::CodestreamBuffer encoded;
    {
      ResourceContextScope resource({&reservation, ResourceClass::kPreparation});
      Ok(EncodePreparedModularStreams(input, mode, &encoded));
    }
    Check(encoded.size() <= plan.maximum_codestream_bytes &&
              budget.snapshot().peak_backing_bytes <= plan.working.peak_bytes,
          "Scoped encoder exceeded managed storage bound");
    Check(ref::DecodeLossless(encoded.view()) == original, "Scoped file does not decode exactly");
    if (scenario == 0 && scope == 0) {
      ModularEncodingOptions options;
      options.entropy = mode;
      std::vector<uint8_t> legacy;
      Ok(EncodeModularImage(packed, options, &legacy));
      Check(std::ranges::equal(encoded.view(), legacy), "Shared-tree bridge changed legacy bytes");
    }
    if (!artifacts.empty()) {
      std::ofstream out(artifacts / ("case-" + std::to_string(files) + ".jxl"), std::ios::binary);
      out.write(reinterpret_cast<const char *>(encoded.data()), encoded.size());
      Check(bool(out), "Artifact write failed");
    }
    encoded.Reset();
    Check(budget.snapshot().total.backing_count == 0, "Scoped result leaked backing");
    ++files;
  }
  ++cases;
}
} // namespace
int main(int argc, char **argv) try {
  if (argc > 1) {
    artifacts = argv[1];
    std::filesystem::create_directories(artifacts);
  }
  for (auto format :
       {PackedModularFormat::kGray8, PackedModularFormat::kRgb8, PackedModularFormat::kRgba8,
        PackedModularFormat::kGray16, PackedModularFormat::kRgb16, PackedModularFormat::kRgba16})
    for (size_t scenario = 0; scenario < 6; ++scenario)
      for (size_t scope = 0; scope < 3; ++scope) {
        try {
          Case(format, scenario, scope);
        } catch (...) {
          std::cerr << "format " << int(format) << " scenario " << scenario << " scope " << scope
                    << '\n';
          throw;
        }
      }
  std::cout << "Prepared stream reference: " << cases << " cases, " << files << " exact files\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
