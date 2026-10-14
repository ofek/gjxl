// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/workflow.h"
#include "codestream/modular/storage_plan.h"
#include "codestream/modular/search.h"
#include "codestream/modular/stream_encoder.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/enc_ma.h"
#include "lib/jxl/pack_signed.h"
#include "modular_reference.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/transform.h"
#include "lib/jxl/modular/transform/enc_squeeze.h"
#include "lib/jxl/modular/transform/enc_rct.h"
#include <map>
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <fstream>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
namespace ref = gjxl::test::modular_reference;
size_t cases = 0, files = 0, headers = 0;
std::filesystem::path artifacts;
void Check(bool b, const char *m) {
  if (!b)
    throw std::runtime_error(m);
}
void Ok(Status s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
void ReferenceStep(jxl::Image &image, const TransformDescriptor &t) {
  if (t.kind == TransformKind::kRct) {
    // The reference's image overload returns false for the identity no-op.
    if (t.rct_type != 0)
      Check(bool(jxl::FwdRct(image, t.begin, t.rct_type, nullptr)), "Pinned ordered RCT failed");
    return;
  }
  if (t.kind == TransformKind::kSqueeze) {
    jxl::SqueezeParams params;
    params.horizontal = t.horizontal;
    params.in_place = t.in_place;
    params.begin_c = t.begin;
    params.num_c = t.count;
    Check(bool(jxl::FwdSqueeze(image, {params}, nullptr)), "Pinned forward squeeze failed");
    return;
  }
  // Prescribed exact palette: independent ordered dictionary and pinned shape application.
  std::map<std::vector<int32_t>, int32_t> dictionary;
  std::vector<std::vector<int32_t>> pixels;
  const size_t w = image.channel[t.begin].w, h = image.channel[t.begin].h;
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x) {
      std::vector<int32_t> v;
      for (size_t c = 0; c < t.count; ++c)
        v.push_back(image.channel[t.begin + c].Row(y)[x]);
      dictionary.emplace(v, 0);
      pixels.push_back(std::move(v));
    }
  Check(dictionary.size() <= t.colors, "Bad palette fixture");
  jxl::Transform transform(jxl::TransformId::kPalette);
  transform.begin_c = t.begin;
  transform.num_c = t.count;
  transform.nb_colors = t.colors;
  transform.nb_deltas = 0;
  Check(bool(transform.MetaApply(image)), "Pinned palette shape failed");
  for (size_t c = 0; c < t.count; ++c)
    std::fill_n(image.channel[0].Row(c), t.colors, 0);
  int32_t index = 0;
  for (auto &[color, id] : dictionary) {
    id = index++;
    for (size_t c = 0; c < t.count; ++c)
      image.channel[0].Row(c)[id] = color[c];
  }
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x)
      image.channel[t.begin + 1].Row(y)[x] = dictionary.at(pixels[y * w + x]);
}
// Evaluate every learned token with pinned libjxl prediction and tree traversal
// over independently transformed samples. The stream-layout contract is shared;
// pixel reconstruction is additionally checked by the independent decoder.
void LearnedTokens(const ModularEncoderFrame &frame, const ModularStreamPlan &plan,
                   const ModularCodingPolicy &policy, const jxl::Image &reference,
                   JxlMemoryManager *memory) {
  PreparedModularTokens tokens;
  Ok(TokenizeModular(frame, plan, policy, &tokens));
  jxl::Tree tree, decoded;
  for (size_t i = 0; i < policy.tree.size; ++i) {
    const auto &n = policy.tree.nodes[i];
    tree.emplace_back(n.property, n.split, n.left, n.right,
                      static_cast<jxl::Predictor>(n.predictor), n.offset, n.multiplier);
  }
  std::vector<jxl::Token> tree_tokens;
  Check(bool(jxl::TokenizeTree(tree, &tree_tokens, &decoded)), "Learned tree tokenization failed");
  Check(tree_tokens.size() == tokens.tree_token_count, "Learned tree token count differs");
  for (size_t i = 0; i < tree_tokens.size(); ++i)
    Check(tree_tokens[i].context == tokens.tree_tokens[i].context &&
              tree_tokens[i].value == tokens.tree_tokens[i].value, "Learned tree token differs");
  jxl::weighted::Header header;
  const auto &p = policy.weighted.coefficients;
  header.p1C = p[0]; header.p2C = p[1]; header.p3Ca = p[2]; header.p3Cb = p[3];
  header.p3Cc = p[4]; header.p3Cd = p[5]; header.p3Ce = p[6];
  for (size_t i = 0; i < 4; ++i) header.w[i] = policy.weighted.weights[i];
  for (size_t si = 0; si < plan.streams.size(); ++si) {
    const auto &stream = plan.streams[si];
    size_t token = 0, property_channel = 0;
    for (const auto &slice : std::span(plan.slices).subspan(stream.slice_begin, stream.slice_count)) {
      const auto [w, h] = slice.rect.extent;
      auto tile_result = jxl::Channel::Create(memory, w, h);
      auto refs_result = jxl::Channel::Create(memory, 0, w);
      Check(tile_result.ok() && refs_result.ok(), "Learned reference allocation failed");
      auto tile = std::move(tile_result).value_(), refs = std::move(refs_result).value_();
      for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x)
          tile.Row(y)[x] = reference.channel[slice.channel].Row(slice.rect.y + y)
                                                      [slice.rect.x + x];
      jxl::weighted::State state(header, w, h);
      jxl::Properties properties(16);
      for (size_t y = 0; y < h; ++y) {
        jxl::InitPropsRow(&properties,
                          {static_cast<int32_t>(property_channel), static_cast<int32_t>(stream.id)},
                          static_cast<int>(y));
        for (size_t x = 0; x < w; ++x) {
          std::array<int64_t, 14> predictions{};
          jxl::PredictLearnAll(&properties, w, tile.Row(y) + x, tile.plane.PixelsPerRow(),
                               static_cast<int>(x), static_cast<int>(y), refs, &state,
                               predictions.data());
          size_t leaf = 0;
          while (decoded[leaf].property != -1) {
            const auto &n = decoded[leaf];
            leaf = properties[n.property] > n.splitval ? n.lchild : n.rchild;
          }
          const auto &n = decoded[leaf];
          const int64_t residual = tile.Row(y)[x] - predictions[static_cast<size_t>(n.predictor)] -
                                   n.predictor_offset;
          Check(residual % n.multiplier == 0, "Learned leaf cannot represent sample");
          Check(token < tokens.streams[si].size() &&
                    tokens.streams[si][token].value ==
                        jxl::PackSigned(static_cast<int32_t>(residual / n.multiplier)) &&
                    tokens.streams[si][token].context == n.lchild,
                "Learned transformed token differs from pinned libjxl");
          ++token;
          state.UpdateErrors(tile.Row(y)[x], x, y, w);
        }
      }
      ++property_channel;
    }
    Check(token == tokens.streams[si].size(), "Learned token coverage differs");
  }
}
void Case(Extent2D extent, PackedModularFormat format, size_t colors, ModularCodingPolicy policy,
          bool learn = false, bool check_tokens = false) {
  ModularInputProfile profile;
  Ok(ResolveModularInput(extent, format, &profile));
  ref::IntegerImage expected{extent.width,
                             extent.height,
                             static_cast<uint32_t>(profile.channel_count),
                             profile.metadata.bits,
                             {}};
  const size_t row = extent.width * profile.channel_count * profile.bytes_per_sample,
               stride = row + 3;
  std::vector<uint8_t> bytes((extent.height - 1) * stride + row + 1, 0xcd);
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < extent.width; ++x)
      for (size_t c = 0; c < profile.channel_count; ++c) {
        const size_t v = colors ? ((x + 7 * y) % colors) : (x * 739 + y * 131 + x * y * 37);
        uint16_t sample = static_cast<uint16_t>((v * (c * 2 + 1) + c * 19) &
                                                (profile.metadata.bits == 8 ? 255 : 65535));
        // Exercise more than 256 distinct RGB tuples in the new 8-bit cases.
        if (colors > 256 && profile.metadata.bits == 8)
          sample = static_cast<uint16_t>((v >> (c * 4)) & 255);
        if (c == 3 && v % 3 == 0)
          sample = 0;
        expected.samples.push_back(sample);
        const size_t i =
            1 + y * stride + (x * profile.channel_count + c) * profile.bytes_per_sample;
        bytes[i] = static_cast<uint8_t>(sample);
        if (profile.bytes_per_sample == 2)
          bytes[i + 1] = static_cast<uint8_t>(sample >> 8);
      }
  const auto unchanged = bytes;
  PackedModularImageView input{std::span(bytes).subspan(1), extent, stride, format};
  ModularEncoderFrame initial, transformed;
  Ok(ModularEncoderFrame::Prepare(input, policy.rct, &initial));
  Ok(ModularEncoderFrame::Prepare(input, policy, &transformed));
  ModularImage restored;
  Ok(InverseTransforms(transformed.image(), profile.channels(), 0, policy.transforms, &restored));
  for (size_t c = 0; c < profile.channel_count; ++c)
    Check(std::ranges::equal(restored.view(c).backing, initial.image().view(c).backing),
          "Native inverse mismatch");
  JxlMemoryManager memory;
  Check(bool(jxl::MemoryManagerInit(&memory, nullptr)), "Pinned allocator failed");
  auto created =
      jxl::Image::Create(&memory, extent.width, extent.height, expected.bits, expected.channels);
  Check(created.ok(), "Pinned image failed");
  auto reference = std::move(created).value_();
  for (size_t c = 0; c < expected.channels; ++c)
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        reference.channel[c].Row(y)[x] =
            expected.samples[(y * extent.width + x) * expected.channels + c];
  if (policy.rct) {
    auto next =
        jxl::Image::Create(&memory, extent.width, extent.height, expected.bits, expected.channels);
    Check(next.ok(), "Pinned RCT allocation failed");
    auto rct = std::move(next).value_();
    Check(
        bool(jxl::FwdRct({&reference.channel[0], &reference.channel[1], &reference.channel[2]},
                         {&rct.channel[0], &rct.channel[1], &rct.channel[2]}, policy.rct, nullptr)),
        "Pinned RCT failed");
    if (expected.channels == 4)
      rct.channel[3] = std::move(reference.channel[3]);
    reference = std::move(rct);
  }
  TransformSequence prefix;
  for (size_t i = 0; i < policy.transforms.size; ++i) {
    prefix.entries[i] = policy.transforms.entries[i];
    ++prefix.size;
    ReferenceStep(reference, prefix.entries[i]);
    ModularImage native;
    Ok(ForwardTransforms(initial.image(), prefix, &native));
    Check(native.channel_count() == reference.channel.size() &&
              native.metadata_channels() == reference.nb_meta_channels,
          "Transformed channel ordering mismatch");
    for (size_t c = 0; c < native.channel_count(); ++c) {
      const auto v = native.view(c);
      const auto &r = reference.channel[c];
      Check(v.descriptor.extent == Extent2D{r.w, r.h}, "Transform shape mismatch");
      if (c >= native.metadata_channels())
        Check(v.descriptor.hshift == r.hshift && v.descriptor.vshift == r.vshift,
              "Transform shifts mismatch");
      for (size_t y = 0; y < r.h; ++y)
        for (size_t x = 0; x < r.w; ++x)
          Check(v.Row(y)[x] == r.Row(y)[x], "Prescribed transformed sample mismatch");
    }
  }
  if (learn || check_tokens) {
    ChannelShape shape;
    Ok(DescribeImage(transformed.image(), &shape));
    ModularFrameGeometry geometry;
    Ok(ModularFrameGeometry::Create(extent, &geometry));
    ModularStreamPlan layout;
    Ok(BuildModularStreamPlan(geometry, shape.channels(), shape.metadata, &layout));
    if (learn) {
      ModularCodingPolicy single, split, learned, again;
      Ok(LearnModularPolicies(transformed, layout, policy, &single, &split, &learned));
      Ok(LearnModularPolicies(transformed, layout, policy, &single, &split, &again));
      Check(learned == again && learned.transforms == policy.transforms &&
                learned.rct == policy.rct && learned.weighted == policy.weighted,
            "Transformed learning lost policy or determinism");
      policy = learned;
    }
    LearnedTokens(transformed, layout, policy, reference, &memory);
  }
  for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    ModularWorkflowStoragePlan plan;
    Ok(ComputeModularWorkflowStoragePlan(extent, format, mode, policy, &plan));
    std::shared_ptr<const ExecutionDomain> domain;
    Ok(ExecutionDomain::Create(
        {.managed_memory_bytes = plan.working.peak_bytes, .cpu_participant_limit = 1}, &domain));
    std::vector<uint8_t> output;
    Ok(EncodeModularImage(input, {domain, mode, policy}, &output));
    try {
      Check(ref::DecodeLossless(output) == expected, "Pinned full decode mismatch");
    } catch (...) {
      std::cerr << "transform decode " << extent.width << 'x' << extent.height
                << " format " << static_cast<int>(format) << " coder " << static_cast<int>(mode)
                << " learned " << learn << " nodes " << policy.tree.size << '\n';
      for (size_t n = 0; n < policy.tree.size; ++n) {
        const auto &v = policy.tree.nodes[n];
        std::cerr << n << ": prop " << v.property << " split " << v.split << " children "
                  << unsigned(v.left) << '/' << unsigned(v.right) << " pred "
                  << static_cast<int>(v.predictor) << '\n';
      }
      if (!artifacts.empty()) {
        std::ofstream failed(artifacts / "failed.jxl", std::ios::binary);
        failed.write(reinterpret_cast<const char *>(output.data()), output.size());
      }
      throw;
    }
    if (extent.width > 256 || extent.height > 256) {
      ModularWorkflowStoragePlan parallel_plan;
      Ok(ComputeModularWorkflowStoragePlan(extent, format, mode, policy, &parallel_plan, 4));
      std::shared_ptr<const ExecutionDomain> parallel_domain;
      Ok(ExecutionDomain::Create(
          {.managed_memory_bytes = parallel_plan.working.peak_bytes, .cpu_participant_limit = 4},
          &parallel_domain));
      std::vector<uint8_t> parallel;
      Ok(EncodeModularImage(input, {parallel_domain, mode, policy, false, 4}, &parallel));
      Check(output == parallel, "Transformed serial/parallel bytes differ");
    }

    Check(output.size() <= plan.maximum_codestream_bytes &&
              domain->snapshot().peak_backing_bytes <= plan.working.peak_bytes,
          "Transform exceeded storage bound");
    if (!artifacts.empty()) {
      std::ofstream file(artifacts / ("case-" + std::to_string(files) + ".jxl"), std::ios::binary);
      file.write(reinterpret_cast<const char *>(output.data()), output.size());
      Check(bool(file), "Artifact write failed");
    }
    ++files;
  }
  Check(bytes == unchanged, "Transform mutated source");
  ++cases;
}
TransformDescriptor Rct(uint8_t begin, uint8_t type) {
  TransformDescriptor t{TransformKind::kRct, begin, 3};
  t.rct_type = type;
  return t;
}
void Header(const ModularCodingPolicy &p) {
  for (size_t offset = 0; offset < 8; ++offset) {
    BitWriter writer;
    Ok(writer.WriteBits(offset, 0));
    Ok(WriteCodingStreamHeader(p.weighted, p.rct, p.transforms, &writer));
    Check(writer.bits_written() - offset <= MaximumCodingStreamHeaderBits(p.transforms),
          "Ordered header exceeded its bound");
    const auto bytes = writer.padded_bytes();
    jxl::BitReader reader(jxl::Bytes(bytes.data(), bytes.size()));
    struct Closer {
      jxl::BitReader &reader;
      ~Closer() { (void)reader.Close(); }
    } closer{reader};
    reader.SkipBits(offset);
    jxl::GroupHeader header;
    Check(bool(jxl::Bundle::Read(&reader, &header)), "Pinned stream header parse failed");
    Check(reader.AllReadsWithinBounds() && reader.TotalBitsConsumed() == writer.bits_written() &&
              header.use_global_tree && header.transforms.size() == p.transforms.size + (p.rct != 0),
          "Ordered header extent or scope differs");
    const auto &w = header.wp_header;
    const std::array<int32_t, 7> coefficients = {w.p1C, w.p2C, w.p3Ca, w.p3Cb, w.p3Cc,
                                               w.p3Cd, w.p3Ce};
    Check(std::ranges::equal(coefficients, p.weighted.coefficients) &&
              std::ranges::equal(w.w, p.weighted.weights), "Ordered header WP differs");
    for (size_t i = 0; i < header.transforms.size(); ++i) {
      const auto t = p.rct && i == 0 ? Rct(0, p.rct) : p.transforms.entries[i - (p.rct != 0)];
      const auto &r = header.transforms[i];
      if (t.kind == TransformKind::kRct) {
        Check(r.id == jxl::TransformId::kRCT && r.begin_c == t.begin && r.rct_type == t.rct_type,
              "Ordered RCT header differs");
      } else if (t.kind == TransformKind::kPalette) {
        Check(r.id == jxl::TransformId::kPalette && r.begin_c == t.begin && r.num_c == t.count &&
                  r.nb_colors == t.colors && r.nb_deltas == 0 && r.predictor == jxl::Predictor::Zero,
              "Extended palette header differs");
      } else {
        Check(r.id == jxl::TransformId::kSqueeze && r.squeezes.size() == 1,
              "Ordered squeeze header differs");
        const auto &s = r.squeezes[0];
        Check(s.begin_c == t.begin && s.num_c == t.count && s.horizontal == t.horizontal &&
                  s.in_place == t.in_place, "Ordered squeeze parameters differ");
      }
    }
    ++headers;
  }
}
void OrderedCases() {
  for (uint8_t type = 0; type < 42; ++type) {
    for (uint8_t begin : {0, 1, 7, 8, 61}) {
      ModularCodingPolicy p;
      p.transforms.size = 1;
      p.transforms.entries[0] = Rct(begin, type);
      Header(p);
      p.rct = 6;
      p.weighted.coefficients = {31, 0, 16, 8, 1, 7, 9};
      p.weighted.weights = {0, 15, 1, 14};
      Header(p);
    }
    for (auto format : {PackedModularFormat::kRgb8, PackedModularFormat::kRgba16}) {
      ModularCodingPolicy p;
      p.transforms.size = 1;
      p.transforms.entries[0] = Rct(format == PackedModularFormat::kRgb8 ? 0 : 1, type);
      p.tree.nodes[0].predictor = Predictor::kWeighted;
      Case({17, 19}, format, 0, p, false, true);
    }
    ModularCodingPolicy p;
    p.transforms.size = 2;
    p.transforms.entries[0] = {TransformKind::kPalette, 1, 1, 257};
    p.transforms.entries[1] = Rct(1, type);
    Case({257, 3}, PackedModularFormat::kRgba16, 257, p, false, true);
  }
  for (uint16_t colors : {255, 256, 257, 285, 338, 417, 662, 862, 1024}) {
    for (uint8_t count : {1, 2, 3, 4})
      for (uint8_t begin : {0, 7, 8, 60}) {
        ModularCodingPolicy p;
        p.transforms.size = 1;
        p.transforms.entries[0] = {TransformKind::kPalette, begin, count, colors};
        Header(p);
      }
    for (auto format : {PackedModularFormat::kGray16, PackedModularFormat::kRgb8,
                       PackedModularFormat::kRgba8, PackedModularFormat::kRgb16,
                       PackedModularFormat::kRgba16}) {
      ModularInputProfile profile;
      Ok(ResolveModularInput({1, 1}, format, &profile));
      ModularCodingPolicy p;
      p.transforms.size = 1;
      p.transforms.entries[0] = {TransformKind::kPalette, 0,
                                 static_cast<uint8_t>(profile.channel_count), colors};
      p.tree.nodes[0].predictor = Predictor::kWeighted;
      p.weighted.coefficients = {17, 9, 6, 8, 7, 1, 2};
      Case({1025, 3}, format, colors, p, false, true);
    }
  }
  ModularCodingPolicy p;
  p.transforms.size = 3;
  p.transforms.entries[0] = {TransformKind::kPalette, 0, 1, 182};
  p.transforms.entries[1] = {TransformKind::kPalette, 2, 1, 73};
  p.transforms.entries[2] = Rct(2, 33);
  Case({257, 3}, PackedModularFormat::kRgba16, 73, p, false, true);
  p.transforms.entries[0] = {TransformKind::kSqueeze, 0, 4, 1, true, false};
  p.transforms.entries[1] = {TransformKind::kSqueeze, 0, 8, 1, false, false};
  p.transforms.entries[2] = Rct(8, 40);
  Case({16, 16}, PackedModularFormat::kRgba16, 0, p, false, true);
  p.rct = 9;
  p.transforms.size = kMaximumTransforms;
  for (size_t i = 0; i < p.transforms.size; ++i)
    p.transforms.entries[i] = Rct(i % 2, static_cast<uint8_t>(6 + i));
  Header(p);
  Case({257, 3}, PackedModularFormat::kRgba16, 0, p, false, true);
  // Ordered RCT before palette, with learned context and retained legacy RCT.
  p.transforms.size = 2;
  p.transforms.entries[0] = Rct(0, 40);
  p.transforms.entries[1] = {TransformKind::kPalette, 0, 3, 862};
  Case({1025, 3}, PackedModularFormat::kRgb16, 862, p, true);
  std::cout << "Ordered transforms qualified: " << cases << " stages, " << files
            << " complete files, " << headers << " headers\n";
}
} // namespace
int main(int argc, char **argv) try {
  if (argc > 1) {
    artifacts = argv[1];
    std::filesystem::create_directories(artifacts);
  }
  if (argc > 2 && std::string(argv[2]) == "ordered") {
    OrderedCases();
    return 0;
  }
  for (auto format :
       {PackedModularFormat::kGray8, PackedModularFormat::kRgb8, PackedModularFormat::kRgba8,
        PackedModularFormat::kGray16, PackedModularFormat::kRgb16, PackedModularFormat::kRgba16}) {
    ModularInputProfile profile;
    Ok(ResolveModularInput({1, 1}, format, &profile));
    for (auto extent : {Extent2D{1, 1}, Extent2D{1, 17}, Extent2D{17, 1}, Extent2D{17, 19},
                        Extent2D{257, 3}, Extent2D{3, 257}, Extent2D{2049, 1}})
      for (size_t colors : {size_t{1}, size_t{2}, size_t{16}, size_t{256}}) {
        ModularCodingPolicy policy;
        policy.transforms.size = 1;
        policy.transforms.entries[0] = {TransformKind::kPalette, 0,
                                        static_cast<uint8_t>(profile.channel_count),
                                        static_cast<uint16_t>(colors)};
        Case(extent, format, colors, policy);
      }
  }
  std::cout << "Palette qualified: " << cases << " stages, " << files << " complete files\n"
            << std::flush;
  if (argc > 2 && std::string(argv[2]) == "palette")
    return 0;
  for (auto format :
       {PackedModularFormat::kGray8, PackedModularFormat::kRgb8, PackedModularFormat::kRgba8,
        PackedModularFormat::kGray16, PackedModularFormat::kRgb16, PackedModularFormat::kRgba16}) {
    ModularInputProfile profile;
    Ok(ResolveModularInput({1, 1}, format, &profile));
    for (auto extent :
         {Extent2D{3, 3}, Extent2D{17, 19}, Extent2D{257, 3}, Extent2D{3, 257}, Extent2D{513, 513}})
      for (bool horizontal : {false, true})
        for (bool in_place : {false, true}) {
          ModularCodingPolicy policy;
          policy.rct = profile.channel_count >= 3 ? 6 : 0;
          policy.transforms.size = 1;
          policy.transforms.entries[0] = {TransformKind::kSqueeze,
                                          0,
                                          static_cast<uint8_t>(profile.channel_count),
                                          1,
                                          horizontal,
                                          in_place};
          Case(extent, format, 0, policy);
        }
    ModularCodingPolicy combined;
    combined.rct = profile.channel_count >= 3 ? 9 : 0;
    combined.transforms.size = 3;
    combined.transforms.entries[0] = {TransformKind::kPalette, 0,
                                      static_cast<uint8_t>(profile.channel_count), 16};
    combined.transforms.entries[1] = {TransformKind::kSqueeze, 1, 1, 1, true, false};
    combined.transforms.entries[2] = {TransformKind::kSqueeze, 1, 1, 1, false, true};
    Case({513, 257}, format, 16, combined);
    ModularCodingPolicy dc;
    dc.transforms.size = 6;
    for (size_t i = 0; i < 6; ++i)
      dc.transforms.entries[i] = {TransformKind::kSqueeze,
                                  0,
                                  static_cast<uint8_t>(profile.channel_count),
                                  1,
                                  i % 2 == 0,
                                  false};
    Case({2057, 17}, format, 0, dc);
  }
  for (auto format : {PackedModularFormat::kRgba8, PackedModularFormat::kRgba16}) {
    ModularCodingPolicy p;
    p.transforms.size = 2;
    p.transforms.entries[0] = {TransformKind::kPalette, 0, 3, 16};
    p.transforms.entries[1] = {TransformKind::kPalette, 2, 1, 16};
    Case({257, 3}, format, 16, p);
    p.transforms.size = 1;
    p.transforms.entries[0] = {TransformKind::kPalette, 1, 2, 16};
    Case({3, 257}, format, 16, p);
    p.transforms.size = 2;
    p.transforms.entries[0] = {TransformKind::kSqueeze, 1, 2, 1, true, true};
    p.transforms.entries[1] = {TransformKind::kPalette, 1, 2, 256};
    Case({257, 3}, format, 2, p);
    p.transforms.size = 8;
    for (size_t i = 0; i < 8; ++i)
      p.transforms.entries[i] = {TransformKind::kSqueeze, 0, 4, 1, i % 2 == 0, false};
    Case({2057, 17}, format, 0, p);
  }
  for (bool palette : {false, true}) {
    ModularCodingPolicy p;
    p.tree.size = 3;
    p.tree.nodes[0] = {.property = 0, .split = 1, .left = 1, .right = 2};
    p.tree.nodes[1].predictor = Predictor::kWeighted;
    p.tree.nodes[2].predictor = Predictor::kAverage4;
    p.transforms.size = 1;
    p.transforms.entries[0] = {
        palette ? TransformKind::kPalette : TransformKind::kSqueeze, 0, 4, 256, true, false};
    Case({17, 19}, PackedModularFormat::kRgba16, 16, p);
    Case({257, 3}, PackedModularFormat::kRgba16, 16, p);
  }
  {
    ModularCodingPolicy p;
    p.rct = 6;
    p.transforms.size = 5;
    p.transforms.entries[0] = {TransformKind::kSqueeze, 0, 4, 1, true, false};
    p.transforms.entries[1] = {TransformKind::kSqueeze, 0, 8, 1, false, false};
    p.transforms.entries[2] = {TransformKind::kSqueeze, 0, 16, 1, true, false};
    p.transforms.entries[3] = {TransformKind::kSqueeze, 0, 19, 1, false, false};
    p.transforms.entries[4] = {TransformKind::kSqueeze, 0, 13, 1, true, false};
    Case({33, 35}, PackedModularFormat::kRgba16, 0, p);
    std::array<ChannelShape, kMaximumTransforms + 1> shapes;
    ModularInputProfile profile;
    Ok(ResolveModularInput({33, 35}, PackedModularFormat::kRgba16, &profile));
    Ok(PlanTransforms(profile.channels(), 0, p.transforms, &shapes));
    Check(shapes[5].count == 64, "Channel limit fixture is incomplete");
    p.transforms.size = 6;
    p.transforms.entries[5] = {TransformKind::kSqueeze, 0, 1, 1, true, false};
    Check(!PlanTransforms(profile.channels(), 0, p.transforms, &shapes).ok(),
          "Excess channels accepted");
  }
  // Regression: group-local channel 0 must not inherit its full-image
  // index after a global palette dictionary or a globally coded lowpass.
  {
    ModularCodingPolicy policy;
    policy.tree.size = 3;
    policy.tree.nodes[0] = {.property = 0, .split = 0, .left = 1, .right = 2};
    policy.tree.nodes[1].predictor = Predictor::kZero;
    policy.tree.nodes[2].predictor = Predictor::kGradient;
    policy.transforms.size = 1;
    policy.transforms.entries[0] = {TransformKind::kPalette, 0, 1, 16};
    Case({257, 3}, PackedModularFormat::kGray8, 16, policy);
    policy.transforms.size = 2;
    policy.transforms.entries[0] = {TransformKind::kSqueeze, 0, 4, 1, true, false};
    policy.transforms.entries[1] = {TransformKind::kSqueeze, 0, 4, 1, false, false};
    Case({3, 257}, PackedModularFormat::kRgba16, 16, policy);
  }
  // Train on palette dictionaries/indices and shifted Squeeze channels in all
  // formats, with clipped groups, tiny/one-dimensional images and custom WP.
  for (auto format : {PackedModularFormat::kGray8, PackedModularFormat::kRgb8,
                     PackedModularFormat::kRgba8, PackedModularFormat::kGray16,
                     PackedModularFormat::kRgb16, PackedModularFormat::kRgba16}) {
    for (auto extent : {Extent2D{1, 1}, {1, 17}, {17, 1}, {17, 19}, {257, 3},
                        {3, 257}, {513, 257}}) {
      std::array<ModularCodingPolicy, 4> policies;
      size_t count;
      Ok(BuildTransformCandidates(extent, format, 16, &policies, &count));
      for (size_t i = 0; i < count; ++i)
        Case(extent, format, 16, policies[i], true);
    }
    ModularInputProfile profile;
    Ok(ResolveModularInput({1, 1}, format, &profile));
    ModularCodingPolicy combined;
    combined.rct = profile.channel_count >= 3 ? 9 : 0;
    combined.weighted.coefficients = {17, 9, 6, 8, 7, 1, 2};
    combined.weighted.weights = {15, 0, 9, 11};
    combined.transforms.size = 3;
    combined.transforms.entries[0] = {TransformKind::kPalette, 0,
                                      static_cast<uint8_t>(profile.channel_count), 16};
    combined.transforms.entries[1] = {TransformKind::kSqueeze, 1, 1, 1, true, false};
    combined.transforms.entries[2] = {TransformKind::kSqueeze, 1, 1, 1, false, true};
    Case({257, 3}, format, 16, combined, true);
    combined.transforms.size = 6;
    for (size_t i = 0; i < 6; ++i)
      combined.transforms.entries[i] = {TransformKind::kSqueeze, 0,
                                        static_cast<uint8_t>(profile.channel_count), 1,
                                        i % 2 == 0, false};
    Case({2057, 17}, format, 0, combined, true);
  }
  std::cout << "Transforms qualified: " << cases << " stages, " << files << " complete files\n";
} catch (const std::exception &e) {
  std::cerr << "case " << cases << ": " << e.what() << '\n';
  return 1;
}
