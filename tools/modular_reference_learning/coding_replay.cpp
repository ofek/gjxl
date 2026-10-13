// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "coding_replay.h"
#include "codestream/modular/prepared_streams.h"
#include "entropy_replay.h"
#include "io/pnm.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/encoding/enc_ma.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "modular_reference.h"
#include <bit>
#include <fstream>
#include <iomanip>
#include <map>
#include <ostream>
#include <stdexcept>

namespace reference_learning {
namespace native = gjxl::modular_internal;
namespace ref = gjxl::test::modular_reference;
namespace {
void Require(bool good, const char *message) {
  if (!good)
    throw std::runtime_error(message);
}
void Ok(const gjxl::Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
class Reader {
public:
  explicit Reader(const std::filesystem::path &path) : in_(path) {
    Require(std::filesystem::file_size(path) <= (16u << 20), "Coding description too large");
    Require(in_.good(), "Cannot open coding description");
  }
  int64_t Integer(int64_t lo, int64_t hi) {
    int64_t value;
    Require(bool(in_ >> value) && value >= lo && value <= hi, "Invalid coding integer");
    return value;
  }
  void Tag(const char *expected) {
    std::string value;
    Require(bool(in_ >> value) && value == expected, "Invalid coding field tag");
  }
  std::filesystem::path Path() {
    std::string value;
    Require(bool(in_ >> std::quoted(value)) && value.size() <= 4096, "Invalid pixel path");
    return value;
  }
  void Finish() {
    Tag("end");
    in_ >> std::ws;
    Require(in_.peek() == std::char_traits<char>::eof(), "Trailing coding input");
  }

private:
  std::ifstream in_;
};
bool ReadTree(Reader &in, native::DecisionTree *tree) {
  in.Tag("tree");
  const size_t size = in.Integer(0, native::kMaximumTreeNodes);
  if (!size)
    return false;
  tree->size = size;
  tree->nodes.resize(size);
  for (size_t i = 0; i < size; ++i) {
    auto &n = tree->nodes[i];
    n.property = in.Integer(-1, 15);
    n.split = in.Integer(INT32_MIN, INT32_MAX);
    n.left = in.Integer(0, size - 1);
    n.right = in.Integer(0, size - 1);
    n.predictor = static_cast<native::Predictor>(in.Integer(0, 13));
    n.offset = in.Integer(INT32_MIN, INT32_MAX);
    n.multiplier = in.Integer(1, INT32_MAX);
  }
  native::TreeLayout layout;
  Ok(native::ValidateTree(*tree, &layout));
  return true;
}
native::TransformSequence ReadTransforms(Reader &in) {
  native::TransformSequence sequence;
  in.Tag("transforms");
  sequence.size = in.Integer(0, native::kMaximumTransforms);
  for (size_t i = 0; i < sequence.size; ++i) {
    auto &t = sequence.entries[i];
    in.Tag("transform");
    t.kind = static_cast<native::TransformKind>(in.Integer(0, 2));
    t.begin = in.Integer(0, 255);
    t.count = in.Integer(1, 255);
    t.colors = in.Integer(1, native::kMaximumPaletteColors);
    t.horizontal = in.Integer(0, 1);
    t.in_place = in.Integer(0, 1);
    t.rct_type = in.Integer(0, 41);
  }
  Ok(native::ValidateTransforms(sequence));
  return sequence;
}
struct Plane {
  gjxl::Extent2D extent;
  int hshift = 0, vshift = 0;
  std::vector<int32_t> pixels;
};
struct Stream {
  size_t id = 0, width = 0, height = 0, metadata = 0;
  bool local = false;
  jxl::ModularOptions options;
  native::DecisionTree tree;
  native::WeightedPredictorParameters weighted;
  native::TransformSequence transforms;
  std::vector<Plane> planes;
  std::vector<native::ModularChannelView> views;
};
ref::IntegerImage Unpack(const gjxl::io::IntegerImage &input,
                         const native::ModularInputProfile &profile) {
  ref::IntegerImage result{input.extent.width,
                           input.extent.height,
                           static_cast<uint32_t>(profile.channel_count),
                           profile.metadata.bits,
                           {}};
  result.samples.reserve(input.extent.width * input.extent.height * profile.channel_count);
  for (size_t y = 0; y < input.extent.height; ++y)
    for (size_t x = 0; x < input.extent.width * profile.channel_count; ++x) {
      const size_t at = y * input.row_stride + x * profile.bytes_per_sample;
      result.samples.push_back(profile.bytes_per_sample == 1
                                   ? input.bytes[at]
                                   : (uint16_t{input.bytes[at]} << 8) | input.bytes[at + 1]);
    }
  return result;
}
void CheckTokens(const Stream &stream, const native::DecisionTree &tree,
                 std::span<const gjxl::EntropyToken> actual_tree,
                 gjxl::EntropyTokenStreamView actual, uint32_t bits) {
  jxl::Tree raw, decoded;
  for (size_t i = 0; i < tree.size; ++i) {
    const auto &n = tree.nodes[i];
    raw.emplace_back(n.property, n.split, n.left, n.right, static_cast<jxl::Predictor>(n.predictor),
                     n.offset, n.multiplier);
  }
  std::vector<jxl::Token> tree_tokens;
  Require(bool(jxl::TokenizeTree(raw, &tree_tokens, &decoded)),
          "Reference tree tokenization failed");
  Require(tree_tokens.size() == actual_tree.size(), "Captured tree token count differs");
  for (size_t i = 0; i < tree_tokens.size(); ++i)
    Require(tree_tokens[i].context == actual_tree[i].context &&
                tree_tokens[i].value == actual_tree[i].value,
            "Captured tree token differs");
  JxlMemoryManager manager;
  Require(bool(jxl::MemoryManagerInit(&manager, nullptr)), "Reference memory setup failed");
  jxl::Image image(&manager);
  image.w = stream.width;
  image.h = stream.height;
  image.bitdepth = bits;
  image.nb_meta_channels = stream.metadata;
  image.error = false;
  for (const auto &plane : stream.planes) {
    auto created = jxl::Channel::Create(&manager, plane.extent.width, plane.extent.height,
                                        plane.hshift, plane.vshift);
    Require(created.ok(), "Reference replay plane allocation failed");
    auto channel = std::move(created).value_();
    for (size_t y = 0; y < plane.extent.height; ++y)
      std::copy_n(plane.pixels.data() + y * plane.extent.width, plane.extent.width, channel.Row(y));
    image.channel.push_back(std::move(channel));
  }
  jxl::GroupHeader header;
  std::vector<jxl::Token> reference;
  size_t width = 0;
  Require(bool(jxl::ModularCompress(image, stream.options, stream.id, decoded, header, reference,
                                    &width)),
          "Reference captured stream tokenization failed");
  const auto &w = header.wp_header;
  const std::array<int, 7> coefficients = {w.p1C, w.p2C, w.p3Ca, w.p3Cb, w.p3Cc, w.p3Cd, w.p3Ce};
  Require(std::ranges::equal(coefficients, stream.weighted.coefficients) &&
              std::ranges::equal(w.w, stream.weighted.weights),
          "Captured WP header differs");
  Require(reference.size() == actual.size(), "Captured stream token count differs");
  for (size_t i = 0; i < reference.size(); ++i)
    if (reference[i].context != actual[i].context || reference[i].value != actual[i].value)
      throw std::runtime_error("Captured sample token differs at stream " +
                               std::to_string(stream.id) + " token " + std::to_string(i));
}
} // namespace

namespace {
void ReplayInput(const std::filesystem::path &source, const std::filesystem::path &description,
                 const std::filesystem::path &output, const std::string &variants,
                 std::ostream &out, bool measure = false) {
  static_assert(std::endian::native == std::endian::little);
  gjxl::io::IntegerImage original;
  Ok(gjxl::io::ReadPnm(source, &original));
  const auto format = static_cast<native::PackedModularFormat>(original.format);
  native::ModularInputProfile profile;
  Ok(native::ResolveModularInput(original.extent, format, &profile));
  const auto expected = Unpack(original, profile);
  Reader in(description);
  in.Tag(variants.empty() ? "gjxl-coding-input-v1" : "gjxl-coding-input-v2");
  native::DecisionTree global_tree;
  const bool has_global = ReadTree(in, &global_tree);
  in.Tag("streams");
  std::vector<Stream> captured(in.Integer(1, 65535));
  std::map<size_t, size_t> by_id;
  size_t pixel_bytes = 0, local_trees = 0, transforms = 0;
  for (size_t i = 0; i < captured.size(); ++i) {
    auto &stream = captured[i];
    in.Tag("stream");
    stream.id = in.Integer(0, 65535);
    Require(by_id.emplace(stream.id, i).second, "Duplicate coding stream ID");
    stream.width = in.Integer(1, 1u << 20);
    stream.height = in.Integer(1, 1u << 20);
    Require(in.Integer(8, 16) == profile.metadata.bits, "Captured/source bit depths differ");
    stream.metadata = in.Integer(0, native::kMaximumTransformedChannels);
    stream.planes.resize(in.Integer(stream.metadata, native::kMaximumTransformedChannels));
    in.Tag("options");
    auto &o = stream.options;
    o.max_chan_size = in.Integer(1, UINT32_MAX);
    o.group_dim = in.Integer(256, 256);
    o.predictor = static_cast<jxl::Predictor>(in.Integer(0, 15));
    o.wp_mode = in.Integer(0, 4);
    o.skip_encoder_fast_path = in.Integer(0, 1);
    o.max_properties = 0;
    o.zero_tokens = false;
    if (!variants.empty()) {
      in.Tag("entropy");
      auto &p = o.histogram_params;
      p.clustering = static_cast<jxl::HistogramParams::ClusteringType>(in.Integer(0, 2));
      p.uint_method = static_cast<jxl::HistogramParams::HybridUintMethod>(in.Integer(0, 4));
      p.lz77_method = static_cast<jxl::HistogramParams::LZ77Method>(in.Integer(0, 15));
      p.ans_histogram_strategy =
          static_cast<jxl::HistogramParams::ANSHistogramStrategy>(in.Integer(0, 2));
      const auto maximum = in.Integer(-1, 256);
      Require(maximum == -1 || maximum > 0, "Invalid histogram limit");
      p.max_histograms = maximum == -1 ? SIZE_MAX : maximum;
      p.force_huffman = in.Integer(0, 1);
      p.initialize_global_state = in.Integer(1, 1);
      p.streaming_mode = in.Integer(0, 1);
      p.add_missing_symbols = in.Integer(0, 0);
      p.add_fixed_histograms = in.Integer(0, 0);
      Require(in.Integer(0, 0) == 0, "Unexpected initial image widths");
    }
    in.Tag("wp");
    for (auto &c : stream.weighted.coefficients)
      c = in.Integer(0, 31);
    for (auto &w : stream.weighted.weights)
      w = in.Integer(0, 15);
    stream.transforms = ReadTransforms(in);
    transforms += stream.transforms.size;
    stream.local = ReadTree(in, &stream.tree);
    Require(stream.local || has_global, "Stream has no tree scope");
    local_trees += stream.local && !stream.planes.empty();
    in.Tag("pixels");
    const auto pixels = in.Path();
    const size_t bytes = in.Integer(0, 512u << 20);
    Require(std::filesystem::file_size(pixels) == bytes, "Captured pixel file size differs");
    std::ifstream raw(pixels, std::ios::binary);
    raw.exceptions(std::ios::badbit | std::ios::failbit);
    size_t next = 0;
    for (size_t c = 0; c < stream.planes.size(); ++c) {
      auto &plane = stream.planes[c];
      in.Tag("channel");
      plane.extent = {static_cast<size_t>(in.Integer(1, 1u << 20)),
                      static_cast<size_t>(in.Integer(1, 1u << 20))};
      plane.hshift = in.Integer(-1, 31);
      plane.vshift = in.Integer(-1, 31);
      const size_t offset = in.Integer(0, bytes), size = in.Integer(0, bytes - offset);
      Require(size == uint64_t{plane.extent.width} * plane.extent.height * 4 &&
                  (!c || offset == next) && size <= (512u << 20) - pixel_bytes,
              "Invalid captured plane range");
      pixel_bytes += size;
      next = offset + size;
      plane.pixels.resize(size / 4);
      raw.seekg(offset);
      raw.read(reinterpret_cast<char *>(plane.pixels.data()), size);
    }
  }
  in.Finish();
  Require(by_id.contains(0), "Missing global coding header");
  native::ModularFrameGeometry geometry;
  Ok(native::ModularFrameGeometry::Create(original.extent, &geometry));
  const auto &global = captured[by_id.at(0)];
  std::array<native::ChannelShape, native::kMaximumTransforms + 1> shapes;
  Ok(native::PlanTransforms(profile.channels(), 0, global.transforms, &shapes));
  const auto full = shapes[global.transforms.size];
  native::ModularStreamPlan plan;
  Ok(native::BuildModularStreamPlan(geometry, full.channels(), full.metadata, &plan));
  std::vector<native::PreparedModularStreamView> streams(plan.streams.size());
  std::vector<jxl::HistogramParams> parameters(streams.size());
  size_t mapped = 0, coded_streams = 0;
  for (size_t i = 0; i < plan.streams.size(); ++i) {
    const auto &slot = plan.streams[i];
    const auto found = by_id.find(slot.id);
    if (found == by_id.end()) {
      Require(i && !slot.slice_count, "Missing captured coding stream");
      continue;
    }
    ++mapped;
    auto &stream = captured[found->second];
    native::ChannelShape shape;
    for (const auto &slice : std::span(plan.slices).subspan(slot.slice_begin, slot.slice_count)) {
      auto d = full.descriptors[slice.channel];
      d.extent = slice.rect.extent;
      shape.descriptors[shape.count++] = d;
      shape.metadata += d.role == native::ChannelRole::kMetadata;
    }
    if (i) {
      Ok(native::PlanTransforms(shape.channels(), shape.metadata, stream.transforms, &shapes));
      shape = shapes[stream.transforms.size];
    }
    Require(shape.count == stream.planes.size() && shape.metadata == stream.metadata,
            "Captured coding shape differs from transform plan");
    for (size_t c = 0; c < shape.count; ++c) {
      const auto &plane = stream.planes[c];
      const auto &d = shape.descriptors[c];
      const bool metadata = d.role == native::ChannelRole::kMetadata;
      Require(plane.extent == d.extent && plane.hshift == (metadata ? -1 : d.hshift) &&
                  plane.vshift == (metadata ? -1 : d.vshift),
              "Captured channel shape differs");
      stream.views.push_back({plane.pixels, d, d.extent.width});
    }
    streams[i] = {stream.views, stream.weighted, 0, stream.transforms,
                  stream.local ? &stream.tree : nullptr};
    parameters[i] = stream.options.histogram_params;
    coded_streams += !stream.planes.empty();
  }
  Require(mapped == captured.size(), "Unexpected captured coding stream ID");
  const native::PreparedModularStreamsView input{original.extent, format,
                                                 has_global ? &global_tree : nullptr, streams};
  size_t token_count = 0;
  {
    native::PreparedModularStreamTokens tokens;
    Ok(native::TokenizePreparedModularStreams(input, &tokens));
    token_count = tokens.tokens.size();
    Require(token_count == pixel_bytes / 4, "Captured token population differs");
    for (size_t i = 0; i < streams.size(); ++i) {
      if (streams[i].channels.empty())
        continue; // Decoder stops after the header for an empty coded prefix.
      const auto &stream = captured[by_id.at(plan.streams[i].id)];
      CheckTokens(stream, stream.local ? stream.tree : global_tree,
                  stream.local ? tokens.local_tree_tokens[i] : tokens.global_tree_tokens,
                  tokens.streams[i], profile.metadata.bits);
    }
    if (!variants.empty()) {
      ReplayEntropyAblations(input, plan, tokens, parameters, expected, output, variants, out, measure);
      return;
    }
  }
  Require(std::filesystem::create_directory(output), "Coding output already exists");
  out << "{\"schema\":1,\"captured_streams\":" << captured.size()
      << ",\"canonical_slots\":" << streams.size() << ",\"coded_streams\":" << coded_streams
      << ",\"local_trees\":" << local_trees << ",\"global_trees\":" << has_global
      << ",\"transforms\":" << transforms << ",\"tokens\":" << token_count
      << ",\"tokens_exact\":true,\"trees_exact\":true,\"decoded_exact\":true"
         ",\"timing_qualified\":false,\"encodes\":[";
  using namespace gjxl::resource_budget_internal;
  for (auto mode : {gjxl::EntropyCodingMode::kPrefix, gjxl::EntropyCodingMode::kAns}) {
    native::ModularWorkflowStoragePlan bound;
    Ok(native::ComputePreparedModularStreamsStoragePlan(input, mode, &bound));
    ResourceBudget budget(bound.working.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.working.peak_bytes, &reservation));
    gjxl::codestream_internal::CodestreamBuffer encoded;
    {
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      Ok(native::EncodePreparedModularStreams(input, mode, &encoded));
    }
    const auto peak = budget.snapshot().peak_backing_bytes;
    Require(peak <= bound.working.peak_bytes && encoded.size() <= bound.maximum_codestream_bytes,
            "Captured encode exceeded managed bound");
    Require(ref::DecodeLossless(encoded.view()) == expected, "Captured encode failed exact decode");
    const bool ans = mode == gjxl::EntropyCodingMode::kAns;
    const char *name = ans ? "ans" : "prefix";
    std::ofstream file(output / (std::string(name) + ".jxl"), std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(reinterpret_cast<const char *>(encoded.data()), encoded.size());
    file.close();
    out << (ans ? "," : "") << "{\"mode\":\"" << name << "\",\"bytes\":" << encoded.size()
        << ",\"storage_peak_bytes\":" << peak
        << ",\"storage_bound_bytes\":" << bound.working.peak_bytes
        << ",\"output_bound_bytes\":" << bound.maximum_codestream_bytes << "}";
    encoded.Reset();
    Require(budget.snapshot().total.backing_count == 0 &&
                budget.snapshot().total.pending_count == 0,
            "Captured encode leaked managed storage");
  }
  out << "]}\n";
}
} // namespace
void ReplayCodingInput(const std::filesystem::path &source,
                       const std::filesystem::path &description,
                       const std::filesystem::path &output, std::ostream &out) {
  ReplayInput(source, description, output, {}, out);
}
void ReplayEntropyInput(const std::filesystem::path &source,
                        const std::filesystem::path &description,
                        const std::filesystem::path &output, const std::string &variants,
                        std::ostream &out, bool measure) {
  Require(!variants.empty(), "Empty entropy variants");
  ReplayInput(source, description, output, variants, out, measure);
}
} // namespace reference_learning
