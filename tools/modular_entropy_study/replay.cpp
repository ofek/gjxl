// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Research-only identical-token replay. No encoder policy or public defaults.
#include "codestream/ans_internal.h"
#include "codestream/headers_internal.h"
#include "codestream/modular/stream_encoder.h"
#include "codestream/modular/tree_codec.h"
#include "codestream/modular/tokenization.h"
#include "codestream/sections.h"
#include "io/pnm.h"
#include "modular_reference.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_context_map.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/memory_manager_internal.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::codestream_internal;
namespace ref = gjxl::test::modular_reference;
using Clock = std::chrono::steady_clock;
void Require(bool b, const char *message) { if (!b) throw std::runtime_error(message); }
void Ok(const Status &s) { if (!s.ok()) throw std::runtime_error(std::string(s.message())); }
double Seconds(Clock::time_point begin) { return std::chrono::duration<double>(Clock::now() - begin).count(); }

ModularCodingPolicy ReadPolicy(const char *path) {
  std::ifstream file(path);
  file.exceptions(std::ios::badbit | std::ios::failbit);
  ModularCodingPolicy policy;
  std::string version;
  unsigned rct;
  file >> version >> rct >> policy.transforms.size >> policy.tree.size;
  Require(version == "POLICY1" && rct < 42 && policy.transforms.size <= kMaximumTransforms &&
      policy.tree.size && policy.tree.size <= kMaximumTreeNodes, "Invalid replay policy dimensions");
  policy.rct = rct;
  for (auto &coefficient : policy.weighted.coefficients) {
    unsigned value; file >> value; Require(value < 32, "Invalid replay WP coefficient"); coefficient = value;
  }
  for (auto &weight : policy.weighted.weights) {
    unsigned value; file >> value; Require(value < 16, "Invalid replay WP weight"); weight = value;
  }
  for (size_t i = 0; i < policy.transforms.size; ++i) {
    auto &t = policy.transforms.entries[i];
    unsigned kind, begin, count;
    file >> kind >> begin >> count >> t.colors >> t.horizontal >> t.in_place;
    Require(kind <= 1 && begin <= 255 && count <= 255, "Invalid replay transform");
    t.kind = static_cast<TransformKind>(kind); t.begin = begin; t.count = count;
  }
  policy.tree.nodes.resize(policy.tree.size);
  for (size_t i = 0; i < policy.tree.size; ++i) {
    auto &n = policy.tree.nodes[i];
    unsigned predictor;
    file >> n.property >> n.split >> n.left >> n.right >> predictor >> n.offset >> n.multiplier;
    Require(predictor < 14, "Invalid replay predictor");
    n.predictor = static_cast<Predictor>(predictor);
  }
  Ok(ValidateCodingPolicy(policy));
  return policy;
}
// Archive is an explicit little-endian header, then interleaved uint32 context,
// value records. The harness records a SHA-256 of it; no policy is learned here.
void Put64(std::ostream &file, uint64_t n) {
  for (size_t i = 0; i < 8; ++i) file.put(char(n >> (8 * i)));
}
uint64_t Get64(std::istream &file) {
  uint64_t n = 0;
  for (size_t i = 0; i < 8; ++i) n |= uint64_t(static_cast<unsigned char>(file.get())) << (8 * i);
  return n;
}
void Freeze(const std::filesystem::path &path, const PreparedModularTokens &tokens) {
  static_assert(std::endian::native == std::endian::little && sizeof(EntropyToken) == 8);
  Require(!std::filesystem::exists(path), "Refusing to overwrite a frozen token archive");
  std::ofstream file(path, std::ios::binary);
  file.exceptions(std::ios::badbit | std::ios::failbit);
  file.write("GJXLTOK1", 8);
  Put64(file, tokens.context_count); Put64(file, tokens.streams.size());
  for (auto stream : tokens.streams) Put64(file, stream.size());
  for (auto stream : tokens.streams) {
    Require(!stream.split, "Freeze requires the scalar token producer");
    file.write(reinterpret_cast<const char *>(stream.tokens.data()), stream.tokens.size_bytes());
  }
  file.close();
}
PreparedModularTokens Replay(const std::filesystem::path &path, const ModularCodingPolicy &policy,
                             const ModularStreamPlan &layout) {
  TreeLayout tree;
  Ok(ValidateTree(policy.tree, &tree));
  auto result = PrepareModularTokenHeader(policy, tree);
  std::ifstream file(path, std::ios::binary);
  file.exceptions(std::ios::badbit | std::ios::failbit);
  std::array<char, 8> magic;
  file.read(magic.data(), magic.size());
  Require(std::string_view(magic.data(), magic.size()) == "GJXLTOK1" &&
      Get64(file) == result.context_count && Get64(file) == layout.streams.size(),
      "Frozen token archive shape differs from the fixed policy/layout");
  std::vector<size_t> counts(layout.streams.size());
  size_t total = 0;
  for (size_t i = 0; i < counts.size(); ++i) {
    size_t expected = 0;
    const auto &stream = layout.streams[i];
    for (size_t j = 0; j < stream.slice_count; ++j) {
      size_t area = 0;
      Require(layout.slices[stream.slice_begin + j].rect.extent.try_area(&area) &&
          area <= SIZE_MAX - expected, "Replay slice shape overflow");
      expected += area;
    }
    Require(Get64(file) == expected && expected <= SIZE_MAX - total, "Frozen stream count differs");
    counts[i] = expected; total += expected;
  }
  Require(total <= SIZE_MAX / sizeof(EntropyToken), "Frozen token byte count overflow");
  result.tokens.resize(total);
  file.read(reinterpret_cast<char *>(result.tokens.data()), total * sizeof(EntropyToken));
  file.exceptions(std::ios::badbit);
  Require(file.peek() == std::char_traits<char>::eof(), "Trailing bytes in frozen token archive");
  size_t offset = 0;
  for (size_t count : counts) {
    result.streams.push_back(EntropyTokenStreamView::Interleaved(std::span(result.tokens).subspan(offset, count)));
    offset += count;
  }
  for (auto token : result.tokens) Require(token.context < result.context_count, "Frozen token context invalid");
  return result;
}
struct Bits { std::vector<uint8_t> bytes; size_t bits = 0; };
Bits Copy(const BitWriter &writer) {
  return {{writer.padded_bytes().begin(), writer.padded_bytes().end()}, writer.bits_written()};
}
Bits Copy(jxl::BitWriter *writer) {
  const size_t bits = writer->BitsWritten();
  writer->ZeroPadToByte();
  auto bytes = writer->GetSpan();
  return {{bytes.begin(), bytes.end()}, bits};
}
void Append(const Bits &source, BitWriter *writer) {
  for (size_t start = 0; start < source.bits; start += 56) {
    const size_t bits = std::min(size_t{56}, source.bits - start);
    uint64_t word = 0;
    for (size_t b = 0; b < (bits + 7) / 8; ++b) word |= uint64_t(source.bytes[start / 8 + b]) << (8 * b);
    Ok(writer->WriteBits(bits, word));
  }
}
struct Result {
  std::string name;
  Bits header;
  std::vector<Bits> payloads;
  std::vector<uint8_t> map;
  size_t map_bits = 0, clusters = 0, width = 0, capacity = 0;
  double model_seconds = 0, emission_seconds = 0;
};
Result Native(const PreparedModularTokens &tokens, std::string name, int kind, size_t capacity,
              const std::vector<uint8_t> *partition = nullptr) {
  const auto begin = Clock::now();
  EntropyCode code;
  EntropyCodeOptions options{.context_count = uint32_t(tokens.context_count), .maximum_ans_clusters = capacity};
  if (kind == 0 || partition) {
    EntropyCode prefix;
    PreparedEntropyClusters populations;
    if (!partition) {
      Ok(OptimizeEntropyCodeAndPrepareClusters(tokens.streams, options, &prefix, nullptr, &populations));
    } else {
      // Private prepared-partition input used by direct-ANS internals: no prefix
      // tables means an explicit partition. Only the context assignment changes;
      // the downstream native configuration/normalization/width policy is fixed.
      prefix.context_count = tokens.context_count;
      prefix.context_map.assign(partition->begin(), partition->end());
      populations.context_count = tokens.context_count;
      populations.context_map = prefix.context_map;
      const size_t clusters = *std::max_element(partition->begin(), partition->end()) + 1;
      Require(clusters <= kMaximumAnsClusters, "Native replay partition exceeds representation capacity");
      std::vector<Storage<uint32_t>> values(clusters);
      for (auto token : tokens.tokens) values[(*partition)[token.context]].push_back(token.value);
      populations.values.resize(clusters);
      for (size_t c = 0; c < clusters; ++c) Ok(AggregateEntropyValues(std::move(values[c]), &populations.values[c]));
    }
    Ok(OptimizeAnsEntropyCodeWithPreparedClusters(tokens.streams, prefix, populations, &code,
                                                  nullptr, nullptr, AnsSelectionMode::kEstimated));
  } else {
    Ok(OptimizeDirectAnsEntropyCode(tokens.streams, options,
        kind == 1 ? DirectAnsEntropyMode::kBalanced : DirectAnsEntropyMode::kHighDensity, &code));
  }
  Result result;
  result.name = std::move(name); result.capacity = capacity;
  result.map.assign(code.context_map.begin(), code.context_map.end());
  result.clusters = code.ans_histograms.size(); result.width = code.ans_log_alpha_size;
  BitWriter header;
  Ok(WriteGlobalModelInTransaction(code, &header));
  result.header = Copy(header);
  result.model_seconds = Seconds(begin);
  // A separate attribution serialization is outside the model timer.
  BitWriter map;
  if (code.context_count > 1) Ok(WriteContextMap(code, &map));
  result.map_bits = map.bits_written();
  const auto emitted = Clock::now();
  for (auto stream : tokens.streams) {
    BitWriter payload;
    if (stream.size()) Ok(WriteTokenStream(stream, code, &payload));
    result.payloads.push_back(Copy(payload));
  }
  result.emission_seconds = Seconds(emitted);
  return result;
}
Result Reference(const PreparedModularTokens &tokens, JxlMemoryManager *manager, size_t capacity,
                 bool streaming, bool e9) {
  std::vector<std::vector<jxl::Token>> input(tokens.streams.size());
  for (size_t s = 0; s < input.size(); ++s) {
    input[s].reserve(tokens.streams[s].size());
    for (size_t i = 0; i < tokens.streams[s].size(); ++i) {
      const auto token = tokens.streams[s][i];
      input[s].emplace_back(token.context, token.value);
    }
  }
  jxl::CompressParams cparams;
  cparams.modular_mode = true;
  cparams.speed_tier = e9 ? jxl::SpeedTier::kTortoise : jxl::SpeedTier::kSquirrel;
  auto params = jxl::HistogramParams::ForModular(cparams, {}, streaming);
  params.lz77_method = jxl::HistogramParams::LZ77Method::kNone;
  params.max_histograms = capacity;
  params.initialize_global_state = false; // Match the native forced-ANS condition.
  jxl::EntropyEncodingData code{};
  code.use_prefix_code = false;
  jxl::BitWriter header(manager);
  const auto begin = Clock::now();
  Require(jxl::BuildAndEncodeHistograms(manager, params, tokens.context_count, input, &code, &header,
      jxl::LayerType::ModularGlobal, nullptr).ok(), "Reference histogram builder failed");
  Result result;
  result.name = std::string("libjxl-") + (e9 ? "e9" : "e7") + (streaming ? "-streaming-" : "-nonstreaming-") + std::to_string(capacity);
  result.capacity = capacity; result.model_seconds = Seconds(begin);
  Require(!code.lz77.enabled && !code.use_prefix_code && input.size() == tokens.streams.size(),
          "Reference changed the fixed ANS/token condition");
  for (size_t s = 0; s < input.size(); ++s) {
    Require(input[s].size() == tokens.streams[s].size(), "Reference changed token count");
    for (size_t i = 0; i < input[s].size(); ++i) {
      const auto expected = tokens.streams[s][i], got = EntropyToken{input[s][i].context, input[s][i].value};
      Require(expected == got && !input[s][i].is_lz77_length, "Reference changed ordered tokens/contexts");
    }
  }
  result.map = code.context_map;
  result.clusters = code.encoding_info.size(); result.width = code.log_alpha_size;
  jxl::BitWriter map(manager);
  // The singleton-map branch writes three bits directly; other branches open
  // nested allotments. Match the caller reservation used by the model builder.
  if (tokens.context_count > 1) Require(bool(map.WithMaxBits(3, jxl::LayerType::ModularGlobal,
      nullptr, [&] { return jxl::EncodeContextMap(code.context_map, result.clusters,
          &map, jxl::LayerType::ModularGlobal, nullptr); })), "Reference context map writer failed");
  result.map_bits = map.BitsWritten();
  result.header = Copy(&header);
  const auto emitted = Clock::now();
  for (const auto &stream : input) {
    jxl::BitWriter payload(manager);
    if (!stream.empty()) Require(bool(jxl::WriteTokens(stream, code, 0, &payload,
        jxl::LayerType::ModularGlobal, nullptr)), "Reference token writer failed");
    result.payloads.push_back(Copy(&payload));
  }
  result.emission_seconds = Seconds(emitted);
  return result;
}

void Report(const Result &result, const codec_internal::ImageMetadata &metadata,
            const ModularStreamPlan &layout, const PreparedModularTokens &tokens,
            const ref::IntegerImage &original, const std::filesystem::path &out) {
  Storage<BitWriter> sections(layout.geometry.section_count());
  GlobalTreeStoragePlan tree;
  Ok(ComputeGlobalTreeStoragePlan(tokens.tree_token_count, &tree));
  auto &global = sections[0];
  Ok(global.WithMaxBits(1 + tree.maximum_bits, [&]() -> Status {
    if (auto s = global.WriteBits(1, 1); !s.ok()) return s;
    return WriteGlobalTreeInTransaction(std::span(tokens.tree_tokens).first(tokens.tree_token_count), &global);
  }));
  const size_t tree_bits = global.bits_written() - 1;
  Append(result.header, &global);
  Ok(WriteCodingStreamHeader(tokens.policy.weighted, tokens.policy.rct, tokens.policy.transforms, &global));
  Append(result.payloads[0], &global);
  for (size_t s = 1; s < tokens.streams.size(); ++s) {
    if (!tokens.streams[s].size()) continue;
    auto &section = sections[layout.streams[s].section];
    Ok(WriteCodingStreamHeader(tokens.policy.weighted, 0, &section));
    Append(result.payloads[s], &section);
  }
  BitWriter file;
  Ok(WriteImageHeader(metadata, &file)); Ok(WriteFrameHeader(metadata, {}, &file));
  Ok(WriteTocAndSections(sections, &file));
  Require(ref::DecodeLossless(file.padded_bytes()) == original, "Replayed model/payload failed independent decode");
  const auto path = out / (result.name + ".jxl");
  Require(!std::filesystem::exists(path), "Refusing to overwrite a rate-study output");
  std::ofstream artifact(path, std::ios::binary);
  artifact.exceptions(std::ios::badbit | std::ios::failbit);
  artifact.write(reinterpret_cast<const char *>(file.padded_bytes().data()), file.padded_size());
  artifact.close();
  size_t payload_bits = 0;
  for (const auto &payload : result.payloads) payload_bits += payload.bits;
  Require(result.header.bits >= result.map_bits + 1, "Invalid model header decomposition");
  std::cout << std::setprecision(17) << "{\"cell\":\"" << result.name << "\",\"capacity\":" << result.capacity
      << ",\"clusters\":" << result.clusters << ",\"contexts\":" << tokens.context_count
      << ",\"tokens\":" << tokens.tokens.size() << ",\"ans_width\":" << result.width
      << ",\"payload_bits\":" << payload_bits << ",\"context_map_bits\":" << result.map_bits
      << ",\"histogram_config_bits\":" << result.header.bits - result.map_bits - 1
      << ",\"lz77_flag_bits\":1,\"model_bits\":" << result.header.bits << ",\"tree_bits\":" << tree_bits
      << ",\"bytes\":" << file.padded_size() << ",\"decoded_exact\":true,\"model_seconds\":" << result.model_seconds
      << ",\"emission_seconds\":" << result.emission_seconds << ",\"context_map\":[";
  for (size_t i = 0; i < result.map.size(); ++i) std::cout << (i ? "," : "") << unsigned(result.map[i]);
  std::cout << "]}" << std::endl;
}
} // namespace

int main(int argc, char **argv) try {
  if (argc != 5 && argc != 6) throw std::runtime_error("entropy_replay freeze|replay INPUT POLICY ARCHIVE [OUT]");
  const bool freeze = std::string_view(argv[1]) == "freeze";
  Require((freeze && argc == 5) || (std::string_view(argv[1]) == "replay" && argc == 6), "Invalid replay command");
  io::IntegerImage image;
  Ok(io::ReadPnm(argv[2], &image));
  PackedModularImageView view{image.bytes, image.extent, image.row_stride,
      static_cast<PackedModularFormat>(image.format), SampleByteOrder::kBigEndian};
  const auto policy = ReadPolicy(argv[3]);
  ModularEncoderFrame frame;
  Ok(ModularEncoderFrame::Prepare(view, policy, &frame));
  auto metadata = frame.metadata();
  if (policy != ModularCodingPolicy{}) metadata.modular_16_bit_buffer_sufficient = false;
  ChannelShape shape;
  Ok(DescribeImage(frame.image(), &shape));
  ModularFrameGeometry geometry;
  Ok(ModularFrameGeometry::Create(image.extent, &geometry));
  ModularStreamPlan layout;
  Ok(BuildModularStreamPlan(geometry, shape.channels(), shape.metadata, &layout));
  if (freeze) {
    PreparedModularTokens tokens;
    Ok(TokenizeModular(frame, layout, policy, &tokens));
    Freeze(argv[4], tokens);
    std::cout << "{\"tokens\":" << tokens.tokens.size() << ",\"contexts\":" << tokens.context_count
              << ",\"streams\":" << tokens.streams.size() << "}\n";
    return 0;
  }
  const auto tokens = Replay(argv[4], policy, layout);
  frame = {};
  ModularInputProfile source;
  Ok(ResolveModularInput(image.extent, view.format, &source));
  ref::IntegerImage original{image.extent.width, image.extent.height, uint32_t(source.channel_count), metadata.bits, {}};
  for (size_t y = 0; y < image.extent.height; ++y)
    for (size_t x = 0; x < image.extent.width * source.channel_count; ++x) {
      const size_t i = y * image.row_stride + x * (metadata.bits / 8);
      original.samples.push_back(metadata.bits == 8 ? image.bytes[i] : (uint16_t(image.bytes[i]) << 8) | image.bytes[i + 1]);
    }
  std::filesystem::create_directories(argv[5]);
  const auto report = [&](const Result &r) { Report(r, metadata, layout, tokens, original, argv[5]); };
  report(Native(tokens, "native-prefix-estimated-32", 0, 32));
  for (size_t capacity : {size_t{32}, size_t{64}}) {
    report(Native(tokens, "native-direct-balanced-" + std::to_string(capacity), 1, capacity));
    report(Native(tokens, "native-direct-density-" + std::to_string(capacity), 2, capacity));
  }
  JxlMemoryManager manager;
  Require(bool(jxl::MemoryManagerInit(&manager, nullptr)), "Reference memory manager failed");
  for (size_t capacity : {size_t{16}, size_t{32}, size_t{64}, size_t{128}}) {
    auto reference = Reference(tokens, &manager, capacity, true, false);
    report(reference);
    if (capacity <= kMaximumAnsClusters) {
      report(Native(tokens, "native-on-libjxl-partition-" + std::to_string(capacity), 0, capacity, &reference.map));
    }
  }
  report(Reference(tokens, &manager, 128, true, true));
  report(Reference(tokens, &manager, 128, false, true));
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
