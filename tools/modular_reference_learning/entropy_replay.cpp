// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "entropy_replay.h"
#include "codestream/ans_internal.h"
#include "codestream/profile_internal.h"
#include "codestream/headers_internal.h"
#include "codestream/modular/stream_encoder.h"
#include "codestream/modular/tree_codec.h"
#include "codestream/sections.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_cluster.h"
#include "lib/jxl/enc_context_map.h"
#include "lib/jxl/memory_manager_internal.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

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
using HP = jxl::HistogramParams;
struct Variant {
  std::string name;
  bool native_tree = false, native_data = false;
  int lz77 = -1;
  bool data_lz77_only = false;
  size_t clusters = SIZE_MAX;
  bool hybrid420 = false, approximate = false, nonstreaming = false;
  bool reference_partition = false;
  int direct_ans_mode = -1;
  bool estimated = false;
};
std::vector<Variant> Variants(const std::string &names) {
  const std::vector<Variant> supported = {
      {"reference"},
      {"native_tree", true},
      {"native_data", false, true},
      {"native", true, true},
      {"no_lz77", false, false, 0},
      {"rle", false, false, 1},
      {"data_no_lz77", false, false, 0, true},
      {"clusters32", false, false, -1, false, 32},
      {"clusters64", false, false, -1, false, 64},
      {"no_lz77_clusters32", false, false, 0, false, 32},
      {"hybrid420", false, false, -1, false, SIZE_MAX, true},
      {"ans_approx", false, false, -1, false, SIZE_MAX, false, true},
      {"nonstreaming", false, false, -1, false, SIZE_MAX, false, false, true},
      {"reference_map32", false, true, 0, false, 32, false, false, false, true},
      {"reference_map32_native_tree", true, true, 0, false, 32, false, false, false, true},
      {.name = "native_direct_balanced", .native_tree = true, .native_data = true,
       .clusters = 32, .direct_ans_mode = 0},
      {.name = "native_direct_high_density", .native_tree = true, .native_data = true,
       .clusters = 32, .direct_ans_mode = 1},
      {.name = "native_direct_rate", .native_tree = true, .native_data = true,
       .clusters = 32, .direct_ans_mode = 2},
      {.name = "native_direct_rate64", .native_tree = true, .native_data = true,
       .clusters = 64, .direct_ans_mode = 2},
      {.name = "native_direct_estimated", .native_tree = true, .native_data = true,
       .clusters = 32, .direct_ans_mode = 2, .estimated = true},
      {.name = "native_direct_estimated64", .native_tree = true, .native_data = true,
       .clusters = 64, .direct_ans_mode = 2, .estimated = true}};
  std::vector<Variant> result;
  std::set<std::string> seen;
  std::istringstream in(names);
  std::string name;
  while (std::getline(in, name, ',')) {
    const auto found = std::find_if(supported.begin(), supported.end(),
                                    [&](const auto &v) { return v.name == name; });
    Require(found != supported.end() && seen.insert(name).second, "Invalid entropy variant");
    result.push_back(*found);
  }
  Require(!result.empty() && !names.ends_with(','), "Empty entropy variant");
  return result;
}
HP Parameters(HP params, const Variant &v, bool tree) {
  if (v.lz77 >= 0 && (!v.data_lz77_only || !tree))
    params.lz77_method = static_cast<HP::LZ77Method>(v.lz77);
  if (!tree) {
    params.max_histograms = std::min(params.max_histograms, v.clusters);
    if (v.hybrid420)
      params.uint_method = HP::HybridUintMethod::kNone;
  }
  if (v.approximate)
    params.ans_histogram_strategy = HP::ANSHistogramStrategy::kApproximate;
  if (v.nonstreaming)
    params.streaming_mode = false;
  return params;
}
// Copy precisely the valid reference bits; no padding is inserted between a
// model/tree and its payload. The source writer is private to this operation.
void Append(jxl::BitWriter &source, gjxl::BitWriter *destination) {
  const size_t bits = source.BitsWritten();
  Require(bool(source.WithMaxBits(7, jxl::LayerType::Header, nullptr,
                                  [&]() -> jxl::Status {
                                    source.ZeroPadToByte();
                                    return true;
                                  })),
          "Reference bit padding failed");
  const auto bytes = source.GetSpan();
  Ok(destination->WithMaxBits(bits, [&]() -> gjxl::Status {
    for (size_t offset = 0; offset < bits; offset += 56) {
      const size_t count = std::min(size_t{56}, bits - offset);
      uint64_t value = 0;
      for (size_t b = 0; b < (count + 7) / 8; ++b)
        value |= uint64_t{bytes[offset / 8 + b]} << (8 * b);
      if (count < 56)
        value &= (uint64_t{1} << count) - 1;
      if (auto s = destination->WriteBits(count, value); !s.ok())
        return s;
    }
    return gjxl::Status::Ok();
  }));
}
struct ScopeStats {
  bool tree = false, native = false, prefix = false, lz77 = false;
  int64_t id = -1;
  size_t original_tokens = 0, coded_tokens = 0, lz77_lengths = 0;
  size_t contexts = 0, clusters = 0, model_bits = 0, token_bits = 0, alphabet_log = 0;
  std::vector<std::array<size_t, 3>> configs;
  bool reference_partition_exact = false;
  std::vector<uint8_t> context_map;
  bool profiled = false;
  gjxl::codestream_internal::EntropyWorkProfile work;
  double model_cpu_seconds = 0, model_wall_seconds = 0;
  void Write(std::ostream &out) const {
    out << "{\"kind\":\"" << (tree ? "tree" : "data") << "\",\"id\":" << id
        << ",\"implementation\":\"" << (native ? "native" : "reference")
        << "\",\"prefix\":" << (prefix ? "true" : "false")
        << ",\"lz77\":" << (lz77 ? "true" : "false") << ",\"original_tokens\":" << original_tokens
        << ",\"coded_tokens\":" << coded_tokens << ",\"lz77_lengths\":" << lz77_lengths
        << ",\"contexts\":" << contexts << ",\"clusters\":" << clusters
        << ",\"model_bits\":" << model_bits << ",\"token_bits\":" << token_bits
        << ",\"alphabet_log\":" << alphabet_log << ",\"uint_configs\":[";
    for (size_t i = 0; i < configs.size(); ++i) {
      const auto &c = configs[i];
      out << (i ? "," : "") << '[' << c[0] << ',' << c[1] << ',' << c[2] << ']';
    }
    out << "],\"reference_partition_exact\":" << (reference_partition_exact ? "true" : "false")
        << ",\"context_map\":[";
    for (size_t i = 0; i < context_map.size(); ++i)
      out << (i ? "," : "") << unsigned(context_map[i]);
    out << "],\"model_cpu_seconds\":" << model_cpu_seconds
        << ",\"model_wall_seconds\":" << model_wall_seconds << ",\"ans_work\":";
    if (profiled) {
      out << "{\"width_candidates\":" << work.ans_alphabet_width_candidate_count
          << ",\"token_cost_candidates\":" << work.ans_token_cost_candidate_count
          << ",\"uint_config_candidates\":" << work.ans_uint_config_candidate_count
          << ",\"histogram_candidates\":" << work.ans_histogram_candidate_count << '}';
    } else out << "null";
    out << '}';
  }
};
// Research transplant: cluster precisely the same default-420 populations as
// the pinned no-LZ77 path. Build only the valid prefix carrier required by the
// unchanged native ANS optimizer, without native prefix clustering. The native
// optimizer reads this carrier's map/count, not its integer configurations or
// prefix lengths; its configuration portfolio is independent of the carrier.
gjxl::EntropyCode ReferencePartition(std::span<const gjxl::EntropyTokenStreamView> streams,
                                     size_t contexts, HP params) {
  Require(params.uint_method != HP::HybridUintMethod::k000 &&
              params.uint_method != HP::HybridUintMethod::kContextMap,
          "Reference partition requires the captured 420 clustering proxy");
  std::vector<jxl::Histogram> builder(contexts), clustered;
  const auto config = params.UintConfig();
  for (auto stream : streams)
    for (size_t i = 0; i < stream.size(); ++i) {
      const auto token = stream[i];
      uint32_t symbol, nbits, bits;
      config.Encode(token.value, &symbol, &nbits, &bits);
      Require(symbol < gjxl::kPrefixAlphabetSize, "Reference token exceeds prefix carrier");
      builder[token.context].Add(symbol);
    }
  params.max_histograms = 32;
  std::vector<uint32_t> symbols;
  Require(bool(jxl::ClusterHistograms(params, builder, jxl::kClustersLimit, &clustered, &symbols)),
          "Reference partition clustering failed");
  Require(!clustered.empty() && clustered.size() <= 32 && symbols.size() == contexts,
          "Invalid reference partition shape");
  gjxl::EntropyCode partition;
  partition.context_count = contexts;
  partition.context_map.assign(symbols.begin(), symbols.end());
  partition.uint_configs.resize(clustered.size());
  partition.prefix_codes.resize(clustered.size());
  for (size_t c = 0; c < clustered.size(); ++c) {
    std::array<uint64_t, gjxl::kPrefixAlphabetSize> counts{};
    Require(clustered[c].counts.size() <= counts.size(), "Invalid clustered alphabet");
    for (size_t s = 0; s < clustered[c].counts.size(); ++s) {
      Require(clustered[c].counts[s] >= 0, "Negative reference population");
      counts[s] = clustered[c].counts[s];
    }
    Ok(gjxl::BuildPrefixCode(counts, &partition.prefix_codes[c]));
  }
  return partition;
}
class Scope {
public:
  Scope(JxlMemoryManager *memory, std::vector<gjxl::EntropyTokenStreamView> streams,
        std::vector<size_t> ids, size_t contexts, HP params, const Variant &variant, bool tree,
        int64_t scope_id, gjxl::BitWriter *writer, bool measure)
      : memory_(memory), streams_(std::move(streams)), ids_(std::move(ids)) {
    stats.tree = tree;
    stats.id = scope_id;
    stats.contexts = contexts;
    stats.native = tree ? variant.native_tree : variant.native_data;
    for (auto stream : streams_)
      stats.original_tokens += stream.size();
    const auto begin = writer->bits_written();
    if (stats.native) {
      const auto wall = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      const auto cpu = measure ? std::clock() : 0;
      std::vector<uint8_t> reference_map;
      if (!tree && variant.direct_ans_mode >= 0) {
        stats.profiled = !measure;
        Ok(gjxl::codestream_internal::OptimizeDirectAnsEntropyCode(
            streams_, {.context_count = static_cast<uint32_t>(contexts),
                       .maximum_ans_clusters = variant.clusters},
            static_cast<gjxl::codestream_internal::DirectAnsEntropyMode>(variant.direct_ans_mode),
            &native_, nullptr, stats.profiled ? &stats.work : nullptr,
            variant.estimated ? gjxl::AnsSelectionMode::kEstimated : gjxl::AnsSelectionMode::kExact));
      } else if (!tree && variant.reference_partition) {
        native_ = ReferencePartition(streams_, contexts, params);
        reference_map.assign(native_.context_map.begin(), native_.context_map.end());
      } else {
        Ok(gjxl::OptimizeEntropyCode(streams_, {.context_count = static_cast<uint32_t>(contexts)},
                                     &native_));
      }
      if (!tree && variant.direct_ans_mode < 0) {
        gjxl::EntropyCode ans;
        Ok(gjxl::OptimizeAnsEntropyCode(streams_, native_, &ans));
        native_ = std::move(ans);
      }
      if (measure) {
        stats.model_cpu_seconds = double(std::clock() - cpu) / CLOCKS_PER_SEC;
        stats.model_wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall).count();
      }
      if (!reference_map.empty()) {
        Require(std::ranges::equal(reference_map, native_.context_map),
                "Native ANS changed the reference partition");
        stats.reference_partition_exact = true;
      }
      Ok(native::WriteGlobalModelInTransaction(native_, writer));
      stats.prefix = native_.mode == gjxl::EntropyCodingMode::kPrefix;
      stats.coded_tokens = stats.original_tokens;
      stats.clusters = native_.uint_configs.size();
      stats.alphabet_log = stats.prefix ? 15 : native_.ans_log_alpha_size;
      stats.context_map.assign(native_.context_map.begin(), native_.context_map.end());
      for (auto c : native_.uint_configs)
        stats.configs.push_back({c.split_exponent, c.msb_in_token, c.lsb_in_token});
    } else {
      params = Parameters(std::move(params), variant, tree);
      Require(!ids_.empty(), "No reference stream slots");
      tokens_.resize(*std::max_element(ids_.begin(), ids_.end()) + 1);
      for (size_t i = 0; i < streams_.size(); ++i) {
        auto &dest = tokens_[ids_[i]];
        dest.reserve(streams_[i].size());
        for (size_t j = 0; j < streams_[i].size(); ++j) {
          const auto token = streams_[i][j];
          dest.emplace_back(token.context, token.value);
        }
      }
      jxl::BitWriter encoded(memory_);
      const auto cost = jxl::BuildAndEncodeHistograms(
          memory_, params, contexts, tokens_, &reference_, &encoded,
          tree ? jxl::LayerType::ModularTree : jxl::LayerType::ModularGlobal, nullptr);
      Require(cost.ok(), "Reference histogram encoding failed");
      Append(encoded, writer);
      stats.prefix = reference_.use_prefix_code;
      stats.lz77 = reference_.lz77.enabled;
      stats.clusters = reference_.uint_config.size();
      stats.alphabet_log = reference_.log_alpha_size;
      stats.context_map = reference_.context_map;
      for (const auto &stream : tokens_)
        for (const auto &token : stream) {
          ++stats.coded_tokens;
          stats.lz77_lengths += token.is_lz77_length;
        }
      for (auto c : reference_.uint_config)
        stats.configs.push_back({c.split_exponent, c.msb_in_token, c.lsb_in_token});
    }
    stats.model_bits = writer->bits_written() - begin;
  }
  void Write(size_t index, gjxl::BitWriter *writer) {
    const auto begin = writer->bits_written();
    if (stats.native) {
      Ok(native::WriteStreamTokensWithValidatedModel(streams_[index], native_, writer));
    } else {
      jxl::BitWriter encoded(memory_);
      Require(bool(jxl::WriteTokens(tokens_[ids_[index]], reference_, 0, &encoded,
                                    stats.tree ? jxl::LayerType::ModularTree
                                               : jxl::LayerType::ModularGlobal,
                                    nullptr)),
              "Reference token emission failed");
      Append(encoded, writer);
    }
    stats.token_bits += writer->bits_written() - begin;
  }
  ScopeStats stats;

private:
  JxlMemoryManager *memory_;
  std::vector<gjxl::EntropyTokenStreamView> streams_;
  std::vector<size_t> ids_;
  gjxl::EntropyCode native_;
  jxl::EntropyEncodingData reference_{};
  std::vector<std::vector<jxl::Token>> tokens_;
};
} // namespace

void ReplayEntropyAblations(const native::PreparedModularStreamsView &input,
                            const native::ModularStreamPlan &plan,
                            const native::PreparedModularStreamTokens &tokens,
                            std::span<const HP> parameters, const ref::IntegerImage &expected,
                            const std::filesystem::path &output, const std::string &names,
                            std::ostream &out, bool measure) {
  const auto variants = Variants(names);
  Require(!measure || std::ranges::all_of(variants, [](const auto &v) {
            return v.name == "native" || v.direct_ans_mode >= 0;
          }),
          "Model timing requires native-only controls");
  Require(parameters.size() == input.streams.size(), "Missing captured entropy parameters");
  Require(std::filesystem::create_directory(output), "Entropy output already exists");
  JxlMemoryManager memory;
  Require(bool(jxl::MemoryManagerInit(&memory, nullptr)), "Reference memory setup failed");
  native::ModularInputProfile profile;
  Ok(native::ResolveModularInput(input.source, input.format, &profile));
  std::vector<size_t> ids;
  for (const auto &slot : plan.streams)
    ids.push_back(slot.id);
  std::vector<size_t> widths(ids.back() + 1);
  for (size_t i = 0; i < input.streams.size(); ++i) {
    const auto &stream = input.streams[i];
    if (stream.rct || stream.transforms.size)
      profile.metadata.modular_16_bit_buffer_sufficient = false;
    for (const auto &channel : stream.channels)
      widths[ids[i]] = std::max(widths[ids[i]], channel.descriptor.extent.width);
  }
  for (const auto &variant : variants) {
    std::vector<gjxl::BitWriter> sections(plan.geometry.section_count());
    auto &global = sections[0];
    Ok(global.WriteBits(1, 1)); // Same native default DC matrices.
    Ok(global.WriteBits(1, input.global_tree != nullptr));
    size_t stream_header_bits = 2;
    std::vector<ScopeStats> records;
    std::unique_ptr<Scope> shared;
    if (input.global_tree) {
      Scope tree(&memory, {gjxl::EntropyTokenStreamView::Interleaved(tokens.global_tree_tokens)},
                 {0}, native::kTreeContextCount, parameters[0], variant, true, -1, &global, measure);
      tree.Write(0, &global);
      records.push_back(std::move(tree.stats));
      std::vector<gjxl::EntropyTokenStreamView> views(input.streams.size());
      for (size_t i = 0; i < views.size(); ++i)
        if (!input.streams[i].local_tree)
          views[i] = tokens.streams[i];
      auto params = parameters[0];
      params.image_widths = widths;
      shared =
          std::make_unique<Scope>(&memory, std::move(views), ids, (input.global_tree->size + 1) / 2,
                                  std::move(params), variant, false, -1, &global, measure);
    }
    for (size_t i = 0; i < input.streams.size(); ++i) {
      const auto &stream = input.streams[i];
      if (i && !tokens.streams[i].size())
        continue;
      auto &writer = sections[plan.streams[i].section];
      const auto start = writer.bits_written();
      Ok(native::WriteCodingStreamHeader(stream.weighted, stream.rct, stream.transforms,
                                         stream.local_tree == nullptr, &writer));
      stream_header_bits += writer.bits_written() - start;
      if (!tokens.streams[i].size())
        continue;
      if (stream.local_tree) {
        Scope tree(&memory,
                   {gjxl::EntropyTokenStreamView::Interleaved(tokens.local_tree_tokens[i])}, {0},
                   native::kTreeContextCount, parameters[i], variant, true, ids[i], &writer, measure);
        tree.Write(0, &writer);
        records.push_back(std::move(tree.stats));
        auto params = parameters[i];
        params.image_widths.push_back(widths[ids[i]]);
        Scope data(&memory, {tokens.streams[i]}, {0}, (stream.local_tree->size + 1) / 2,
                   std::move(params), variant, false, ids[i], &writer, measure);
        data.Write(0, &writer);
        records.push_back(std::move(data.stats));
      } else {
        Require(shared != nullptr, "Missing shared entropy model");
        shared->Write(i, &writer);
      }
    }
    if (shared)
      records.push_back(std::move(shared->stats));
    size_t section_bits = 0, padding = 0, modeled_bits = 0;
    size_t tree_bits = 0, data_model_bits = 0, data_token_bits = 0;
    for (const auto &section : sections) {
      section_bits += section.bits_written();
      padding += section.padded_size() * 8 - section.bits_written();
    }
    for (const auto &r : records) {
      modeled_bits += r.model_bits + r.token_bits;
      if (r.tree)
        tree_bits += r.model_bits + r.token_bits;
      else {
        data_model_bits += r.model_bits;
        data_token_bits += r.token_bits;
      }
    }
    Require(section_bits == stream_header_bits + modeled_bits,
            "Entropy section accounting differs");
    gjxl::BitWriter file;
    Ok(gjxl::codestream_internal::WriteImageHeader(profile.metadata, &file));
    Ok(gjxl::codestream_internal::WriteFrameHeader(profile.metadata, {}, &file));
    const size_t header_bits = file.bits_written();
    Ok(gjxl::WriteTocAndSections(sections, &file));
    const size_t toc_bits = file.bits_written() - header_bits - section_bits - padding;
    Require(ref::DecodeLossless(file.padded_bytes()) == expected,
            "Entropy ablation decode differs");
    if (variant.name == "native") {
      gjxl::codestream_internal::CodestreamBuffer control;
      Ok(native::EncodePreparedModularStreams(input, gjxl::EntropyCodingMode::kAns, &control));
      Require(std::ranges::equal(control.view(), file.padded_bytes()),
              "Native entropy control bytes differ");
    }
    const auto path = output / (variant.name + ".jxl");
    std::ofstream binary(path, std::ios::binary);
    binary.exceptions(std::ios::badbit | std::ios::failbit);
    binary.write(reinterpret_cast<const char *>(file.padded_bytes().data()), file.padded_size());
    binary.close();
    out << "{\"schema\":1,\"variant\":\"" << variant.name << "\",\"bytes\":" << file.padded_size()
        << ",\"original_tokens\":" << tokens.tokens.size()
        << ",\"decoded_exact\":true"
           ",\"tokens_exact\":true,\"trees_exact\":true,\"timing_qualified\":false"
        << ",\"model_timing_recorded\":" << (measure ? "true" : "false")
        << ",\"native_control_exact\":" << (variant.name == "native" ? "true" : "null")
        << ",\"header_bits\":" << header_bits << ",\"toc_bits\":" << toc_bits
        << ",\"stream_header_bits\":" << stream_header_bits
        << ",\"section_padding_bits\":" << padding << ",\"tree_bits\":" << tree_bits
        << ",\"data_model_bits\":" << data_model_bits << ",\"data_token_bits\":" << data_token_bits
        << ",\"scopes\":[";
    for (size_t i = 0; i < records.size(); ++i) {
      if (i)
        out << ',';
      records[i].Write(out);
    }
    out << "]}\n" << std::flush;
  }
}
} // namespace reference_learning
