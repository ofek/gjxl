// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/candidate_screen.h"
#include "codestream/ans_internal.h"
#include "codestream/modular/stream_encoder.h"
#include "codestream/modular/tree_codec.h"
#include "codec/modular/profile.h"
#include <cmath>

namespace gjxl::modular_internal {
using namespace codestream_internal;
Status ScreenModularCandidate(const codec_internal::ImageMetadata &metadata,
                               const ModularStreamPlan &layout, const PreparedModularTokens &tokens,
                               EntropyCodingMode mode, ModularCandidateEstimate *out) try {
  if (!out || tokens.streams.empty() || tokens.streams.size() != layout.streams.size() ||
      (mode != EntropyCodingMode::kPrefix && mode != EntropyCodingMode::kAns))
    return Status::InvalidArgument("Invalid Modular screening handoff");
  size_t count = 0;
  for (auto stream : tokens.streams) {
    if (!stream.valid() || stream.size() > SIZE_MAX - count)
      return Status::InvalidArgument("Modular screening token count overflow");
    count += stream.size();
  }
  const size_t stride = std::max(size_t{1}, count / kMaximumScreenSamples + (count % kMaximumScreenSamples != 0));
  Storage<size_t> offsets;
  offsets.reserve(tokens.streams.size() + 1);
  offsets.push_back(0);
  for (auto stream : tokens.streams) {
    const size_t quota = stream.size() / stride + (stream.size() % stride != 0);
    if (quota > SIZE_MAX - offsets.back())
      return Status::InvalidArgument("Modular screening sample count overflow");
    offsets.push_back(offsets.back() + quota);
  }
  Storage<EntropyToken> sampled(offsets.back());
  Storage<EntropyTokenStreamView> streams;
  streams.reserve(tokens.streams.size());
  for (size_t s = 0; s < tokens.streams.size(); ++s) {
    const auto stream = tokens.streams[s];
    const size_t quota = offsets[s + 1] - offsets[s];
    for (size_t i = 0; i < quota; ++i) {
      // Partition into nearly equal intervals, selecting each midpoint. All
      // intermediate products are <= the stream length, including huge inputs.
      const size_t base = stream.size() / quota, extra = stream.size() % quota;
      const size_t start = i * base + std::min(i, extra);
      const size_t width = base + (i < extra);
      sampled[offsets[s] + i] = stream[start + width / 2];
    }
    streams.push_back(EntropyTokenStreamView::Interleaved(std::span(sampled).subspan(offsets[s], quota)));
  }
  EntropyCode model;
  auto *profile = active_modular_profile ? &active_modular_profile->entropy_work : nullptr;
  const EntropyCodeOptions options{.context_count = static_cast<uint32_t>(tokens.context_count)};
  if (mode == EntropyCodingMode::kAns) {
    PreparedEntropyClusters values;
    EntropyCode prefix;
    if (auto s = OptimizeEntropyCodeAndPrepareClusters(streams, options, &prefix, nullptr,
                                                       &values, profile); !s.ok()) return s;
    if (auto s = OptimizeAnsEntropyCodeWithPreparedClusters(streams, prefix, values, &model,
        nullptr, profile, AnsSelectionMode::kEstimated); !s.ok()) return s;
  } else if (auto s = OptimizeEntropyCode(streams, options, &model, nullptr, profile); !s.ok()) return s;
  Storage<uint64_t> section_bits(layout.geometry.section_count(), 0);
  if (section_bits.empty()) return Status::InvalidArgument("Empty Modular screening geometry");
  GlobalTreeStoragePlan tree;
  if (auto s = ComputeGlobalTreeStoragePlan(tokens.tree_token_count, &tree); !s.ok()) return s;
  BitWriter global;
  if (auto s = global.WithMaxBits(1 + tree.maximum_bits, [&]() -> Status {
      if (auto s = global.WriteBits(1, 1); !s.ok()) return s;
      return WriteGlobalTreeInTransaction(std::span(tokens.tree_tokens).first(tokens.tree_token_count), &global);
    }); !s.ok()) return s;
  if (auto s = WriteGlobalModelInTransaction(model, &global); !s.ok()) return s;
  if (auto s = WriteCodingStreamHeader(tokens.policy.weighted, tokens.policy.rct,
      tokens.policy.transforms, &global); !s.ok()) return s;
  section_bits[0] = global.bits_written();
  BitWriter group;
  if (auto s = WriteCodingStreamHeader(tokens.policy.weighted, 0, &group); !s.ok()) return s;
  for (size_t s = 0; s < streams.size(); ++s) {
    if (!streams[s].size()) continue;
    double bits = 0;
    for (size_t i = 0; i < streams[s].size(); ++i) {
      const auto token = streams[s][i];
      const size_t cluster = model.context_map[token.context];
      const auto encoded = EncodeHybridUintValidated(token.value, model.uint_configs[cluster]);
      bits += encoded.extra_bit_count;
      if (mode == EntropyCodingMode::kAns) {
        const auto frequency = model.ans_histograms[cluster].frequencies[encoded.symbol];
        if (!frequency) return Status::Internal("Screening model omits a sampled symbol");
        bits += 12.0 - std::log2(double(frequency));
      } else {
        const auto &prefix = model.prefix_codes[cluster];
        bits += prefix.degenerate_symbol == encoded.symbol ? 0 : prefix.depths[encoded.symbol];
      }
    }
    bits *= double(tokens.streams[s].size()) / streams[s].size();
    if (mode == EntropyCodingMode::kAns) bits += 32;
    if (s) bits += group.bits_written();
    const size_t section = layout.streams[s].section;
    if (section >= section_bits.size() || !std::isfinite(bits) || bits < 0 || bits >= double(UINT64_MAX) ||
        uint64_t(std::ceil(bits)) > UINT64_MAX - section_bits[section])
      return Status::InvalidArgument("Modular screening cost overflow");
    section_bits[section] += uint64_t(std::ceil(bits));
  }
  ModularCandidateCost size;
  if (auto s = MeasureModularSections(metadata, section_bits, &size); !s.ok()) return s;
  *out = {double(size.bytes) * 8, sampled.size()};
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular screening allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular screening storage overflow");
}
} // namespace gjxl::modular_internal
