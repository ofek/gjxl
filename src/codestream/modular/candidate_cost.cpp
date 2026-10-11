// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/candidate_cost.h"
#include "codestream/headers_internal.h"
#include "codestream/modular/stream_encoder.h"
#include "codestream/modular/tree_codec.h"
#include "codestream/sections.h"
#include <new>
#include <stdexcept>

namespace gjxl::modular_internal {
using namespace codestream_internal;
namespace {
bool Add(uint64_t value, uint64_t *total) {
  if (value > UINT64_MAX - *total) return false;
  *total += value;
  return true;
}
Status Overflow() { return Status::InvalidArgument("Modular candidate size overflow"); }
} // namespace

Status MeasureModularSections(const codec_internal::ImageMetadata &metadata,
                               std::span<const uint64_t> section_bits,
                               ModularCandidateCost *out) try {
  if (!out || section_bits.empty())
    return Status::InvalidArgument("Invalid Modular section-size output");
  auto result = *out;
  result.padding_bits = 0;
  Storage<size_t> sizes;
  sizes.reserve(section_bits.size());
  uint64_t payload_bytes = 0;
  for (uint64_t bits : section_bits) {
    const uint64_t bytes = bits / 8 + (bits % 8 != 0);
    if (bytes > kMaximumTocSectionSize || bytes > SIZE_MAX ||
        !Add(bytes, &payload_bytes) || !Add((8 - bits % 8) % 8, &result.padding_bits))
      return Overflow();
    sizes.push_back(static_cast<size_t>(bytes));
  }
  BitWriter file;
  if (auto s = WriteImageHeader(metadata, &file); !s.ok()) return s;
  if (auto s = WriteFrameHeader(metadata, {}, &file); !s.ok()) return s;
  result.frame_header_bits = file.bits_written();
  if (auto s = WriteTocSizes(sizes, &file); !s.ok()) return s;
  result.toc_bits = file.bits_written() - result.frame_header_bits;
  if (!file.byte_aligned() || !Add(file.padded_size(), &payload_bytes) ||
      payload_bytes > SIZE_MAX)
    return Overflow();
  result.bytes = static_cast<size_t>(payload_bytes);
  *out = result;
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular section measurement allocation failed");
} catch (const std::length_error &) {
  return Overflow();
}

Status MeasureModularCandidate(const codec_internal::ImageMetadata &metadata,
                                const ModularStreamPlan &layout,
                                const PreparedModularTokens &tokens, const EntropyCode &model,
                                const EntropyCodeCost &token_cost,
                                ModularCandidateCost *out) try {
  if (!out || tokens.streams.empty() || !layout.geometry.section_count() ||
      tokens.streams.size() != layout.streams.size() ||
      token_cost.section_token_bits.size() != tokens.streams.size())
    return Status::InvalidArgument("Invalid Modular candidate measurement handoff");
  ModularCandidateCost result;
  result.contexts = tokens.context_count;
  result.clusters = model.mode == EntropyCodingMode::kAns ? model.ans_histograms.size()
                                                        : model.prefix_codes.size();
  Storage<uint64_t> bits(layout.geometry.section_count(), 0);
  GlobalTreeStoragePlan tree;
  EntropyModelStoragePlan model_plan;
  if (auto s = ComputeGlobalTreeStoragePlan(tokens.tree_token_count, &tree); !s.ok()) return s;
  if (auto s = ComputeEntropyModelStoragePlan(model.mode, model.context_count, result.clusters,
                                              &model_plan); !s.ok()) return s;
  uint64_t maximum = 2; // default DC matrices and disabled LZ77
  if (!Add(tree.maximum_bits, &maximum) || !Add(model_plan.maximum_bits, &maximum) ||
      !Add(MaximumCodingStreamHeaderBits(tokens.policy.transforms), &maximum) || maximum > SIZE_MAX)
    return Overflow();
  BitWriter global;
  if (auto s = global.WithMaxBits(static_cast<size_t>(maximum), [&]() -> Status {
        if (auto s = global.WriteBits(1, 1); !s.ok()) return s;
        const size_t tree_start = global.bits_written();
        if (auto s = WriteGlobalTreeInTransaction(
                std::span(tokens.tree_tokens).first(tokens.tree_token_count), &global); !s.ok())
          return s;
        result.tree_bits = global.bits_written() - tree_start;
        const size_t model_start = global.bits_written();
        if (auto s = WriteGlobalModelInTransaction(model, &global); !s.ok()) return s;
        result.model_bits = global.bits_written() - model_start;
        const size_t stream_start = global.bits_written();
        if (auto s = WriteCodingStreamHeader(tokens.policy.weighted, tokens.policy.rct,
                                             tokens.policy.transforms, &global); !s.ok()) return s;
        result.stream_header_bits = 1 + global.bits_written() - stream_start;
        return Status::Ok();
      }); !s.ok()) return s;
  bits[0] = global.bits_written();
  // Verify that the retained exact cost describes the model just serialized.
  if (!result.model_bits || result.model_bits - 1 != token_cost.model_bits)
    return Status::InvalidArgument("Modular model cost differs from serialization");
  BitWriter group_header;
  if (auto s = WriteCodingStreamHeader(tokens.policy.weighted, 0, &group_header); !s.ok()) return s;
  for (size_t i = 0; i < tokens.streams.size(); ++i) {
    const size_t section = layout.streams[i].section;
    if (section >= bits.size() || tokens.streams[i].size() > SIZE_MAX - result.tokens)
      return Overflow();
    result.tokens += tokens.streams[i].size();
    if (tokens.streams[i].size() == 0) continue;
    if (i && (!Add(group_header.bits_written(), &bits[section]) ||
              !Add(group_header.bits_written(), &result.stream_header_bits))) return Overflow();
    if (!Add(token_cost.section_token_bits[i], &bits[section]) ||
        !Add(token_cost.section_token_bits[i], &result.payload_bits)) return Overflow();
  }
  if (auto s = MeasureModularSections(metadata, bits, &result); !s.ok()) return s;
  uint64_t sum = 0;
  for (uint64_t component : {result.payload_bits, result.tree_bits, result.model_bits,
       result.stream_header_bits, result.frame_header_bits, result.toc_bits, result.padding_bits})
    if (!Add(component, &sum)) return Overflow();
  if (result.bytes > UINT64_MAX / 8 || sum != result.bytes * 8)
    return Status::Internal("Modular candidate categories do not cover the complete file");
  *out = result;
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular candidate measurement allocation failed");
} catch (const std::length_error &) {
  return Overflow();
}
} // namespace gjxl::modular_internal
