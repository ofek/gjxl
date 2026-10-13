// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/prepared_streams.h"
#include "codec/modular/profile.h"
#include "codestream/ans_internal.h"
#include "codestream/headers_internal.h"
#include "codestream/modular/stream_encoder.h"
#include "codestream/modular/tree_codec.h"
#include "codestream/sections.h"
#include "core/thread_budget.h"
#include <algorithm>

namespace gjxl::modular_internal {
namespace {
using namespace codestream_internal;
using namespace resource_budget_internal;
Status Invalid() { return Status::InvalidArgument("Invalid prepared Modular streams"); }
Status Overflow() { return Status::InvalidArgument("Prepared Modular stream storage overflow"); }
using EntropyPolicy = PreparedModularEntropyPolicy;
Status ValidateOptions(const PreparedModularEntropyOptions &options) {
  if (auto s = ValidateAnsSelectionMode(options.selection); !s.ok())
    return s;
  if (options.policy == EntropyPolicy::kDirectRateAns)
    return options.maximum_ans_clusters && options.maximum_ans_clusters <= kMaximumAnsClusters
               ? Status::Ok()
               : Invalid();
  return (options.policy == EntropyPolicy::kPrefix ||
          options.policy == EntropyPolicy::kAnsFromPrefix) &&
                 options.maximum_ans_clusters == kDefaultDirectAnsClusters
             ? Status::Ok()
             : Invalid();
}
EntropyCodingMode CodingMode(const PreparedModularEntropyOptions &options) {
  return options.policy == EntropyPolicy::kPrefix ? EntropyCodingMode::kPrefix
                                                  : EntropyCodingMode::kAns;
}
PreparedModularEntropyOptions LegacyOptions(EntropyCodingMode mode,
                                            AnsSelectionMode selection = AnsSelectionMode::kExact) {
  return {.policy = mode == EntropyCodingMode::kPrefix ? EntropyPolicy::kPrefix
                                                       : EntropyPolicy::kAnsFromPrefix,
          .selection = selection};
}
bool Add(size_t value, size_t *total) {
  if (value > SIZE_MAX - *total)
    return false;
  *total += value;
  return true;
}
struct FrameInfo {
  ModularInputProfile profile;
  ModularFrameGeometry geometry;
  ChannelShape full;
  size_t global_end = 0;
  TreeLayout global_tree;
};
struct StreamInfo {
  size_t id = 0, section = 0, tokens = 0, contexts = 0, tree_tokens = 0, width = 0;
  bool present = false, weighted = false;
};
Status DescribeFrame(const PreparedModularStreamsView &input, FrameInfo *out) {
  FrameInfo p;
  if (auto s = ResolveModularInput(input.source, input.format, &p.profile); !s.ok())
    return s;
  if (input.streams.empty())
    return Invalid();
  if (auto s = ValidateTransformBitDepth(input.streams[0].transforms, p.profile.metadata.bits); !s.ok())
    return s;
  if (auto s = ModularFrameGeometry::Create(input.source, &p.geometry); !s.ok())
    return s;
  if (input.streams.size() != 1 + p.geometry.dc_group_count() + p.geometry.group_count())
    return Invalid();
  const auto &global = input.streams[0];
  if (global.rct >= 42 || (global.rct && p.profile.channel_count < 3))
    return Invalid();
  std::array<ChannelShape, kMaximumTransforms + 1> shapes;
  if (auto s = PlanTransforms(p.profile.channels(), 0, global.transforms, &shapes); !s.ok())
    return s;
  p.full = shapes[global.transforms.size];
  ModularStreamStoragePlan placement;
  if (auto s = ComputeModularStreamStoragePlan(p.geometry, p.full.channels(), p.full.metadata,
                                               &placement);
      !s.ok())
    return s;
  p.global_end = p.full.metadata;
  while (p.global_end < p.full.count &&
         p.full.descriptors[p.global_end].extent.width <= kGroupDimension &&
         p.full.descriptors[p.global_end].extent.height <= kGroupDimension)
    ++p.global_end;
  if (input.global_tree) {
    if (auto s = ValidateTree(*input.global_tree, &p.global_tree); !s.ok())
      return s;
    const uint64_t population =
        uint64_t{input.source.width} * input.source.height * p.profile.channel_count;
    if (p.global_tree.leaves > MaximumGlobalTreeLeaves(population))
      return Invalid();
  }
  for (const auto &stream : input.streams)
    if (stream.rct || stream.transforms.size)
      p.profile.metadata.modular_16_bit_buffer_sufficient = false;
  *out = p;
  return Status::Ok();
}
Status DescribeStream(const PreparedModularStreamsView &input, const FrameInfo &frame, size_t index,
                      StreamInfo *out, TreeLayout *local_layout = nullptr) {
  const auto &stream = input.streams[index];
  if (!stream.weighted.valid() || stream.rct >= 42)
    return Invalid();
  if (auto s = ValidateTransformBitDepth(stream.transforms, frame.profile.metadata.bits); !s.ok())
    return s;
  StreamInfo p;
  ChannelShape expected;
  if (index == 0) {
    expected = frame.full;
    expected.count = frame.global_end;
    p.present = true; // Even an empty global prefix carries its transform header.
  } else {
    const bool dc = index <= frame.geometry.dc_group_count();
    const size_t group = dc ? index - 1 : index - 1 - frame.geometry.dc_group_count();
    if (auto s = ModularStreamId(frame.geometry, dc ? StreamRole::kDcGroup : StreamRole::kGroup,
                                 group, &p.id);
        !s.ok())
      return s;
    p.section = frame.geometry.group_count() == 1
                    ? 0
                    : (dc ? 1 + group : 2 + frame.geometry.dc_group_count() + group);
    ModularRect rect;
    if (auto s = frame.geometry.GroupRect(group, dc, &rect); !s.ok())
      return s;
    ChannelShape original;
    // Same shifted/clipped channel order as BuildModularStreamPlan. Full shape
    // validation above establishes the shift and group-dimension preconditions.
    for (size_t c = frame.global_end; c < frame.full.count; ++c) {
      auto ch = frame.full.descriptors[c];
      if ((std::min(ch.hshift, ch.vshift) >= 3) != dc)
        continue;
      const size_t x = rect.x >> ch.hshift, y = rect.y >> ch.vshift;
      if (x >= ch.extent.width || y >= ch.extent.height)
        continue;
      ch.extent = {std::min(rect.extent.width >> ch.hshift, ch.extent.width - x),
                   std::min(rect.extent.height >> ch.vshift, ch.extent.height - y)};
      if (!ch.extent.empty())
        original.descriptors[original.count++] = ch;
    }
    if (original.count == 0) {
      if (!stream.channels.empty() || stream.transforms.size || stream.rct || stream.local_tree)
        return Invalid();
      *out = p;
      return Status::Ok();
    }
    p.present = true;
    if (stream.rct) {
      if (original.count < 3)
        return Invalid();
      for (size_t c = 1; c < 3; ++c)
        if (original.descriptors[c].extent != original.descriptors[0].extent ||
            original.descriptors[c].hshift != original.descriptors[0].hshift ||
            original.descriptors[c].vshift != original.descriptors[0].vshift)
          return Invalid();
    }
    std::array<ChannelShape, kMaximumTransforms + 1> shapes;
    if (auto s = PlanTransforms(original.channels(), 0, stream.transforms, &shapes); !s.ok())
      return s;
    expected = shapes[stream.transforms.size];
  }
  if (stream.channels.size() != expected.count)
    return Invalid();
  for (size_t c = 0; c < expected.count; ++c) {
    const auto &view = stream.channels[c];
    if (view.descriptor != expected.descriptors[c])
      return Invalid();
    if (auto s = view.Validate(); !s.ok())
      return s;
    size_t area;
    if (!view.descriptor.extent.try_area(&area) || !Add(area, &p.tokens))
      return Overflow();
    p.width = std::max(p.width, view.descriptor.extent.width);
  }
  TreeLayout local;
  const TreeLayout *layout = &frame.global_tree;
  const DecisionTree *tree = input.global_tree;
  if (stream.local_tree) {
    tree = stream.local_tree;
    if (auto s = ValidateTree(*tree, &local); !s.ok())
      return s;
    // Local decoder limit uses transformed coded pixels, unlike a global tree.
    if (tree->size > std::min(uint64_t{1} << 20, uint64_t{1024} + p.tokens))
      return Invalid();
    layout = &local;
    if (local_layout)
      *local_layout = local;
  } else if (p.tokens && !tree)
    return Invalid();
  if (tree) {
    p.contexts = layout->leaves;
    p.weighted = layout->weighted;
    if (p.tokens && stream.local_tree)
      p.tree_tokens = ModularTreeTokenCount(*tree, *layout);
  }
  *out = p;
  return Status::Ok();
}
struct ModelBound {
  size_t bits = 0;
  HostStorageBound owned, optimization, writing;
};
Status ModelStorage(size_t tokens, size_t contexts, size_t streams,
                    const PreparedModularEntropyOptions &entropy, ModelBound *out) {
  ModelBound p;
  const bool direct = entropy.policy == EntropyPolicy::kDirectRateAns;
  const auto mode = CodingMode(entropy);
  EntropyOptimizationStorageOptions options{
      .policy = direct ? EntropyStoragePolicy::kRateOptimizedAns : EntropyStoragePolicy::kPrefix,
      .tokens = tokens,
      .contexts = contexts,
      .sections = streams,
      .return_cost = false,
      .maximum_ans_clusters = entropy.maximum_ans_clusters};
  EntropyOptimizationStoragePlan prefix, ans;
  if (auto s = ComputeEntropyOptimizationStoragePlan(options, &prefix); !s.ok())
    return s;
  p.optimization = prefix.working;
  if (entropy.policy == EntropyPolicy::kAnsFromPrefix) {
    options.policy = EntropyStoragePolicy::kAnsFromPrefix;
    if (auto s = ComputeEntropyOptimizationStoragePlan(options, &ans); !s.ok())
      return s;
    if (!p.optimization.Add(ans.working))
      return Overflow();
  }
  EntropyModelStoragePlan model;
  if (auto s = ComputeEntropyModelStoragePlan(
          mode, contexts,
          std::min(contexts, direct ? entropy.maximum_ans_clusters : kMaximumPrefixClusters),
          &model);
      !s.ok())
    return s;
  p.bits = 1 + model.maximum_bits; // No LZ77, followed by the model.
  p.owned = model.owned;
  p.writing = model.write_scratch;
  *out = p;
  return Status::Ok();
}
struct StreamBound {
  size_t bits = 0;
  HostStorageBound scratch;
};
Status StreamStorage(const PreparedModularStreamView &stream, const StreamInfo &info,
                     const PreparedModularEntropyOptions &entropy, StreamBound *out) {
  StreamBound p;
  if (!info.present) {
    *out = p;
    return Status::Ok();
  }
  p.bits = MaximumCodingStreamHeaderBits(stream.transforms);
  if (info.tokens) {
    EntropyTokenEmissionStoragePlan payload;
    if (auto s = ComputeEntropyTokenEmissionStoragePlan(CodingMode(entropy), info.tokens, &payload);
        !s.ok())
      return s;
    if (!Add(payload.maximum_bits, &p.bits) || !p.scratch.Add(payload.scratch))
      return Overflow();
    if (stream.local_tree) {
      GlobalTreeStoragePlan tree;
      ModelBound model;
      if (auto s = ComputeGlobalTreeStoragePlan(info.tree_tokens, &tree); !s.ok())
        return s;
      if (auto s = ModelStorage(info.tokens, info.contexts, 1, entropy, &model); !s.ok())
        return s;
      if (!Add(tree.maximum_bits, &p.bits) || !Add(model.bits, &p.bits) ||
          !p.scratch.Add(tree.scratch) || !p.scratch.Add(model.optimization) ||
          !p.scratch.Add(model.writing))
        return Overflow();
    }
  }
  *out = p;
  return Status::Ok();
}
Status Plan(const PreparedModularStreamsView &input, const PreparedModularEntropyOptions &entropy,
            ModularWorkflowStoragePlan *out, FrameInfo *frame_out = nullptr) {
  if (!out)
    return Invalid();
  if (auto s = ValidateOptions(entropy); !s.ok())
    return s;
  FrameInfo frame;
  if (auto s = DescribeFrame(input, &frame); !s.ok())
    return s;
  ModularWorkflowStoragePlan p;
  p.geometry = frame.geometry;
  p.streams = input.streams.size();
  size_t tree_tokens =
      input.global_tree ? ModularTreeTokenCount(*input.global_tree, frame.global_tree) : 0;
  size_t global_tokens = 0, weighted_width = 0, section_bits = 0;
  HostStorageBound sections, scratch;
  if (!sections.AddVector<BitWriter>(p.geometry.section_count(), VectorCapacityPolicy::kFreshExact))
    return Overflow();
  // Count scopes first; a local stream never contributes to the global model.
  for (size_t i = 0; i < p.streams; ++i) {
    StreamInfo info;
    if (auto s = DescribeStream(input, frame, i, &info); !s.ok())
      return s;
    if (!Add(info.tokens, &p.tokens) || !Add(info.tree_tokens, &tree_tokens) ||
        (!input.streams[i].local_tree && !Add(info.tokens, &global_tokens)))
      return Overflow();
    if (info.weighted)
      weighted_width = std::max(weighted_width, info.width);
  }
  HostStorageBound owned;
  if (!owned.AddVector<EntropyToken>(p.tokens, VectorCapacityPolicy::kFreshExact) ||
      !owned.AddVector<EntropyToken>(tree_tokens, VectorCapacityPolicy::kFreshExact) ||
      !owned.AddVector<EntropyTokenStreamView>(p.streams, VectorCapacityPolicy::kFreshExact) ||
      !owned.AddVector<std::span<const EntropyToken>>(p.streams, VectorCapacityPolicy::kFreshExact))
    return Overflow();
  p.preparation = owned;
  if (weighted_width && !p.preparation.AddVector<uint32_t>((weighted_width + 2) * 2,
                                                           VectorCapacityPolicy::kFreshExact, 5))
    return Overflow();
  ModelBound global_model;
  p.maximum_global_bits = 1; // Default DC matrices.
  if (input.global_tree) {
    GlobalTreeStoragePlan tree;
    if (auto s = ComputeGlobalTreeStoragePlan(
            ModularTreeTokenCount(*input.global_tree, frame.global_tree), &tree);
        !s.ok())
      return s;
    if (auto s = ModelStorage(global_tokens, frame.global_tree.leaves, p.streams, entropy,
                              &global_model);
        !s.ok())
      return s;
    if (!Add(tree.maximum_bits, &p.maximum_global_bits) ||
        !Add(global_model.bits, &p.maximum_global_bits) || !scratch.Add(tree.scratch) ||
        !scratch.Add(global_model.writing))
      return Overflow();
  } else if (!Add(1, &p.maximum_global_bits))
    return Overflow(); // No global tree.
  for (size_t i = 0; i < p.streams; ++i) {
    StreamInfo info;
    StreamBound stream;
    if (auto s = DescribeStream(input, frame, i, &info); !s.ok())
      return s;
    if (auto s = StreamStorage(input.streams[i], info, entropy, &stream); !s.ok())
      return s;
    scratch.peak_bytes = std::max(scratch.peak_bytes, stream.scratch.peak_bytes);
    if (i == 0) {
      if (!Add(stream.bits, &p.maximum_global_bits) || !Add(7, &p.maximum_global_bits))
        return Overflow();
    } else if (info.present) {
      size_t bits = stream.bits;
      if (!Add(7, &bits))
        return Overflow();
      p.maximum_group_bits = std::max(p.maximum_group_bits, bits);
      if (bits / 8 + 1 > kMaximumTocSectionSize)
        return Invalid();
      HostStorageBound writer;
      if (auto s = ComputeEntropyWriterStorageBound(bits, &writer); !s.ok())
        return s;
      if (!sections.Add(writer) || !Add(bits, &section_bits))
        return Overflow();
    }
  }
  if (p.maximum_global_bits / 8 + 1 > kMaximumTocSectionSize)
    return Invalid();
  HostStorageBound writer;
  if (auto s = ComputeEntropyWriterStorageBound(p.maximum_global_bits, &writer); !s.ok())
    return s;
  if (!sections.Add(writer) || !Add(p.maximum_global_bits, &section_bits))
    return Overflow();
  p.modeling = owned;
  if (!p.modeling.Add(global_model.optimization) ||
      (input.global_tree &&
       !p.modeling.AddVector<EntropyTokenStreamView>(p.streams, VectorCapacityPolicy::kFreshExact)))
    return Overflow();
  p.emission = owned;
  if (!p.emission.Add(global_model.owned) || !p.emission.Add(sections) || !p.emission.Add(scratch))
    return Overflow();
  constexpr size_t header_bits = 15 + 256 + 128;
  if (p.geometry.section_count() > (SIZE_MAX - header_bits) / 32)
    return Overflow();
  size_t file_bits = section_bits;
  if (!Add(32 * p.geometry.section_count() + header_bits, &file_bits))
    return Overflow();
  HostStorageBound file_writer;
  CommonHeaderStoragePlan headers;
  if (auto s = ComputeEntropyWriterStorageBound(file_bits, &file_writer); !s.ok())
    return s;
  if (auto s = ComputeCommonHeaderStoragePlan(0, &headers); !s.ok())
    return s;
  p.assembly = sections;
  if (!p.assembly.Add(file_writer) || !p.assembly.Add(headers.temporary) ||
      !p.assembly.AddVector<size_t>(p.geometry.section_count(), VectorCapacityPolicy::kFreshExact))
    return Overflow();
  p.maximum_codestream_bytes = file_bits / 8 + (file_bits % 8 != 0);
  if (!p.output.AddVector<uint8_t>(p.maximum_codestream_bytes, VectorCapacityPolicy::kFreshExact))
    return Overflow();
  p.publication = file_writer;
  if (!p.publication.Add(p.output))
    return Overflow();
  p.working = p.output;
  for (auto phase : {p.preparation, p.modeling, p.emission, p.assembly, p.publication})
    p.working.peak_bytes = std::max(p.working.peak_bytes, phase.peak_bytes);
  if (frame_out)
    *frame_out = frame;
  *out = p;
  return Status::Ok();
}
Status Tokenize(const PreparedModularStreamsView &input, const FrameInfo &frame,
                PreparedModularStreamTokens *out) {
  ProfileScope profile(ProfileStage::kTokens);
  size_t total = 0, total_tree = input.global_tree
                                     ? ModularTreeTokenCount(*input.global_tree, frame.global_tree)
                                     : 0;
  for (size_t i = 0; i < input.streams.size(); ++i) {
    StreamInfo info;
    if (auto s = DescribeStream(input, frame, i, &info); !s.ok())
      return s;
    if (!Add(info.tokens, &total) || !Add(info.tree_tokens, &total_tree))
      return Overflow();
  }
  PreparedModularStreamTokens result;
  result.tokens.resize(total);
  result.tree_tokens.resize(total_tree);
  result.streams.resize(input.streams.size());
  result.local_tree_tokens.resize(input.streams.size());
  size_t next = 0, next_tree = 0;
  if (input.global_tree) {
    next_tree = ModularTreeTokenCount(*input.global_tree, frame.global_tree);
    auto span = std::span(result.tree_tokens).first(next_tree);
    TokenizeValidatedModularTree(*input.global_tree, frame.global_tree, span);
    result.global_tree_tokens = span;
  }
  for (size_t i = 0; i < input.streams.size(); ++i) {
    const auto &stream = input.streams[i];
    StreamInfo info;
    TreeLayout local;
    if (auto s = DescribeStream(input, frame, i, &info, &local); !s.ok())
      return s;
    const auto *tree = stream.local_tree ? stream.local_tree : input.global_tree;
    const auto &layout = stream.local_tree ? local : frame.global_tree;
    const size_t first = next;
    if (info.tree_tokens) {
      auto span = std::span(result.tree_tokens).subspan(next_tree, info.tree_tokens);
      TokenizeValidatedModularTree(*tree, layout, span);
      result.local_tree_tokens[i] = span;
      next_tree += info.tree_tokens;
    }
    for (size_t c = 0; c < stream.channels.size(); ++c) {
      const auto &channel = stream.channels[c];
      const size_t count = channel.descriptor.extent.width * channel.descriptor.extent.height;
      if (!count)
        continue; // Preserve channel IDs without dereferencing an absent tree.
      if (auto s =
              TokenizeValidatedModularChannel(channel, c, info.id, *tree, layout, stream.weighted,
                                              std::span(result.tokens).subspan(next, count));
          !s.ok())
        return s;
      next += count;
    }
    result.streams[i] =
        EntropyTokenStreamView::Interleaved(std::span(result.tokens).subspan(first, next - first));
  }
  *out = std::move(result);
  return Status::Ok();
}
Status MakeModel(std::span<const EntropyTokenStreamView> streams, size_t contexts,
                 const PreparedModularEntropyOptions &entropy, EntropyCode *out) {
  ProfileScope profile(ProfileStage::kModel);
  if (entropy.policy == EntropyPolicy::kDirectRateAns)
    return OptimizeDirectAnsEntropyCode(streams,
                                        {.context_count = static_cast<uint32_t>(contexts),
                                         .maximum_ans_clusters = entropy.maximum_ans_clusters},
                                        DirectAnsEntropyMode::kRateOptimized, out, nullptr, nullptr,
                                        entropy.selection);
  if (auto s =
          OptimizeEntropyCode(streams, {.context_count = static_cast<uint32_t>(contexts)}, out);
      !s.ok())
    return s;
  if (entropy.policy == EntropyPolicy::kAnsFromPrefix) {
    EntropyCode ans;
    if (auto s = OptimizeAnsEntropyCode(streams, *out, &ans, nullptr, nullptr, entropy.selection);
        !s.ok())
      return s;
    *out = std::move(ans);
  }
  return Status::Ok();
}
} // namespace

Status ComputePreparedModularStreamsStoragePlan(const PreparedModularStreamsView &input,
                                                EntropyCodingMode mode,
                                                ModularWorkflowStoragePlan *out) {
  if (mode != EntropyCodingMode::kPrefix && mode != EntropyCodingMode::kAns)
    return Invalid();
  return Plan(input, LegacyOptions(mode), out);
}
Status ComputePreparedModularStreamsStoragePlan(const PreparedModularStreamsView &input,
                                                const PreparedModularEntropyOptions &options,
                                                ModularWorkflowStoragePlan *out) {
  return Plan(input, options, out);
}
Status TokenizePreparedModularStreams(const PreparedModularStreamsView &input,
                                      PreparedModularStreamTokens *out) try {
  if (!out)
    return Invalid();
  FrameInfo frame;
  if (auto s = DescribeFrame(input, &frame); !s.ok())
    return s;
  return Tokenize(input, frame, out);
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular stream token allocation failed");
} catch (const std::length_error &) {
  return Overflow();
}

Status EncodePreparedModularStreams(const PreparedModularStreamsView &input, EntropyCodingMode mode,
                                    CodestreamBuffer *out, AnsSelectionMode selection) {
  if (mode != EntropyCodingMode::kPrefix && mode != EntropyCodingMode::kAns)
    return Invalid();
  return EncodePreparedModularStreams(input, LegacyOptions(mode, selection), out);
}
Status EncodePreparedModularStreams(const PreparedModularStreamsView &input,
                                    const PreparedModularEntropyOptions &entropy,
                                    CodestreamBuffer *out) try {
  if (!out)
    return Invalid();
  ModularWorkflowStoragePlan plan;
  FrameInfo frame;
  if (auto s = Plan(input, entropy, &plan, &frame); !s.ok())
    return s;
  thread_budget_internal::EncodeScope serial(1);
  Storage<BitWriter> sections;
  {
    PreparedModularStreamTokens tokens;
    if (auto s = Tokenize(input, frame, &tokens); !s.ok())
      return s;
    EntropyCode global_model;
    if (input.global_tree) {
      Storage<EntropyTokenStreamView> shared(input.streams.size());
      for (size_t i = 0; i < shared.size(); ++i)
        if (!input.streams[i].local_tree)
          shared[i] = tokens.streams[i];
      if (auto s = MakeModel(shared, frame.global_tree.leaves, entropy, &global_model); !s.ok())
        return s;
    }
    sections.resize(plan.geometry.section_count());
    auto emit = [&](size_t i, BitWriter *writer) -> Status {
      const auto &stream = input.streams[i];
      EntropyCode local_model;
      const EntropyCode *model = &global_model;
      if (stream.local_tree && tokens.streams[i].size()) {
        if (auto s = MakeModel(std::span(tokens.streams).subspan(i, 1),
                               (stream.local_tree->size + 1) / 2, entropy, &local_model);
            !s.ok())
          return s;
        model = &local_model;
      }
      ProfileScope emission(ProfileStage::kEmission);
      if (auto s = WriteCodingStreamHeader(stream.weighted, stream.rct, stream.transforms,
                                           stream.local_tree == nullptr, writer);
          !s.ok())
        return s;
      if (!tokens.streams[i].size())
        return Status::Ok();
      if (stream.local_tree) {
        if (auto s = WriteLocalTreeInTransaction(tokens.local_tree_tokens[i], writer); !s.ok())
          return s;
        if (auto s = WriteGlobalModelInTransaction(*model, writer); !s.ok())
          return s;
      }
      return WriteStreamTokensWithValidatedModel(tokens.streams[i], *model, writer);
    };
    auto &global = sections[0];
    if (auto s = global.WithMaxBits(
            plan.maximum_global_bits,
            [&]() -> Status {
              {
                ProfileScope emission(ProfileStage::kEmission);
                if (auto s = global.WriteBits(1, 1); !s.ok())
                  return s;
                if (input.global_tree) {
                  if (auto s = WriteGlobalTreeInTransaction(tokens.global_tree_tokens, &global);
                      !s.ok())
                    return s;
                  if (auto s = WriteGlobalModelInTransaction(global_model, &global); !s.ok())
                    return s;
                } else if (auto s = global.WriteBits(1, 0); !s.ok())
                  return s;
              }
              return emit(0, &global);
            });
        !s.ok())
      return s;
    for (size_t i = 1; i < input.streams.size(); ++i) {
      if (!tokens.streams[i].size())
        continue;
      StreamInfo info;
      StreamBound bound;
      if (auto s = DescribeStream(input, frame, i, &info); !s.ok())
        return s;
      if (auto s = StreamStorage(input.streams[i], info, entropy, &bound); !s.ok())
        return s;
      if (!Add(7, &bound.bits))
        return Overflow();
      auto &section = sections[info.section];
      if (auto s = section.WithMaxBits(bound.bits, [&] { return emit(i, &section); }); !s.ok())
        return s;
    }
  }
  ProfileScope assembly(ProfileStage::kAssembly);
  BitWriter file;
  if (auto s = WriteImageHeader(frame.profile.metadata, &file); !s.ok())
    return s;
  if (auto s = WriteFrameHeader(frame.profile.metadata, {}, &file); !s.ok())
    return s;
  if (auto s = WriteTocAndSections(sections, &file); !s.ok())
    return s;
  Storage<BitWriter>().swap(sections);
  return CodestreamBuffer::CopyFrom(file.padded_bytes(), out, ResourceClass::kRetainedResult);
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular stream encoding allocation failed");
} catch (const std::length_error &) {
  return Overflow();
}
} // namespace gjxl::modular_internal
