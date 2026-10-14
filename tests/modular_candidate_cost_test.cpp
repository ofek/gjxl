// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/candidate_cost.h"
#include "codestream/modular/frame_encoder.h"
#include "codestream/modular/search.h"
#include "codestream/ans_internal.h"
#include "codestream/headers_internal.h"
#include "codestream/sections.h"
#include <array>
#include <iostream>
#include <stdexcept>
#ifdef GJXL_CANDIDATE_COST_REFERENCE
#include "modular_reference.h"
#endif

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool v, const char *message) { if (!v) throw std::runtime_error(message); }
void Ok(const Status &s) { if (!s.ok()) throw std::runtime_error(std::string(s.message())); }
void TocBoundaries() {
  ModularInputProfile input;
  Ok(ResolveModularInput({513, 17}, PackedModularFormat::kRgb8, &input));
  for (uint64_t bytes : std::array<uint64_t, 9>{0, 1, 1023, 1024, 17407, 17408,
                        4211711, 4211712, uint64_t(kMaximumTocSectionSize)}) {
    for (uint64_t tail : {0ull, 1ull, 7ull}) {
      if (!bytes && tail) continue;
      const std::array<uint64_t, 3> bits{bytes * 8 - tail, 0, 7};
      ModularCandidateCost cost;
      Ok(MeasureModularSections(input.metadata, bits, &cost));
      BitWriter header;
      Ok(WriteImageHeader(input.metadata, &header));
      Ok(WriteFrameHeader(input.metadata, {}, &header));
      const size_t before = header.bits_written();
      const std::array<size_t, 3> sizes{size_t(bytes), 0, 1};
      Ok(WriteTocSizes(sizes, &header));
      Check(cost.bytes == header.padded_size() + bytes + 1 &&
                cost.frame_header_bits == before && cost.toc_bits == header.bits_written() - before &&
                cost.padding_bits == tail + 1,
            "Measured TOC threshold or padding differs from the real writer");
    }
  }
  ModularCandidateCost sentinel{.bytes = 71}, cost = sentinel;
  const std::array<uint64_t, 1> overflow{(uint64_t(kMaximumTocSectionSize) + 1) * 8};
  Check(!MeasureModularSections(input.metadata, overflow, &cost).ok() && cost == sentinel,
        "Rejected oversized TOC changed cost output");
}
size_t Fixtures() {
  size_t count = 0;
  for (auto format : {PackedModularFormat::kGray8, PackedModularFormat::kGray16,
                      PackedModularFormat::kRgb8, PackedModularFormat::kRgba8,
                      PackedModularFormat::kRgb16, PackedModularFormat::kRgba16}) {
    for (Extent2D extent : {Extent2D{1, 1}, Extent2D{33, 17}, Extent2D{257, 19}}) {
      ModularInputProfile input;
      Ok(ResolveModularInput(extent, format, &input));
      const size_t channels = input.channel_count, stride = extent.width * channels * (input.metadata.bits / 8);
      std::vector<uint8_t> samples(stride * extent.height);
      for (size_t i = 0; i < samples.size(); ++i) samples[i] = uint8_t(i * 13 + i / 7);
      PackedModularImageView source{samples, extent, stride, format, SampleByteOrder::kBigEndian};
      std::array<ModularCodingPolicy, 4> transforms;
      size_t transform_count = 0;
      // Multi-channel palette fixtures use a two-color source in the separate
      // end-to-end search suite. Squeeze boundaries and metadata live here.
      Ok(BuildTransformCandidates(extent, format, 0, &transforms, &transform_count));
      for (size_t variant = 0; variant < 2 + transform_count; ++variant) {
        ModularCodingPolicy policy;
        if (variant == 1) {
          policy.rct = channels >= 3 ? 6 : 0;
          Ok(MakeFixedWeightedTree(input.metadata.bits, &policy.tree));
        } else if (variant >= 2) policy = transforms[variant - 2];
        for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
          for (auto selection : {AnsSelectionMode::kExact, AnsSelectionMode::kEstimated}) {
            if (mode == EntropyCodingMode::kPrefix && selection == AnsSelectionMode::kEstimated) continue;
            ModularWorkflowStoragePlan storage;
            Ok(ComputeModularWorkflowStoragePlan(extent, format, mode, policy, &storage));
            ModularEncoderFrame frame;
            Ok(ModularEncoderFrame::Prepare(source, policy, &frame));
            auto metadata = frame.metadata();
            if (policy != ModularCodingPolicy{}) metadata.modular_16_bit_buffer_sufficient = false;
            ChannelShape shape;
            Ok(DescribeImage(frame.image(), &shape));
            ModularStreamPlan layout;
            Ok(BuildModularStreamPlan(storage.geometry, shape.channels(), shape.metadata, &layout));
            PreparedModularTokens tokens;
            Ok(TokenizeModular(frame, layout, policy, &tokens));
            EntropyCode model;
            EntropyCodeCost token_cost;
            PreparedEntropyClusters populations;
            Ok(OptimizeEntropyCodeAndPrepareClusters(tokens.streams,
                {.context_count = uint32_t(tokens.context_count)}, &model,
                mode == EntropyCodingMode::kPrefix ? &token_cost : nullptr, &populations));
            if (mode == EntropyCodingMode::kAns) {
              EntropyCode ans;
              Ok(OptimizeAnsEntropyCodeWithPreparedClusters(tokens.streams, model, populations,
                                                             &ans, &token_cost, nullptr, selection));
              model = std::move(ans);
            }
            ModularCandidateCost measured;
            Ok(MeasureModularCandidate(metadata, layout, tokens, model, token_cost, &measured));
            if (!count) {
              size_t fail = 0;
              for (; fail < 10000; ++fail) {
                const ModularCandidateCost sentinel{.bytes = 97};
                auto output = sentinel;
                ArmManagedHostAllocationFailureAfterForTest(fail);
                const auto status = MeasureModularCandidate(metadata, layout, tokens, model, token_cost, &output);
                const bool pending = ManagedHostAllocationFailurePendingForTest();
                DisarmManagedHostAllocationFailureForTest();
                if (status.ok()) {
                  Check(pending && output == measured, "Fault-injected measurement changed success");
                  break;
                }
                Check(!pending && output == sentinel && status.code() == StatusCode::kOutOfMemory,
                      "Measurement allocation failure changed output");
              }
              Check(fail < 10000, "Measurement fault sweep did not reach success");
            }
            CodestreamBuffer encoded;
            Ok(EncodeModularFrame(metadata, layout, std::move(tokens), mode, storage, &encoded, selection));
            Check(measured.bytes == encoded.view().size(), "Measured complete size differs from emission");
#ifdef GJXL_CANDIDATE_COST_REFERENCE
            const auto decoded = gjxl::test::modular_reference::DecodeLossless(encoded.view());
            Check(decoded.samples.size() == extent.width * extent.height * channels,
                  "Candidate decode shape differs");
            for (size_t i = 0; i < decoded.samples.size(); ++i) {
              const uint16_t expected = metadata.bits == 8 ? samples[i]
                  : (uint16_t(samples[2 * i]) << 8) | samples[2 * i + 1];
              Check(decoded.samples[i] == expected, "Candidate decoded samples differ");
            }
#endif
            ++count;
          }
        }
      }
    }
  }
  return count;
}
} // namespace
int main() try {
  TocBoundaries();
  std::cout << Fixtures() << " candidate sizes equal complete emitted files; TOC boundaries and failures passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
