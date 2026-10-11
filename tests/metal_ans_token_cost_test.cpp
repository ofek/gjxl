// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/ans_internal.h"
#include "core/resource_context.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/ans_token_cost.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::ans_gpu_internal;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;
size_t cases = 0, states = 0, tokens_checked = 0, fault_checks = 0;
unsigned widths = 0;
void Check(bool v, const char *message) {
  if (!v)
    throw std::runtime_error(message);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
GpuAnsTokenCost &Operation(GpuBackend &gpu) {
  auto *op = QueryGpuAnsTokenCost(gpu);
  Check(op, "ANS token-cost capability missing");
  return *op;
}
struct Input {
  std::vector<EntropyToken> tokens;
  std::vector<AnsTokenCostSection> sections;
  std::vector<std::vector<AnsTokenCostClusterView>> clusters;
  std::vector<AnsTokenCostModelView> models;
  std::vector<uint32_t> values;
  std::vector<uint16_t> contexts;
  std::unique_ptr<DeviceBuffer> value_buffer, context_buffer;
  AnsTokenCostInput request;
  Input(GpuBackend &gpu, std::vector<EntropyToken> source, std::span<const EntropyCode> codes,
        size_t section_count)
      : tokens(std::move(source)), clusters(codes.size()) {
    values.push_back(0xa5a55a5a);
    contexts.push_back(0xa55a);
    for (auto t : tokens) {
      values.push_back(t.value);
      contexts.push_back(static_cast<uint16_t>(t.context));
    }
    values.push_back(0x5a5aa5a5);
    contexts.push_back(0x5aa5);
    Ok(gpu.Allocate(values.size() * 4, &value_buffer));
    Ok(gpu.Allocate(contexts.size() * 2, &context_buffer));
    Ok(gpu.CopyHostToDevice(*value_buffer, values.data(), values.size() * 4));
    Ok(gpu.CopyHostToDevice(*context_buffer, contexts.data(), contexts.size() * 2));
    // Leading/trailing empty sections and irregular nonempty sections.
    sections.push_back({0, 0});
    size_t begin = 0;
    for (size_t i = 1; i <= section_count; ++i) {
      const size_t end = tokens.size() * i / section_count;
      sections.push_back({begin, end - begin});
      begin = end;
    }
    sections.push_back({begin, 0});
    for (size_t i = 0; i < codes.size(); ++i) {
      const auto &code = codes[i];
      for (size_t j = 0; j < code.ans_histograms.size(); ++j) {
        const auto &h = code.ans_histograms[j];
        const auto c = code.uint_configs[j];
        clusters[i].push_back({c.split_exponent, c.msb_in_token, c.lsb_in_token, h.frequencies,
                               h.reverse_offsets, h.reverse_map, h.reciprocal_frequencies});
      }
      models.push_back({code.context_map, clusters[i]});
    }
    request = {{value_buffer.get(), 4, tokens.size() * 4},
               {context_buffer.get(), 2, tokens.size() * 2},
               tokens.size(),
               sections,
               models};
  }
  void Unchanged(GpuBackend &gpu) {
    auto a = values;
    auto b = contexts;
    Ok(gpu.CopyDeviceToHost(*value_buffer, a.data(), a.size() * 4));
    Ok(gpu.CopyDeviceToHost(*context_buffer, b.data(), b.size() * 2));
    Check(a == values && b == contexts, "ANS cost modified input or guard backing");
  }
};
AnsTokenCostResult Reference(std::span<const EntropyToken> tokens, const EntropyCode &code) {
  AnsTokenCostResult r{32, kCostInitialState, 0};
  for (size_t i = tokens.size(); i; --i) {
    auto t = tokens[i - 1];
    const auto cluster = code.context_map[t.context];
    HybridUintToken encoded;
    Ok(EncodeHybridUint(t.value, code.uint_configs[cluster], &encoded));
    const auto &h = code.ans_histograms[cluster];
    const uint32_t f = h.frequencies[encoded.symbol];
    Check(f != 0, "Reference symbol absent");
    r.bits += encoded.extra_bit_count;
    if (uint64_t{r.final_state} >= uint64_t{f} * (uint64_t{1} << 20)) {
      r.bits += 16;
      r.final_state /= 65536;
    }
    // Ordinary division, independent of the GPU reciprocal implementation.
    r.final_state = (r.final_state / f) * 4096 +
                    h.reverse_map[h.reverse_offsets[encoded.symbol] + r.final_state % f];
  }
  uint64_t actual = 0;
  Ok(CountAnsTokenStreamBits(tokens, code, &actual));
  Check(actual == r.bits, "Independent reference differs from production CPU counter");
  return r;
}
void Compare(GpuBackend &gpu, Input &input, std::span<const EntropyCode> codes) {
  const auto before = gpu.stats();
  std::unique_ptr<PreparedGpuAnsTokenCosts> output;
  Ok(Operation(gpu).MeasureAnsTokenCosts(input.request, &output));
  Check(gpu.stats().committed_submissions == before.committed_submissions + 1,
        "ANS cost did not use exactly one device submission");
  Check(output && output->backend_id() == gpu.id() &&
            output->section_count() == input.sections.size() &&
            output->candidate_count() == codes.size() &&
            output->results().size() == codes.size() * input.sections.size(),
        "ANS result dimensions or ownership differ");
  for (size_t c = 0; c < codes.size(); ++c)
    for (size_t s = 0; s < input.sections.size(); ++s) {
      auto section = input.sections[s];
      const auto tokens = std::span(input.tokens).subspan(section.offset, section.count);
      const auto expected = Reference(tokens, codes[c]);
      Check(output->results()[c * input.sections.size() + s] == expected,
            "GPU ANS bits/final state differ");
      ++states;
      tokens_checked += tokens.size();
      if (cases % 41 == 0) {
        BitWriter writer;
        Ok(WriteAnsTokenStream(tokens, codes[c], &writer));
        uint32_t state = 0;
        std::memcpy(&state, writer.padded_bytes().data(), 4);
        Check(writer.bits_written() == expected.bits && state == expected.final_state,
              "GPU ANS count/state differs from final writer");
      }
    }
  input.Unchanged(gpu);
  ++cases;
}
EntropyCode Synthetic(HybridUintConfig config, unsigned variant, size_t context_count = 3) {
  EntropyCode c;
  c.mode = EntropyCodingMode::kAns;
  c.context_count = static_cast<uint32_t>(context_count);
  c.context_map.resize(context_count);
  for (size_t i = 0; i < context_count; ++i)
    c.context_map[i] = static_cast<uint8_t>(i % 3);
  c.uint_configs.assign(3, config);
  c.ans_histograms.resize(3);
  c.ans_log_alpha_size = 8;
  for (size_t k = 0; k < 3; ++k) {
    auto &h = c.ans_histograms[k];
    h.frequencies.resize(256);
    h.reverse_offsets.resize(256);
    h.reciprocal_frequencies.resize(256);
    h.reverse_map.resize(4096);
    uint16_t offset = 0;
    for (size_t s = 0; s < 256; ++s) {
      const uint16_t f = variant % 2 ? (s == 255 ? 3841 : 1) : 16;
      h.frequencies[s] = f;
      h.reverse_offsets[s] = offset;
      h.reciprocal_frequencies[s] = AnsFrequencyReciprocal(f);
      offset += f;
    }
    // General legal-range permutation, including different reverse maps for the
    // same normalized population. Real alias-table models are tested separately.
    for (size_t i = 0; i < 4096; ++i)
      h.reverse_map[i] = static_cast<uint16_t>((i * (1 + 2 * variant) + k * 43) % 4096);
  }
  return c;
}
void Configurations(GpuBackend &gpu) {
  for (uint8_t split = 0; split <= 15; ++split)
    for (uint8_t msb = 0; msb <= split; ++msb)
      for (uint8_t lsb = 0; lsb <= split - msb; ++lsb) {
        const HybridUintConfig config{split, msb, lsb};
        std::vector<EntropyToken> tokens;
        auto add = [&](uint32_t v) {
          HybridUintToken e;
          Ok(EncodeHybridUint(v, config, &e));
          if (e.symbol < 256)
            tokens.push_back({static_cast<uint32_t>(tokens.size() % 3), v});
        };
        for (uint32_t v = 0; v < 64; ++v)
          add(v);
        for (uint32_t b = 0; b < 32; ++b) {
          add((uint32_t{1} << b) - 1);
          add(uint32_t{1} << b);
        }
        add(UINT32_MAX);
        std::array<EntropyCode, 4> codes;
        for (unsigned i = 0; i < 4; ++i)
          codes[i] = Synthetic(config, i);
        Input input(gpu, std::move(tokens), codes, 65);
        Compare(gpu, input, codes);
      }
}
void Native(GpuBackend &gpu) {
  for (unsigned pattern = 0; pattern < 5; ++pattern) {
    std::vector<EntropyToken> tokens;
    uint32_t random = 0x915fea52;
    for (uint32_t i = 0; i < 8193; ++i) {
      random ^= random << 13;
      random ^= random >> 17;
      random ^= random << 5;
      uint32_t v = pattern == 0   ? i % 4
                   : pattern == 1 ? (i % 11 ? 0 : i % 63)
                   : pattern == 2 ? i % 97
                   : pattern == 3 ? random
                                  : 0;
      tokens.push_back({i % 3, v});
    }
    auto view = EntropyTokenStreamView::Interleaved(tokens);
    PreparedAnsEntropyCode prepared;
    Ok(PrepareRateOptimizedAnsEntropyCode(std::span(&view, 1), {.context_count = 3}, &prepared));
    std::vector<EntropyCode> codes;
    for (auto &candidate : prepared.candidates) {
      widths |= 1u << candidate.code.ans_log_alpha_size;
      codes.push_back(std::move(candidate.code));
    }
    Check(!codes.empty() && codes.size() <= 4, "Native model candidates absent");
    Input input(gpu, std::move(tokens), codes, pattern == 0 ? 1 : 193);
    Compare(gpu, input, codes);
  }
  Check((widths & 0x1e0) == 0x1e0, "Real models did not cover alphabet widths 5 through 8");
  // Maximum resident context ID and long, irregular sections.
  std::array<EntropyCode, 1> codes{Synthetic({0, 0, 0}, 0, 65536)};
  std::vector<EntropyToken> tokens(300001);
  for (size_t i = 0; i < tokens.size(); ++i)
    tokens[i] = {static_cast<uint32_t>(i % 65536), static_cast<uint32_t>(i * 7919)};
  Input input(gpu, std::move(tokens), codes, 3);
  Compare(gpu, input, codes);
}
void MixedBatch(GpuBackend &gpu) {
  const HybridUintConfig configs[] = {{0, 0, 0}, {4, 2, 0}, {4, 0, 0}, {2, 1, 1}};
  const size_t counts[] = {1, 3, 32, 64};
  std::array<EntropyCode, 4> codes;
  for (size_t i = 0; i < codes.size(); ++i) {
    codes[i] = Synthetic(configs[i], static_cast<unsigned>(i), 65536);
    auto histogram = codes[i].ans_histograms[0];
    codes[i].ans_histograms.assign(counts[i], histogram);
    codes[i].uint_configs.assign(counts[i], configs[i]);
    for (size_t c = 0; c < 65536; ++c)
      codes[i].context_map[c] = static_cast<uint8_t>(c % counts[i]);
  }
  std::vector<EntropyToken> tokens(131073);
  for (size_t i = 0; i < tokens.size(); ++i)
    tokens[i] = {static_cast<uint32_t>((i * 7919) % 65536), static_cast<uint32_t>(i * 0x1234567)};
  Input input(gpu, std::move(tokens), codes, 67);
  // The adaptive producer uses one shared buffer for both disjoint ranges.
  const size_t contexts_begin = (input.values.size() * 4 + 255) & ~size_t{255};
  std::vector<std::byte> bytes(contexts_begin + input.contexts.size() * 2, std::byte{0xa5});
  std::memcpy(bytes.data(), input.values.data(), input.values.size() * 4);
  std::memcpy(bytes.data() + contexts_begin, input.contexts.data(), input.contexts.size() * 2);
  std::unique_ptr<DeviceBuffer> joined;
  Ok(gpu.Allocate(bytes.size(), &joined));
  Ok(gpu.CopyHostToDevice(*joined, bytes.data(), bytes.size()));
  input.request.values.buffer = joined.get();
  input.request.contexts = {joined.get(), contexts_begin + 2, input.tokens.size() * 2};
  Compare(gpu, input, codes);
  auto after = bytes;
  Ok(gpu.CopyDeviceToHost(*joined, after.data(), after.size()));
  Check(after == bytes, "ANS cost changed shared input ranges or padding");
}
void FailureAndStorage(GpuBackend &gpu, GpuBackend &other) {
  std::array<EntropyCode, 1> codes{Synthetic({4, 2, 0}, 0)};
  Input input(gpu, {{0, 1}, {1, 2}, {2, 3}, {0, 4}}, codes, 3);
  auto &op = Operation(gpu);
  AnsTokenCostStoragePlan plan;
  Ok(ValidateAnsTokenCostInput(input.request, gpu.id(), &plan));
  const size_t exact = plan.output_bytes + plan.dispatch_bytes;
  for (size_t available : {exact - 1, exact, plan.peak_bytes}) {
    ResourceBudget budget(available);
    ResourceReservation reservation;
    Ok(budget.TryReserve(available, &reservation));
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    std::unique_ptr<PreparedGpuAnsTokenCosts> out;
    const auto status = op.MeasureAnsTokenCosts(input.request, &out);
    Check(available < exact ? status.resource_plan_exceeded() : status.ok(),
          "ANS cost admission boundary differs");
    if (out)
      Check(budget.snapshot().total.live_capacity_bytes == plan.output_bytes,
            "ANS result retained transient backing");
    out.reset();
    Check(budget.snapshot().total.live_capacity_bytes == 0 &&
              budget.snapshot().total.pending_count == 0 &&
              budget.snapshot().peak_backing_bytes <= available,
          "ANS cost budget leaked or exceeded admission");
  }
  std::unique_ptr<PreparedGpuAnsTokenCosts> old;
  Ok(op.MeasureAnsTokenCosts(input.request, &old));
  auto *identity = old.get();
  const auto saved = std::vector(old->results().begin(), old->results().end());
  const auto live = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  auto intact = [&] {
    Check(old.get() == identity && std::ranges::equal(old->results(), saved) &&
              DefaultResourceBudget().snapshot().total.live_capacity_bytes == live &&
              DefaultResourceBudget().snapshot().total.pending_count == 0,
          "ANS cost failure published or leaked backing");
    ++fault_checks;
  };
  for (unsigned kind = 0; kind < 3; ++kind) {
    if (kind == 0)
      Ok(ArmNextMetalAllocationFailureForTest(gpu));
    else
      Ok(ArmNextMetalSubmissionFailureForTest(gpu, kind == 1, kind == 2));
    const auto s = op.MeasureAnsTokenCosts(input.request, &old);
    Check(s.code() == (kind == 0   ? StatusCode::kOutOfMemory
                       : kind == 1 ? StatusCode::kSubmissionFailed
                                   : StatusCode::kDeviceError),
          "ANS injected device failure differs");
    intact();
  }
  auto before = gpu.stats();
  input.sections[1].offset = 1;
  Check(!op.MeasureAnsTokenCosts(input.request, &old).ok(), "Malformed ANS section accepted");
  intact();
  input.sections[1].offset = 0;
  Check(gpu.stats().successful_allocations == before.successful_allocations &&
            gpu.stats().committed_submissions == before.committed_submissions,
        "Invalid ANS input reached device work");
  Check(!Operation(other).MeasureAnsTokenCosts(input.request, &old).ok(),
        "Foreign ANS input accepted");
  intact();
  struct Impostor : DeviceBuffer {
    Impostor(BackendId id, size_t bytes) : DeviceBuffer(BackendKind::kMetal, id, bytes) {}
  } impostor(gpu.id(), input.value_buffer->size_bytes());
  input.request.values.buffer = &impostor;
  Check(!op.MeasureAnsTokenCosts(input.request, &old).ok(), "Nonnative ANS buffer accepted");
  intact();
  input.request.values.buffer = input.value_buffer.get();
  Check(gpu.stats().successful_allocations == before.successful_allocations &&
            gpu.stats().committed_submissions == before.committed_submissions,
        "Nonnative ANS buffer reached device work");
  // Well-formed model/ranges, but data-dependent context/symbol errors must be
  // reported after completion without publishing a partially written result.
  uint16_t invalid = 3;
  Ok(gpu.CopyHostToDevice(*input.context_buffer, &invalid, 2, 2));
  Check(op.MeasureAnsTokenCosts(input.request, &old).code() == StatusCode::kInvalidArgument,
        "ANS device context error absent");
  intact();
  Ok(gpu.CopyHostToDevice(*input.context_buffer, input.contexts.data(), input.contexts.size() * 2));
  auto &h = codes[0].ans_histograms[0];
  h.frequencies[0] = 0;
  h.frequencies[1] = 32;
  h.reverse_offsets[1] = 0;
  h.reciprocal_frequencies[0] = 0;
  h.reciprocal_frequencies[1] = AnsFrequencyReciprocal(32);
  uint32_t absent = 0;
  Ok(gpu.CopyHostToDevice(*input.value_buffer, &absent, 4, 4));
  Check(op.MeasureAnsTokenCosts(input.request, &old).code() == StatusCode::kInvalidArgument,
        "ANS absent-symbol error absent");
  intact();
  input.value_buffer.reset();
  input.context_buffer.reset();
  Check(std::ranges::equal(old->results(), saved),
        "Completed ANS result depends on input lifetime");
}
} // namespace
int main() try {
  {
    std::unique_ptr<GpuBackend> gpu, embedded;
    Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu));
    Ok(CreateEmbeddedMetalBackend({}, &embedded));
    std::cout << "ANS token-cost device: " << gpu->name() << std::endl;
    Configurations(*gpu);
    Native(*gpu);
    MixedBatch(*gpu);
    FailureAndStorage(*gpu, *embedded);
    Native(*embedded);
  }
  Check(DefaultResourceBudget().snapshot().total.live_capacity_bytes == 0 &&
            DefaultResourceBudget().snapshot().total.pending_count == 0,
        "ANS device tests leaked managed backing");
  std::cout << cases << " cases, " << states << " exact cost/state pairs, " << tokens_checked
            << " token recurrences, " << fault_checks << " failure-atomic checks\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
