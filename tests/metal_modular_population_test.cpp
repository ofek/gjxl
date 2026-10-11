// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/modular_populations.h"
#include "codestream/modular/workflow.h"
#include "codestream/modular/adaptive_gpu_workflow.h"
#include "codec/modular/profile.h"
#include "core/resource_context.h"
#ifdef GJXL_POPULATION_REFERENCE
#include "modular_reference.h"
#endif
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace {
using namespace gjxl;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool b, const char *m) { if (!b) throw std::runtime_error(m); }
void Ok(const Status &s) { if (!s.ok()) throw std::runtime_error(std::string(s.message())); }
GpuModularPopulations &Op(GpuBackend &gpu) { auto *p = QueryGpuModularPopulations(gpu); Check(p, "Population capability absent"); return *p; }
struct Input {
  std::vector<uint32_t> values;
  std::vector<uint16_t> contexts;
  std::unique_ptr<DeviceBuffer> vb, cb;
  ModularPopulationInput request;
  Input(GpuBackend &gpu, size_t n, size_t c, unsigned pattern) : values(n + 2), contexts(n + 2) {
    values.front() = values.back() = 0xa55aa55a;
    contexts.front() = contexts.back() = 0xa55a;
    for (size_t i = 0; i < n; ++i) {
      values[i + 1] = pattern == 0 ? 4095 : pattern == 1 ? uint32_t((i * 7919 + i / 37) % 4096) :
                       i % 13 == 0 ? UINT32_MAX : i % 7 == 0 ? 4096 : uint32_t(i % 4096);
      contexts[i + 1] = uint16_t((i * 13 + i / 17) % c);
    }
    Ok(gpu.Allocate(values.size() * 4, &vb));
    Ok(gpu.Allocate(contexts.size() * 2, &cb));
    Ok(gpu.CopyHostToDevice(*vb, values.data(), values.size() * 4));
    Ok(gpu.CopyHostToDevice(*cb, contexts.data(), contexts.size() * 2));
    request = {{vb.get(), 4, n * 4}, {cb.get(), 2, n * 2}, n, c};
  }
};
size_t cases = 0, checked = 0;
void Compare(GpuBackend &gpu, size_t n, size_t c, unsigned pattern) {
  Input input(gpu, n, c, pattern);
  std::vector<uint64_t> expected(c * 4096);
  uint64_t wide = 0;
  for (size_t i = 1; i <= n; ++i)
    if (input.values[i] >= 4096) ++wide;
    else ++expected[input.contexts[i] * 4096 + input.values[i]];
  std::vector<std::byte> joined_bytes;
  std::unique_ptr<DeviceBuffer> joined;
  if (n == 2049) {
    const size_t offset = input.values.size() * 4;
    joined_bytes.resize(offset + input.contexts.size() * 2);
    std::memcpy(joined_bytes.data(), input.values.data(), offset);
    std::memcpy(joined_bytes.data() + offset, input.contexts.data(), input.contexts.size() * 2);
    Ok(gpu.Allocate(joined_bytes.size(), &joined));
    Ok(gpu.CopyHostToDevice(*joined, joined_bytes.data(), joined_bytes.size()));
    input.request.values.buffer = joined.get();
    input.request.contexts = {joined.get(), offset + 2, n * 2};
  }
  const auto before = gpu.stats();
  std::unique_ptr<PreparedGpuModularPopulations> out;
  Ok(Op(gpu).ProduceModularPopulations(input.request, &out));
  Check(out && out->backend_id() == gpu.id() && out->context_count() == c && out->token_count() == n &&
        out->wide_value_count() == wide && std::ranges::equal(out->counts(), expected),
        "GPU population differs from exact scalar counts");
  Check(gpu.stats().committed_submissions == before.committed_submissions + 1,
        "Population operation did not submit exactly once");
  auto v = input.values; auto contexts = input.contexts;
  Ok(gpu.CopyDeviceToHost(*input.vb, v.data(), v.size() * 4));
  Ok(gpu.CopyDeviceToHost(*input.cb, contexts.data(), contexts.size() * 2));
  Check(v == input.values && contexts == input.contexts, "Population producer changed input/guards");
  if (joined) {
    auto after = joined_bytes;
    Ok(gpu.CopyDeviceToHost(*joined, after.data(), after.size()));
    Check(after == joined_bytes, "Shared population input ranges or guards changed");
    joined.reset();
  }
  input.vb.reset(); input.cb.reset();
  Check(std::ranges::equal(out->counts(), expected), "Completed population depends on token lifetime");
  ++cases; checked += n;
}
void Failures(GpuBackend &gpu, GpuBackend &other) {
  Input input(gpu, 2049, 3, 1);
  ModularPopulationStoragePlan plan;
  Ok(ValidateModularPopulationInput(input.request, gpu.id(), &plan));
  for (size_t capacity : {plan.peak_bytes - 1, plan.peak_bytes}) {
    ResourceBudget budget(capacity);
    ResourceReservation reservation;
    Ok(budget.TryReserve(capacity, &reservation));
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    std::unique_ptr<PreparedGpuModularPopulations> out;
    const auto status = Op(gpu).ProduceModularPopulations(input.request, &out);
    Check(capacity < plan.peak_bytes ? status.resource_plan_exceeded() : status.ok(),
          "Population admission boundary differs");
    if (out) Check(budget.snapshot().total.live_capacity_bytes == plan.output_bytes,
                   "Population result retains scratch");
    out.reset();
    Check(!budget.snapshot().total.live_capacity_bytes && !budget.snapshot().total.pending_count &&
          budget.snapshot().peak_backing_bytes <= capacity, "Population operation leaked or exceeded admission");
  }
  std::unique_ptr<PreparedGpuModularPopulations> out;
  Ok(Op(gpu).ProduceModularPopulations(input.request, &out));
  auto *identity = out.get();
  const auto saved = std::vector(out->counts().begin(), out->counts().end());
  const auto live = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  auto intact = [&] { Check(out.get() == identity && std::ranges::equal(out->counts(), saved) &&
      DefaultResourceBudget().snapshot().total.live_capacity_bytes == live &&
      !DefaultResourceBudget().snapshot().total.pending_count, "Population failure published or leaked backing"); };
  for (int kind = 0; kind < 3; ++kind) {
    if (!kind) Ok(ArmNextMetalAllocationFailureForTest(gpu));
    else Ok(ArmNextMetalSubmissionFailureForTest(gpu, kind == 1, kind == 2));
    const auto s = Op(gpu).ProduceModularPopulations(input.request, &out);
    Check(s.code() == (!kind ? StatusCode::kOutOfMemory : kind == 1 ? StatusCode::kSubmissionFailed : StatusCode::kDeviceError),
          "Population device failure was swallowed");
    intact();
  }
  const auto before = gpu.stats();
  input.request.values.offset_bytes = 1;
  Check(!Op(gpu).ProduceModularPopulations(input.request, &out).ok(), "Misaligned GPU populations accepted");
  intact(); input.request.values.offset_bytes = 4;
  Check(!Op(other).ProduceModularPopulations(input.request, &out).ok(), "Foreign GPU populations accepted");
  intact();
  Check(before.successful_allocations == gpu.stats().successful_allocations &&
        before.committed_submissions == gpu.stats().committed_submissions, "Invalid population request reached GPU work");
  struct Impostor : DeviceBuffer { Impostor(BackendId id, size_t n) : DeviceBuffer(BackendKind::kMetal, id, n) {} } impostor(gpu.id(), input.vb->size_bytes());
  input.request.values.buffer = &impostor;
  Check(!Op(gpu).ProduceModularPopulations(input.request, &out).ok(), "Nonnative population backing accepted");
  intact(); input.request.values.buffer = input.vb.get();
  Check(before.successful_allocations == gpu.stats().successful_allocations &&
        before.committed_submissions == gpu.stats().committed_submissions, "Impostor population reached GPU work");
  uint16_t invalid = 3;
  Ok(gpu.CopyHostToDevice(*input.cb, &invalid, 2, 2));
  Check(Op(gpu).ProduceModularPopulations(input.request, &out).code() == StatusCode::kInvalidArgument,
        "Invalid context device status was ignored");
  intact();
}
void Workflow(GpuBackend &gpu, PackedModularFormat format, bool search, bool small) {
  const Extent2D extent = small ? Extent2D{33, 17} : Extent2D{1025, 257};
  ModularInputProfile description;
  Ok(ResolveModularInput(extent, format, &description));
  const size_t stride = extent.width * description.channel_count * (description.metadata.bits / 8);
  std::vector<uint8_t> pixels(stride * extent.height);
  for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = uint8_t(i * 13 + i / 37);
  PackedModularImageView input{pixels, extent, stride, format, SampleByteOrder::kBigEndian};
  for (auto entropy : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    if (search && entropy == EntropyCodingMode::kPrefix) continue;
    ModularEncodingOptions options;
    options.entropy = entropy;
    options.ans_selection = format == PackedModularFormat::kRgba8 ? AnsSelectionMode::kExact
                                                                 : AnsSelectionMode::kEstimated;
    options.search = search;
    options.learning = {8192, 32}; options.tree_checkpoints = search;
    options.search_evaluation = search ? ModularSearchEvaluation::kMeasureAll : ModularSearchEvaluation::kEncodeAll;
    options.adaptive_gpu = &gpu; options.adaptive_ans_cost = true;
    options.adaptive_tree_statistics = search;
    std::vector<uint8_t> reference, actual;
    Ok(EncodeModularImage(input, options, &reference));
    options.adaptive_populations = true;
    AdaptiveModularWorkflowStoragePlan plan;
    Ok(ComputeAdaptiveModularWorkflowStoragePlan(extent, format, options, &plan));
    Ok(ExecutionDomain::Create({plan.peak_bytes, 1}, &options.execution_domain));
    ModularProfile profile;
    { ProfileSession session(&profile); Ok(EncodeModularImage(input, options, &actual)); }
    Check(actual == reference, "GPU population workflow changed complete bytes");
    const auto &p = profile.adaptive_gpu;
    if (small) Check(p.population_cpu_small > 0 && !p.population_gpu_calls, "Small population crossover did not apply");
    else if (description.metadata.bits == 16)
      Check(p.population_cpu_wide > 0 && !profile.entropy_work.prefix_population_model_count,
            "Wide values did not select a complete CPU model");
    else Check(p.population_gpu_calls > 0 && profile.entropy_work.prefix_population_model_count > 0 &&
               !p.population_cpu_wide, "Eligible workflow did not use GPU populations");
    Check(profile.entropy_work.prefix_population_model_count + p.population_cpu_wide == p.population_gpu_calls,
          "Population success/fallback counters do not account for calls");
#ifdef GJXL_POPULATION_REFERENCE
    const auto decoded = test::modular_reference::DecodeLossless(actual);
    Check(decoded.samples.size() == extent.width * extent.height * description.channel_count,
          "Population workflow decode shape differs");
    for (size_t i = 0; i < decoded.samples.size(); ++i)
      Check(decoded.samples[i] == (description.metadata.bits == 8 ? pixels[i] :
            (uint16_t(pixels[2 * i]) << 8) | pixels[2 * i + 1]), "Population workflow decoded sample differs");
#endif
    profile.resolved = {};
    const auto resources = options.execution_domain->snapshot();
    Check(!resources.live_capacity_bytes && !resources.reserved_unbacked_bytes && !resources.active_reservations &&
          resources.peak_backing_bytes <= plan.peak_bytes, "Population workflow exceeded or leaked admission");
    std::cout << "workflow " << int(format) << " search " << search << " small " << small << " mode " << int(entropy)
              << " population calls " << p.population_gpu_calls << " wide " << p.population_cpu_wide << '\n';
  }
}
}
int main() try {
  {
    std::unique_ptr<GpuBackend> gpu, other;
    Ok(CreateEmbeddedMetalBackend({}, &gpu)); Ok(CreateEmbeddedMetalBackend({}, &other));
    for (size_t n : {1, 31, 127, 2047, 2048, 2049, 65537})
      for (unsigned pattern = 0; pattern < 3; ++pattern) Compare(*gpu, n, n == 65537 ? 256 : 3, pattern);
    Compare(*gpu, (size_t{1} << 24) + 17, 1, 0);
    Failures(*gpu, *other);
    for (auto format : {PackedModularFormat::kGray8, PackedModularFormat::kGray16,
                        PackedModularFormat::kRgb8, PackedModularFormat::kRgba8,
                        PackedModularFormat::kRgb16, PackedModularFormat::kRgba16}) Workflow(*gpu, format, false, false);
    Workflow(*gpu, PackedModularFormat::kRgb8, true, false);
    Workflow(*gpu, PackedModularFormat::kRgb8, false, true);
  }
  Check(!DefaultResourceBudget().snapshot().total.live_capacity_bytes &&
        !DefaultResourceBudget().snapshot().total.pending_count, "Population test leaked global resources");
  std::cout << cases << " native population cases, " << checked << " exact integer tokens; workflow and failures passed\n";
  return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
