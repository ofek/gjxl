// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "core/cpu_execution.h"
#include "core/thread_budget.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/modular_adaptive_layout.h"
#include "io/pnm.h"
#include <charconv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::thread_budget_internal;
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
size_t Number(std::string_view text) {
  size_t n = 0;
  const auto p = std::from_chars(text.data(), text.data() + text.size(), n);
  if (p.ec != std::errc{} || p.ptr != text.data() + text.size() || !n || n > 100)
    throw std::runtime_error("Repetitions must be in 1..100");
  return n;
}
} // namespace
int main(int argc, char **argv) try {
  if (argc != 4)
    throw std::runtime_error(
        "gjxl_modular_feature_profile INPUT REPETITIONS raw|rct|squeeze|palette");
  const size_t repetitions = Number(argv[2]);
  const std::string_view transform(argv[3]);
  io::IntegerImage image;
  Ok(io::ReadPnm(argv[1], &image));
  const PackedModularImageView input{image.bytes, image.extent, image.row_stride,
                                     static_cast<PackedModularFormat>(image.format),
                                     SampleByteOrder::kBigEndian};
  ModularInputProfile source;
  Ok(ResolveModularInput(input.extent, input.format, &source));
  ModularCodingPolicy policy;
  if (transform == "rct" || transform == "squeeze")
    policy.rct = source.channel_count >= 3 ? 6 : 0;
  if (transform == "squeeze") {
    policy.transforms.size = 1;
    policy.transforms.entries[0] = {
        TransformKind::kSqueeze, 0, static_cast<uint8_t>(source.channel_count), 1, true, false};
  } else if (transform == "palette") {
    policy.transforms.size = 1;
    policy.transforms.entries[0] = {TransformKind::kPalette, 0,
                                    static_cast<uint8_t>(source.channel_count), 256};
  } else if (transform != "raw" && transform != "rct")
    throw std::runtime_error("Unknown transform");
  ModularEncoderFrame frame;
  Ok(ModularEncoderFrame::Prepare(input, policy, &frame));
  ManagedVector<AdaptiveModularTokenSlice> slices;
  Ok(BuildAdaptiveModularSlices(frame, &slices));
  AdaptiveModularLayout layout;
  Ok(ComputeAdaptiveModularLayout(frame, &layout));
  std::unique_ptr<GpuBackend> backend;
  Ok(CreateEmbeddedMetalBackend({}, &backend));
  auto *operation = QueryGpuAdaptiveModularTokenGeneration(*backend);
  if (!operation)
    throw std::runtime_error("Missing adaptive feature capability");
  // Retained reference and prepared source are outside each timed operation and
  // its local reservation. Every trial compares ALL output entries exactly.
  std::unique_ptr<PreparedGpuAdaptiveModularFeatures> reference;
  Ok(operation->PrepareAdaptiveModularFeatures(frame, slices, policy.weighted, true, &reference));
  std::span<const WeightedPredictorFeature> expected;
  Ok(reference->BorrowWeightedHostView(&expected));
  std::cout << std::setprecision(17)
            << "{\"schema\":1,\"device\":" << std::quoted(std::string(backend->name()))
            << ",\"width\":" << input.extent.width << ",\"height\":" << input.extent.height
            << ",\"samples\":" << layout.tokens << ",\"slices\":" << slices.size()
            << ",\"transform\":" << std::quoted(std::string(transform)) << ",\"runs\":[";
  bool first = true;
  // One warm-up per producer; measured rounds reverse the same four producers.
  for (size_t round = 0; round <= repetitions; ++round)
    for (size_t order = 0; order < 4; ++order) {
      const size_t variant = round % 2 ? 3 - order : order;
      const size_t threads = variant == 0 ? 1 : variant == 1 ? 2 : 4;
      const auto mode = variant == 3 ? WeightedFeatureBackend::kGpu : WeightedFeatureBackend::kCpu;
      AdaptiveModularTokenStoragePlan plan;
      Ok(ComputeAdaptiveModularTokenStoragePlan(layout.shape.channels(), layout.shape.metadata,
                                                slices.size(), 1, true, threads, &plan));
      ResourceBudget budget(plan.bridge_peak_bytes);
      ResourceReservation reservation;
      Ok(budget.Reserve(plan.bridge_peak_bytes, &reservation));
      std::shared_ptr<const ExecutionDomain> domain;
      Ok(ExecutionDomain::Create({0, threads}, &domain));
      ModularProfile profile;
      std::unique_ptr<PreparedGpuAdaptiveModularFeatures> features;
      double elapsed;
      {
        CpuExecutionScope cpu;
        Ok(cpu.Start(domain, threads));
        EncodeScope encode(threads);
        ResourceContextScope resources(
            {&reservation, ResourceClass::kPreparation, true, domain.get()});
        ProfileSession session(&profile);
        const auto start = std::chrono::steady_clock::now();
        Ok(operation->PrepareAdaptiveModularFeatures(frame, slices, policy.weighted, true,
                                                     &features, threads, mode));
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      }
      std::span<const WeightedPredictorFeature> actual;
      Ok(features->BorrowWeightedHostView(&actual));
      if (!std::ranges::equal(actual, expected))
        throw std::runtime_error("Feature map differs from serial reference");
      const auto resources = budget.snapshot();
      const auto cpu = domain->snapshot();
      const auto &p = profile.adaptive_gpu.feature_preparation;
      std::cout << (first ? "" : ",") << "{\"round\":" << round
                << ",\"warmup\":" << (round ? "false" : "true") << ",\"producer\":\""
                << (variant == 3   ? "gpu"
                    : variant == 2 ? "cpu4"
                    : variant == 1 ? "cpu2"
                                   : "cpu1")
                << "\",\"seconds\":" << elapsed
                << ",\"allocation_seconds\":" << p.allocation_seconds
                << ",\"copy_seconds\":" << p.copy_seconds
                << ",\"scratch_seconds\":" << p.scratch_seconds
                << ",\"prediction_seconds\":" << p.prediction_seconds
                << ",\"publication_seconds\":" << p.publication_seconds
                << ",\"gpu_calls\":" << p.gpu_calls
                << ",\"cpu_fallback_calls\":" << p.cpu_fallback_calls
                << ",\"feature_participants\":" << p.peak_participants
                << ",\"peak_cpu_participants\":" << cpu.peak_cpu_participants
                << ",\"planned_bytes\":" << plan.bridge_peak_bytes
                << ",\"peak_bytes\":" << resources.peak_backing_bytes << ",\"exact\":true}"
                << std::flush;
      first = false;
      features.reset();
      if (resources.peak_backing_bytes > plan.bridge_peak_bytes ||
          budget.snapshot().total.backing_count || cpu.active_cpu_participants ||
          cpu.reserved_cpu_workers || cpu.suspended_cpu_workers)
        throw std::runtime_error("Feature preparation exceeded admission or leaked resources");
    }
  std::cout << "]}\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
