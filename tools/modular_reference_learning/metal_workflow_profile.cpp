// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Qualification only. CPU/Metal-auto call the public API. Forced producers
// mirror its learned wrapper without adding a public tuning option.
#include "codec/modular/profile.h"
#include "codestream/modular/public.h"
#include "codestream/modular/reference_workflow.h"
#include "codestream/workflow_admission_scope.h"
#include "codestream/workflow_internal.h"
#include "core/thread_budget.h"
#include "io/pnm.h"
#include "modular_reference.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::thread_budget_internal;
namespace reference = gjxl::test::modular_reference;
constexpr std::array<const char *, 4> kNames = {"cpu", "metal-auto",
                                                "metal-cpu", "metal-gpu"};
void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
struct Timer {
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  std::clock_t cpu = std::clock();
  std::array<double, 2> Stop() const {
    return {
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count(),
        double(std::clock() - cpu) / CLOCKS_PER_SEC};
  }
};
template <class T> void Array(std::ostream &out, const T &array) {
  out << '[';
  for (size_t i = 0; i < array.size(); ++i)
    out << (i ? "," : "") << +array[i];
  out << ']';
}
void FeatureProfile(std::ostream &out, const FeaturePreparationProfile &p) {
  out << "{\"allocation_seconds\":" << p.allocation_seconds
      << ",\"copy_seconds\":" << p.copy_seconds
      << ",\"scratch_seconds\":" << p.scratch_seconds
      << ",\"prediction_seconds\":" << p.prediction_seconds
      << ",\"publication_seconds\":" << p.publication_seconds
      << ",\"slice_jobs\":" << p.slice_jobs << ",\"gpu_calls\":" << p.gpu_calls
      << ",\"gpu_samples\":" << p.gpu_samples
      << ",\"cpu_fallback_calls\":" << p.cpu_fallback_calls << '}';
}
void Profile(std::ostream &out, const ModularProfile &p) {
  out << "\"profile_wall\":";
  Array(out, p.seconds);
  out << ",\"profile_cpu\":";
  Array(out, p.cpu_seconds);
  const auto &t = p.training_work;
  const auto &g = p.adaptive_gpu;
  out << ",\"training\":{\"learning_calls\":" << t.learning_calls
      << ",\"population\":" << t.population << ",\"samples\":" << t.samples
      << ",\"weighted_prediction_samples\":" << t.weighted_prediction_samples
      << ",\"reused_prediction_samples\":" << t.reused_prediction_samples
      << "},\"gpu\":{\"feature_seconds\":" << g.feature_seconds
      << ",\"token_seconds\":" << g.token_seconds
      << ",\"feature_preparations\":" << g.feature_preparations
      << ",\"feature_reuses\":" << g.feature_reuses
      << ",\"feature_samples\":" << g.feature_samples
      << ",\"learning_feature_uses\":" << g.learning_feature_uses
      << ",\"encoded_candidates\":" << g.encoded_candidates
      << ",\"preparation\":";
  FeatureProfile(out, g.feature_preparation);
  out << '}';
}
void Resources(std::ostream &out, const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  const auto b =
      codestream_internal::WorkflowAdmission::Budget(domain).snapshot();
  Check(!b.committed_bytes() && !b.total.backing_count &&
            !b.total.pending_count && !b.open_reservations &&
            !b.waiting_requests && !s.active_cpu_participants &&
            !s.reserved_cpu_workers && !s.suspended_cpu_workers &&
            !s.waiting_cpu_callers,
        "Qualification leaked managed memory or CPU participation");
  out << ",\"peak_backing_bytes\":" << s.peak_backing_bytes
      << ",\"peak_committed_bytes\":" << s.peak_committed_bytes
      << ",\"peak_cpu_participants\":" << s.peak_cpu_participants
      << ",\"released\":true";
}
std::shared_ptr<const ExecutionDomain> Domain() {
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({size_t{8} << 30, 1}, &domain));
  return domain;
}
reference::IntegerImage Unpack(const io::IntegerImage &input,
                               const ModularInputProfile &p) {
  reference::IntegerImage image{input.extent.width,
                                input.extent.height,
                                uint32_t(p.channel_count),
                                p.metadata.bits,
                                {}};
  image.samples.reserve(image.width * image.height * image.channels);
  for (size_t y = 0; y < image.height; ++y)
    for (size_t x = 0; x < image.width * image.channels; ++x) {
      const size_t pos = y * input.row_stride + x * p.bytes_per_sample;
      image.samples.push_back(p.bytes_per_sample == 1
                                  ? input.bytes[pos]
                                  : (uint16_t(input.bytes[pos]) << 8) |
                                        input.bytes[pos + 1]);
    }
  return image;
}
// The same input validation, CPU scope, staged admission, frame teardown and
// publication as the public learned API; only the private producer differs.
void ForcedEncode(ModularImageView input,
                  const std::shared_ptr<const ExecutionDomain> &domain,
                  WeightedFeatureBackend producer,
                  std::vector<uint8_t> *output) {
  const PackedModularImageView packed{
      input.bytes, input.extent, input.row_stride_bytes,
      static_cast<PackedModularFormat>(input.format),
      static_cast<SampleByteOrder>(input.byte_order)};
  Ok(packed.Validate());
  const auto &budget = codestream_internal::WorkflowAdmission::Budget(*domain);
  codestream_internal::CodestreamBuffer candidate;
  {
    ResourceContextScope resources(
        {nullptr, ResourceClass::kPreparation, true, domain.get()});
    CpuExecutionScope cpu;
    Ok(cpu.Start(domain, 1));
    EncodeScope threads(1);
    ReferenceWorkflowOptions options;
    Ok(codestream_internal::BorrowProductionMetalBackend(&options.gpu));
    options.weighted_feature_backend = producer;
    Ok(EncodeReferenceModularImage(packed, options, budget, &candidate));
  }
  candidate.PublishTo(output);
}
void Encodes(const io::IntegerImage &input,
             const std::filesystem::path &directory, size_t repeats) {
  const ModularImageView view{input.bytes, input.extent, input.row_stride,
                              input.format, ModularByteOrder::kBigEndian};
  ModularInputProfile source;
  Ok(ResolveModularInput(
      input.extent, static_cast<PackedModularFormat>(input.format), &source));
  const auto original = Unpack(input, source);
  std::vector<uint8_t> expected;
  std::ofstream out(directory / "encodes.jsonl");
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out << std::setprecision(17) << std::boolalpha;
  GpuBackend *gpu = nullptr;
  auto run = [&](size_t variant, int round, size_t order) {
    auto domain = Domain();
    ModularProfile profile;
    std::vector<uint8_t> actual;
    const size_t submissions = gpu ? gpu->stats().committed_submissions : 0;
    std::array<double, 2> elapsed;
    {
      ProfileSession session(&profile);
      const Timer timer;
      if (variant < 2) {
        gjxl::ModularEncodingOptions options;
        options.backend =
            variant ? ModularBackend::kMetal : ModularBackend::kCpu;
        options.cpu_thread_count = 1;
        options.execution_domain = domain;
        ModularEncodingSummary summary;
        Ok(EncodeModularImage(view, options, &actual, &summary));
        Check(summary.backend == options.backend,
              "Public API resolved wrong backend");
      } else {
        ForcedEncode(view, domain,
                     variant == 2 ? WeightedFeatureBackend::kCpu
                                  : WeightedFeatureBackend::kGpu,
                     &actual);
      }
      elapsed = timer.Stop();
    }
    if (variant && !gpu)
      Ok(codestream_internal::BorrowProductionMetalBackend(&gpu));
    const size_t dispatched =
        gpu ? gpu->stats().committed_submissions - submissions : 0;
    if (expected.empty()) {
      expected = actual;
      std::ofstream file(directory / "output.jxl", std::ios::binary);
      file.exceptions(std::ios::failbit | std::ios::badbit);
      file.write(reinterpret_cast<const char *>(expected.data()),
                 expected.size());
    }
    Check(actual == expected, "Completed codestream differs from CPU control");
    Check(reference::DecodeLossless(actual) == original,
          "Independent decoded samples differ");
    out << "{\"schema\":1,\"variant\":\"" << kNames[variant]
        << "\",\"round\":" << round << ",\"order\":" << order
        << ",\"cold\":" << (round == -1) << ",\"warmup\":" << (round == 0)
        << ",\"seconds\":" << elapsed[0] << ",\"cpu_seconds\":" << elapsed[1]
        << ",\"bytes\":" << actual.size()
        << ",\"gpu_submissions\":" << dispatched
        << ",\"exact_bytes\":true,\"exact_decode\":true,";
    Profile(out, profile);
    profile.resolved = {};
    Resources(out, *domain);
    out << "}\n" << std::flush;
    std::cout << kNames[variant] << " round=" << round
              << " seconds=" << elapsed[0] << '\n'
              << std::flush;
  };
  // First public Metal call includes lazy backend initialization. Keep these
  // cold controls separate from warmup and the position-balanced steady runs.
  run(0, -1, 0);
  run(1, -1, 1);
  for (size_t round = 0; round <= repeats; ++round)
    for (size_t order = 0; order < 4; ++order)
      run((order + round) % 4, static_cast<int>(round), order);
}
void Features(const io::IntegerImage &input,
              const std::filesystem::path &directory, size_t repeats) {
  const PackedModularImageView view{
      input.bytes, input.extent, input.row_stride,
      static_cast<PackedModularFormat>(input.format),
      SampleByteOrder::kBigEndian};
  ResourceBudget source_budget(size_t{8} << 30);
  ReferenceModularFrame frame;
  Ok(PrepareReferenceModularFrame(view, {}, source_budget, &frame));
  std::vector<ModularStreamChannel> jobs;
  size_t eligible = 0, unsupported = 0;
  const auto streams = frame.view().streams;
  for (size_t i = 0; i < streams.size(); ++i)
    for (size_t c = 0; c < streams[i].channels.size(); ++c) {
      const auto &ch = streams[i].channels[c];
      const auto [w, h] = ch.descriptor.extent;
      if (!w || !h)
        continue;
      jobs.push_back({ch, streams[i].weighted,
                      static_cast<uint32_t>(frame.streams()[i].id),
                      static_cast<uint32_t>(c), 0});
      (w <= 256 && h <= 256 ? eligible : unsupported) += w * h;
    }
  GpuBackend *gpu;
  Ok(codestream_internal::BorrowProductionMetalBackend(&gpu));
  auto *op = QueryGpuModularStreamGeneration(*gpu);
  Check(op, "Missing Metal stream operation");
  ModularStreamBatchPlan plan;
  const auto limits = op->modular_stream_limits();
  Ok(ComputeModularStreamBatchPlan(jobs, 1, &plan, &limits));
  // The independent serial producer checks every field, including property 15.
  // Its map, selected source channels and descriptors are outside trial
  // budgets.
  std::vector<WeightedPredictorFeature> expected(plan.tokens.token_count);
  size_t offset = 0;
  for (const auto &job : jobs) {
    const auto [w, h] = job.channel.descriptor.extent;
    Ok(PrepareWeightedPredictorFeatures(
        job.channel, job.weighted, std::span(expected).subspan(offset, w * h)));
    offset += w * h;
  }
  std::ofstream out(directory / "features.jsonl");
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out << std::setprecision(17) << std::boolalpha;
  constexpr std::array<WeightedFeatureBackend, 3> modes = {
      WeightedFeatureBackend::kCpu, WeightedFeatureBackend::kGpu,
      WeightedFeatureBackend::kAuto};
  constexpr std::array<const char *, 3> names = {"cpu", "gpu", "auto"};
  for (size_t round = 0; round <= repeats; ++round)
    for (size_t order = 0; order < modes.size(); ++order) {
      const size_t v = (round + order) % modes.size();
      auto domain = Domain();
      const auto &budget =
          codestream_internal::WorkflowAdmission::Budget(*domain);
      ModularProfile profile;
      std::unique_ptr<PreparedGpuModularStreamFeatures> features;
      std::array<double, 2> elapsed;
      {
        ResourceReservation reservation;
        Ok(budget.TryReserve(plan.preparation.peak_bytes, &reservation));
        ResourceContextScope resources(
            {&reservation, ResourceClass::kPreparation, true, domain.get()});
        CpuExecutionScope cpu;
        Ok(cpu.Start(domain, 1));
        EncodeScope threads(1);
        ProfileSession session(&profile);
        const Timer timer;
        Ok(op->PrepareModularStreamFeatures(jobs, &features, modes[v]));
        elapsed = timer.Stop();
      }
      std::span<const WeightedPredictorFeature> actual;
      Ok(features->BorrowWeightedHostView(&actual));
      Check(std::ranges::equal(expected, actual),
            "Prepared weighted features differ");
      const auto &p = profile.adaptive_gpu.feature_preparation;
      if (modes[v] == WeightedFeatureBackend::kGpu)
        Check(p.gpu_samples == eligible,
              "Forced GPU did not process every eligible sample");
      if (modes[v] == WeightedFeatureBackend::kCpu)
        Check(p.gpu_samples == 0,
              "Forced CPU unexpectedly dispatched features");
      Check(budget.snapshot().peak_backing_bytes <= plan.preparation.peak_bytes,
            "Feature operation exceeded planned backing");
      features.reset();
      out << "{\"schema\":1,\"variant\":\"" << names[v]
          << "\",\"round\":" << round << ",\"order\":" << order
          << ",\"warmup\":" << (round == 0) << ",\"seconds\":" << elapsed[0]
          << ",\"cpu_seconds\":" << elapsed[1]
          << ",\"samples\":" << expected.size() << ",\"jobs\":" << jobs.size()
          << ",\"eligible_samples\":" << eligible
          << ",\"unsupported_samples\":" << unsupported
          << ",\"planned_bytes\":" << plan.preparation.peak_bytes
          << ",\"exact_features\":true,\"preparation\":";
      FeatureProfile(out, p);
      Resources(out, *domain);
      out << "}\n" << std::flush;
    }
  frame = {};
  Check(!source_budget.snapshot().committed_bytes(),
        "Prepared source leaked resources");
  std::cout << "features samples=" << expected.size() << " jobs=" << jobs.size()
            << " eligible=" << eligible << " unsupported=" << unsupported
            << '\n';
}
} // namespace
int main(int argc, char **argv) try {
  Check(argc == 5,
        "metal_workflow_profile INPUT OUT REPETITIONS encode|features");
  const std::filesystem::path directory = argv[2];
  Check(!std::filesystem::exists(directory), "Output directory already exists");
  size_t repeats = 0;
  const std::string_view count(argv[3]), mode(argv[4]);
  const auto parsed =
      std::from_chars(count.data(), count.data() + count.size(), repeats);
  Check(parsed.ec == std::errc{} && parsed.ptr == count.data() + count.size() &&
            repeats && repeats <= 100,
        "Repetitions must be in 1..100");
  Check(mode == "encode" || mode == "features", "Unknown qualification mode");
  std::filesystem::create_directories(directory);
  io::IntegerImage input;
  Ok(io::ReadPnm(argv[1], &input));
  if (mode == "encode")
    Encodes(input, directory, repeats);
  else
    Features(input, directory, repeats);
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
