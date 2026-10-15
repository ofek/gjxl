// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Complete-call comparison of the same histogram policy through explicit
// routes.
#include "codec/modular/profile.h"
#include "codestream/modular/reference_workflow.h"
#include "codestream/workflow_admission_scope.h"
#include "core/thread_budget.h"
#include "histogram_profile_json.h"
#include "io/pnm.h"
#include "modular_reference.h"
#if GJXL_HYBRID_PROFILE_METAL
#include "gpu/metal/metal_backend.h"
#endif
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
namespace reference = gjxl::test::modular_reference;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
template <typename T> void Array(std::ostream &out, const T &a) {
  out << '[';
  for (size_t i = 0; i < a.size(); ++i) {
    if (i)
      out << ',';
    out << a[i];
  }
  out << ']';
}
} // namespace
int main(int argc, char **argv) try {
  Check(argc == 6 || argc == 7, "hybrid_profile INPUT OUT cpu|shared-cpu|hybrid|auto auto|buffered "
                                "best|all [fit-off|fit-on]");
  const std::string fitting_mode = argc == 7 ? argv[6] : "fit-off";
  Check(fitting_mode == "fit-off" || fitting_mode == "fit-on", "Invalid fitting profile mode");
  const bool fitting_profile_enabled = fitting_mode == "fit-on";
  const std::string route = argv[3], buffering = argv[4], predictors = argv[5];
  Check(route == "cpu" || route == "shared-cpu" || route == "hybrid" || route == "auto",
        "Invalid route");
  Check(buffering == "auto" || buffering == "buffered", "Invalid buffering policy");
  Check(predictors == "best" || predictors == "all", "Invalid predictor policy");
  const std::filesystem::path directory = argv[2];
  Check(!std::filesystem::exists(directory), "Output directory already exists");
  io::IntegerImage input;
  Ok(io::ReadPnm(argv[1], &input));
  PackedModularImageView view{input.bytes, input.extent, input.row_stride,
                              static_cast<PackedModularFormat>(input.format),
                              SampleByteOrder::kBigEndian};
  ModularInputProfile format;
  Ok(ResolveModularInput(view.extent, view.format, &format));
  reference::IntegerImage expected{view.extent.width,
                                   view.extent.height,
                                   static_cast<uint32_t>(format.channel_count),
                                   format.metadata.bits,
                                   {}};
  expected.samples.reserve(view.extent.width * view.extent.height * format.channel_count);
  for (size_t y = 0; y < view.extent.height; ++y)
    for (size_t x = 0; x < view.extent.width * format.channel_count; ++x) {
      const size_t pos = y * view.row_stride + x * format.bytes_per_sample;
      expected.samples.push_back(format.bytes_per_sample == 1
                                     ? view.bytes[pos]
                                     : (uint16_t(view.bytes[pos]) << 8) | view.bytes[pos + 1]);
    }
  ReferenceWorkflowOptions options;
  options.buffering =
      buffering == "auto" ? ReferenceBuffering::kAutomatic : ReferenceBuffering::kBuffered;
  options.predictors =
      predictors == "best" ? ReferencePredictors::kBest : ReferencePredictors::kAll;
  std::unique_ptr<GpuBackend> gpu;
  std::string device = "CPU";
  if (route != "cpu") {
#if GJXL_HYBRID_PROFILE_METAL
    Ok(CreateEmbeddedMetalBackend({}, &gpu));
    options.gpu = gpu.get();
    device = gpu->name();
    options.feature_producer = route == "shared-cpu" ? WeightedFeatureBackend::kCpu
                               : route == "hybrid"   ? WeightedFeatureBackend::kGpu
                                                     : WeightedFeatureBackend::kAuto;
#else
    throw std::runtime_error("This build has no Metal backend");
#endif
  }
  // Same domain/CPU ceiling for every route. The ordinary histogram algorithm
  // is serial; the shared CPU feature producer may use four participants.
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({size_t{8} << 30, 4}, &domain));
  const auto &budget = codestream_internal::WorkflowAdmission::Budget(*domain);
  auto context = CurrentResourceContext();
  context.domain = domain.get();
  ResourceContextScope resource_context(context);
  thread_budget_internal::CpuExecutionScope cpu;
  Ok(cpu.Start(domain, 4));
  thread_budget_internal::EncodeScope participation(4);
  ReferenceWorkflowStats stats;
  ModularProfile profile;
  HistogramFittingProfile fitting_profile;
  // Diagnostic backing is allocated before timing and outside codec admission.
  // There is at most one fit per local group plus the global stream.
  std::vector<HistogramFitRecord> fitting_records;
  if (fitting_profile_enabled) {
    fitting_records.resize(((view.extent.width + 255) / 256) * ((view.extent.height + 255) / 256) +
                           1);
    fitting_profile.records = fitting_records;
  }
  codestream_internal::CodestreamBuffer encoded;
  double wall, process_cpu;
  {
    ProfileSession session(&profile);
    HistogramFittingProfileSession fitting_session(fitting_profile_enabled ? &fitting_profile
                                                                           : nullptr);
    const auto start = std::chrono::steady_clock::now();
    const auto cpu_start = std::clock();
    Ok(EncodeReferenceModularImage(view, options, budget, &encoded, &stats));
    wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    process_cpu = double(std::clock() - cpu_start) / CLOCKS_PER_SEC;
  }
  Check(reference::DecodeLossless(encoded.view()) == expected, "Exact independent decode differs");
  if (fitting_profile_enabled)
    Check(fitting_profile.calls == stats.fits && fitting_profile.successes == stats.fits &&
              !fitting_profile.records_dropped,
          "Histogram fitting attribution missed a fit");
  if (route == "cpu")
    Check(!stats.feature_batches && !stats.token_batches && !gpu,
          "CPU control used the hybrid route");
  else
    Check(stats.feature_samples == stats.token_samples &&
              stats.feature_samples == stats.scanned_pixels &&
              profile.training_work.reused_prediction_samples == stats.scanned_pixels &&
              profile.training_work.weighted_prediction_samples == 0,
          "Hybrid route repeated or omitted prediction features");
  const size_t bytes = encoded.size();
  std::filesystem::create_directories(directory);
  std::ofstream output(directory / "output.jxl", std::ios::binary);
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output.write(reinterpret_cast<const char *>(encoded.view().data()), bytes);
  output.close();
  const auto memory = budget.snapshot();
  encoded = {};
  const auto released = budget.snapshot();
  Check(!released.committed_bytes() && !released.total.backing_count &&
            !released.total.pending_count && !released.open_reservations,
        "Completed encode leaked managed storage");
  std::ofstream out(directory / "result.json");
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out << std::setprecision(17) << std::boolalpha << "{\"schema\":1,\"route\":" << std::quoted(route)
      << ",\"device\":" << std::quoted(device) << ",\"buffering\":" << std::quoted(buffering)
      << ",\"predictors\":" << std::quoted(predictors) << ",\"width\":" << view.extent.width
      << ",\"height\":" << view.extent.height << ",\"channels\":" << format.channel_count
      << ",\"bits\":" << unsigned(format.metadata.bits) << ",\"bytes\":" << bytes
      << ",\"exact_decode\":true,\"total\":{\"wall\":" << wall << ",\"cpu\":" << process_cpu
      << "},\"profile_wall\":";
  Array(out, profile.seconds);
  out << ",\"profile_cpu\":";
  Array(out, profile.cpu_seconds);
  out << ",\"work\":{\"local_trees\":" << stats.local_trees
      << ",\"selection_calls\":" << stats.selection_calls
      << ",\"learning_calls\":" << stats.learning_calls << ",\"fits\":" << stats.fits
      << ",\"empty_learning_calls\":" << stats.empty_learning_calls
      << ",\"scanned_pixels\":" << stats.scanned_pixels
      << ",\"selected_samples\":" << stats.selected_samples
      << ",\"distinct_samples\":" << stats.distinct_samples
      << ",\"tree_nodes\":" << stats.tree_nodes
      << "},\"routing\":{\"feature_batches\":" << stats.feature_batches
      << ",\"feature_samples\":" << stats.feature_samples
      << ",\"gpu_feature_batches\":" << stats.gpu_feature_batches
      << ",\"gpu_feature_samples\":" << stats.gpu_feature_samples
      << ",\"cpu_feature_batches\":" << stats.cpu_feature_batches
      << ",\"token_batches\":" << stats.token_batches
      << ",\"token_samples\":" << stats.token_samples << ",\"cpu_feature_peak_participants\":"
      << profile.adaptive_gpu.feature_preparation.peak_participants
      << ",\"learning_reused_features\":" << profile.training_work.reused_prediction_samples
      << ",\"learning_computed_features\":" << profile.training_work.weighted_prediction_samples
      << "},\"detail_wall\":{\"sample_preparation\":" << stats.sample_preparation_seconds
      << ",\"fitting\":" << stats.fitting_seconds << ",\"features\":" << stats.feature_seconds
      << ",\"gpu_tokens\":" << stats.token_seconds
      << "},\"memory\":{\"peak_backing_bytes\":" << memory.peak_backing_bytes
      << ",\"peak_committed_bytes\":" << memory.peak_committed_bytes
      << ",\"maximum_stage_bytes\":" << stats.maximum_stage_bytes
      << ",\"peak_feature_samples\":" << stats.peak_feature_samples
      << ",\"peak_feature_bytes\":" << stats.peak_feature_bytes << "},\"histogram_fitting\":";
  WriteHistogramProfile(out, fitting_profile, fitting_profile_enabled);
  out << "}\n";
  out.close();
  std::cout << route << ": " << wall << " s, " << bytes << " bytes, " << stats.gpu_feature_batches
            << " GPU feature batches; exact decode\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
