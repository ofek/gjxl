// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/search.h"
#include "codestream/modular/workflow.h"
#include "codestream/modular/adaptive_gpu_workflow.h"
#if GJXL_MODULAR_PROFILE_METAL
#include "gpu/metal/metal_backend.h"
#endif
#include "io/pnm.h"
#include "modular_reference.h"
#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
namespace ref = gjxl::test::modular_reference;
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
size_t Number(std::string_view text) {
  size_t n = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), n);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !n)
    throw std::runtime_error("Expected a positive integer");
  return n;
}
void Policy(const ModularCodingPolicy &p) {
  std::cout << "{\"rct\":" << unsigned(p.rct) << ",\"transforms\":[";
  for (size_t i = 0; i < p.transforms.size; ++i) {
    const auto &t = p.transforms.entries[i];
    std::cout << (i ? "," : "") << '[' << int(t.kind) << ',' << unsigned(t.begin) << ','
              << unsigned(t.count) << ',' << t.colors << ',' << t.horizontal << ',' << t.in_place
              << ']';
  }
  std::cout << "],\"nodes\":[";
  for (size_t i = 0; i < p.tree.size; ++i) {
    const auto &n = p.tree.nodes[i];
    std::cout << (i ? "," : "") << '[' << n.property << ',' << n.split << ',' << n.left << ','
              << n.right << ',' << int(n.predictor) << ',' << n.offset << ',' << n.multiplier
              << ']';
  }
  std::cout << "],\"weighted_coefficients\":[";
  for (size_t i = 0; i < p.weighted.coefficients.size(); ++i)
    std::cout << (i ? "," : "") << unsigned(p.weighted.coefficients[i]);
  std::cout << "],\"weighted_weights\":[";
  for (size_t i = 0; i < p.weighted.weights.size(); ++i)
    std::cout << (i ? "," : "") << unsigned(p.weighted.weights[i]);
  std::cout << "]}";
}
template <size_t N>
void Stages(const std::array<const char *, N> &names, const std::array<double, N> &seconds) {
  std::cout << '{';
  for (size_t i = 0; i < N; ++i)
    std::cout << (i ? "," : "") << '"' << names[i] << "\":" << seconds[i];
  std::cout << '}';
}
} // namespace

int main(int argc, char **argv) try {
  if (argc < 6 || argc > 8)
    throw std::runtime_error(
        "gjxl_modular_search_profile INPUT OUTPUT SAMPLES LEAVES THREADS [--metal] [--no-profile]");
  bool profiling = true, metal = false;
  for (int i = 6; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if (argument == "--no-profile" && profiling) profiling = false;
    else if (argument == "--metal" && !metal) metal = true;
    else throw std::runtime_error("Unknown or duplicate profiling option");
  }
  std::unique_ptr<GpuBackend> gpu;
  double backend_setup_seconds = 0;
  if (metal) {
#if GJXL_MODULAR_PROFILE_METAL
    const auto begin = std::chrono::steady_clock::now();
    Ok(CreateEmbeddedMetalBackend({}, &gpu));
    backend_setup_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
#else
    throw std::runtime_error("This profile tool was built without Metal");
#endif
  }
  gjxl::modular_internal::ModularEncodingOptions options;
  options.entropy = EntropyCodingMode::kAns;
  options.search = true;
  options.learning = {Number(argv[3]), Number(argv[4])};
  options.tree_checkpoints = true;
  options.cpu_thread_count = Number(argv[5]);
  options.adaptive_gpu = gpu.get();
  Ok(ValidateTreeLearningOptions(options.learning));
  io::IntegerImage image;
  Ok(io::ReadPnm(argv[1], &image));
  PackedModularImageView view{image.bytes, image.extent, image.row_stride,
                              static_cast<PackedModularFormat>(image.format),
                              SampleByteOrder::kBigEndian};
  ModularWorkflowStoragePlan plan;
  Ok(ComputeModularSearchStoragePlan(image.extent, view.format, options.entropy, &plan,
                                     options.cpu_thread_count, options.learning, true));
  size_t planned_capacity = plan.working.peak_bytes;
  if (gpu) {
    AdaptiveModularWorkflowStoragePlan adaptive;
    Ok(ComputeAdaptiveModularWorkflowStoragePlan(image.extent, view.format, options, &adaptive));
    planned_capacity = adaptive.peak_bytes;
  }
  Ok(ExecutionDomain::Create({planned_capacity, options.cpu_thread_count},
                             &options.execution_domain));
  ModularProfile profile;
  std::vector<uint8_t> encoded;
  double seconds;
  const auto gpu_before = gpu ? gpu->stats() : GpuBackendStats{};
  {
    ProfileSession session(profiling ? &profile : nullptr);
    const auto begin = std::chrono::steady_clock::now();
    Ok(EncodeModularImage(view, options, &encoded));
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  }
  const auto gpu_after = gpu ? gpu->stats() : GpuBackendStats{};
  // The diagnostic winning policy can own a large managed tree. Copy it out
  // after timing, without the workflow's resource scope, then release that
  // diagnostic retention before checking that the domain is empty.
  const ModularCodingPolicy resolved = profile.resolved;
  profile.resolved = {};
  const auto resources = options.execution_domain->snapshot();
  if (resources.live_capacity_bytes || resources.reserved_unbacked_bytes ||
      resources.active_reservations || resources.active_cpu_participants ||
      resources.peak_backing_bytes > planned_capacity)
    throw std::runtime_error("Search leaked resources or exceeded its admitted storage");
  // Decoder validation and file I/O are outside the complete encoder-call timer.
  ModularInputProfile source;
  Ok(ResolveModularInput(image.extent, view.format, &source));
  const unsigned bits = source.metadata.bits, channels = source.channel_count;
  ref::IntegerImage original{image.extent.width, image.extent.height, channels, bits, {}};
  original.samples.reserve(image.extent.width * image.extent.height * channels);
  for (size_t y = 0; y < image.extent.height; ++y)
    for (size_t x = 0; x < image.extent.width * channels; ++x) {
      const size_t i = y * image.row_stride + x * (bits / 8);
      original.samples.push_back(bits == 8 ? image.bytes[i]
                                           : (uint16_t{image.bytes[i]} << 8) | image.bytes[i + 1]);
    }
  if (ref::DecodeLossless(encoded) != original)
    throw std::runtime_error("Pinned decoder round trip differs from the original samples");
  std::ofstream output(argv[2], std::ios::binary);
  output.exceptions(std::ios::badbit | std::ios::failbit);
  output.write(reinterpret_cast<const char *>(encoded.data()), encoded.size());
  output.close();
  std::cout << std::setprecision(17)
            << "{\"schema\":1,\"route\":\"" << (metal ? "hybrid-complete-search" : "cpu-complete-search")
            << "\",\"entropy\":\"ans\""
            << ",\"backend_setup_seconds\":" << backend_setup_seconds
            << ",\"gpu_submissions\":" << gpu_after.committed_submissions - gpu_before.committed_submissions
            << ",\"gpu_allocations\":" << gpu_after.successful_allocations - gpu_before.successful_allocations
            << ",\"width\":" << image.extent.width << ",\"height\":" << image.extent.height
            << ",\"channels\":" << channels << ",\"bits\":" << bits
            << ",\"maximum_samples\":" << options.learning.maximum_samples
            << ",\"maximum_leaves\":" << options.learning.maximum_leaves
            << ",\"checkpoints\":true,\"threads\":" << options.cpu_thread_count
            << ",\"profile_enabled\":" << (profiling ? "true" : "false")
            << ",\"seconds\":" << seconds << ",\"encoded_bytes\":" << encoded.size()
            << ",\"decoded_exact\":true,\"planned_capacity\":" << planned_capacity
            << ",\"peak_backing\":" << resources.peak_backing_bytes
            << ",\"peak_cpu_participants\":" << resources.peak_cpu_participants
            << ",\"candidates\":" << profile.encoded_candidates << ",\"stages\":";
  Stages(std::array{"input", "transforms", "training", "tokens", "model", "emission", "assembly",
                    "gpu_input_tokens"},
         profile.seconds);
  const auto &g = profile.adaptive_gpu;
  std::cout << ",\"adaptive_gpu\":{\"feature_seconds\":" << g.feature_seconds
            << ",\"token_seconds\":" << g.token_seconds
            << ",\"feature_preparations\":" << g.feature_preparations
            << ",\"feature_reuses\":" << g.feature_reuses << ",\"evictions\":" << g.evictions
            << ",\"encoded_candidates\":" << g.encoded_candidates
            << ",\"cpu_fallback_candidates\":" << g.cpu_fallback_candidates << '}';
  std::cout << ",\"training\":{\"stages\":";
  const auto &t = profile.training_work;
  Stages(std::array{"rct_ranking", "palette_probe", "sample_gathering", "simple_learning",
                    "entropy_preparation", "split_histograms", "property_sorting", "split_scanning",
                    "tree_control", "quantized_learning"},
         t.seconds);
  std::cout << ",\"learning_calls\":" << t.learning_calls << ",\"population\":" << t.population
            << ",\"samples\":" << t.samples << ",\"split_evaluations\":" << t.split_evaluations
            << ",\"sorted_properties\":" << t.sorted_properties
            << ",\"sorted_samples\":" << t.sorted_samples << "},\"entropy_work\":{";
  const auto &e = profile.entropy_work;
  const std::pair<const char *, uint64_t> entropy[] = {
      {"prefix_histogram_build_ns", e.prefix_histogram_build_nanoseconds},
      {"prefix_histogram_cost_ns", e.prefix_histogram_cost_nanoseconds},
      {"prefix_clustering_ns", e.prefix_clustering_nanoseconds},
      {"prefix_code_build_ns", e.prefix_code_build_nanoseconds},
      {"prefix_value_collection_ns", e.prefix_value_collection_nanoseconds},
      {"prefix_config_search_ns", e.prefix_config_search_nanoseconds},
      {"prefix_value_aggregation_ns", e.prefix_value_aggregation_nanoseconds},
      {"prefix_exact_measurement_ns", e.prefix_exact_measurement_nanoseconds},
      {"ans_prefix_validation_ns", e.ans_prefix_validation_nanoseconds},
      {"ans_value_collection_ns", e.ans_value_collection_nanoseconds},
      {"ans_value_aggregation_ns", e.ans_value_aggregation_nanoseconds},
      {"ans_prepared_value_validation_ns", e.ans_prepared_value_validation_nanoseconds},
      {"ans_uint_config_ns", e.ans_uint_config_nanoseconds},
      {"ans_histogram_build_ns", e.ans_histogram_build_nanoseconds},
      {"ans_model_build_ns", e.ans_model_build_nanoseconds},
      {"ans_token_cost_ns", e.ans_token_cost_nanoseconds},
      {"ans_uint_config_candidates", e.ans_uint_config_candidate_count},
      {"ans_histogram_candidates", e.ans_histogram_candidate_count},
      {"ans_alphabet_width_candidates", e.ans_alphabet_width_candidate_count},
      {"selection_ns", e.selection_nanoseconds}};
  bool first = true;
  for (const auto &[name, value] : entropy) {
    std::cout << (first ? "" : ",") << '"' << name << "\":" << value;
    first = false;
  }
  std::cout << "},\"policy\":";
  if (profiling)
    Policy(resolved);
  else
    std::cout << "null";
  std::cout << "}\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
