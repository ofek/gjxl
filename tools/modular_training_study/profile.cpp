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
#include <sstream>

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
void Policy(const ModularCodingPolicy &p, std::ostream &out = std::cout) {
  out << "{\"rct\":" << unsigned(p.rct) << ",\"transforms\":[";
  for (size_t i = 0; i < p.transforms.size; ++i) {
    const auto &t = p.transforms.entries[i];
    out << (i ? "," : "") << '[' << int(t.kind) << ',' << unsigned(t.begin) << ','
              << unsigned(t.count) << ',' << t.colors << ',' << t.horizontal << ',' << t.in_place
              << ']';
  }
  out << "],\"nodes\":[";
  for (size_t i = 0; i < p.tree.size; ++i) {
    const auto &n = p.tree.nodes[i];
    out << (i ? "," : "") << '[' << n.property << ',' << n.split << ',' << n.left << ','
              << n.right << ',' << int(n.predictor) << ',' << n.offset << ',' << n.multiplier
              << ']';
  }
  out << "],\"weighted_coefficients\":[";
  for (size_t i = 0; i < p.weighted.coefficients.size(); ++i)
    out << (i ? "," : "") << unsigned(p.weighted.coefficients[i]);
  out << "],\"weighted_weights\":[";
  for (size_t i = 0; i < p.weighted.weights.size(); ++i)
    out << (i ? "," : "") << unsigned(p.weighted.weights[i]);
  out << "]}";
}
class CandidateLedger final : public ModularCandidateObserver {
public:
  std::vector<std::string> rows;
  Status Candidate(const ModularCodingPolicy &policy, const ModularCandidateRecord &record) override {
    std::ostringstream out;
    const auto &c = record.cost;
    out << std::setprecision(17) << "{\"index\":" << record.index
        << ",\"checkpoint\":" << (record.checkpoint ? "true" : "false")
        << ",\"improved\":" << (record.improved ? "true" : "false")
        << ",\"exact\":" << (record.exact ? "true" : "false")
        << ",\"estimated_bits\":" << record.estimated_bits
        << ",\"sampled_tokens\":" << record.sampled_tokens
        << ",\"seconds\":" << record.seconds << ",\"bytes\":" << c.bytes
        << ",\"tokens\":" << c.tokens << ",\"contexts\":" << c.contexts
        << ",\"clusters\":" << c.clusters << ",\"payload_bits\":" << c.payload_bits
        << ",\"tree_bits\":" << c.tree_bits << ",\"model_bits\":" << c.model_bits
        << ",\"stream_header_bits\":" << c.stream_header_bits
        << ",\"frame_header_bits\":" << c.frame_header_bits << ",\"toc_bits\":" << c.toc_bits
        << ",\"padding_bits\":" << c.padding_bits << ",\"policy\":";
    Policy(policy, out);
    out << '}';
    rows.push_back(out.str());
    return Status::Ok();
  }
};
template <size_t N>
void Stages(const std::array<const char *, N> &names, const std::array<double, N> &seconds) {
  std::cout << '{';
  for (size_t i = 0; i < N; ++i)
    std::cout << (i ? "," : "") << '"' << names[i] << "\":" << seconds[i];
  std::cout << '}';
}
} // namespace

int main(int argc, char **argv) try {
  if (argc < 6 || argc > 13)
    throw std::runtime_error(
        "gjxl_modular_search_profile INPUT OUTPUT SAMPLES LEAVES THREADS [--metal|--metal-ans|--metal-learn] [--estimated-ans] [--measure-all|--shortlist] [--candidate-ledger] [--metal-populations] [--no-profile]");
  bool profiling = true, metal = false, metal_ans = false, metal_learn = false, estimated = false;
  bool measure = false, shortlist = false, ledger = false, populations = false;
  CandidateLedger candidates;
  for (int i = 6; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if (argument == "--no-profile" && profiling) profiling = false;
    else if (argument == "--metal" && !metal) metal = true;
    else if (argument == "--metal-ans" && !metal) { metal = true; metal_ans = true; }
    else if (argument == "--metal-learn" && !metal) { metal = true; metal_ans = true; metal_learn = true; }
    else if (argument == "--estimated-ans" && !estimated) estimated = true;
    else if (argument == "--measure-all" && !measure && !shortlist) measure = true;
    else if (argument == "--shortlist" && !measure && !shortlist) shortlist = true;
    else if (argument == "--metal-populations" && !populations) populations = true;
    else if (argument == "--candidate-ledger" && !ledger) ledger = true;
    else throw std::runtime_error("Unknown or duplicate profiling option");
  }
  if (populations && !metal) throw std::runtime_error("--metal-populations requires a Metal route");
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
  options.adaptive_ans_cost = metal_ans;
  options.adaptive_tree_statistics = metal_learn;
  options.adaptive_populations = populations;
  options.ans_selection = estimated ? AnsSelectionMode::kEstimated : AnsSelectionMode::kExact;
  options.search_evaluation = shortlist ? ModularSearchEvaluation::kShortlist : measure ? ModularSearchEvaluation::kMeasureAll : ModularSearchEvaluation::kEncodeAll;
  options.candidate_observer = ledger ? &candidates : nullptr;
  Ok(ValidateTreeLearningOptions(options.learning));
  io::IntegerImage image;
  Ok(io::ReadPnm(argv[1], &image));
  PackedModularImageView view{image.bytes, image.extent, image.row_stride,
                              static_cast<PackedModularFormat>(image.format),
                              SampleByteOrder::kBigEndian};
  ModularWorkflowStoragePlan plan;
  Ok(ComputeModularSearchStoragePlan(image.extent, view.format, options.entropy, &plan,
                                     options.cpu_thread_count, options.learning, true, measure || shortlist || ledger));
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
            << "{\"schema\":1,\"route\":\"" << (metal_learn ? "hybrid-learning-complete-search" : metal_ans ? "hybrid-ans-complete-search" : metal ? "hybrid-complete-search" : "cpu-complete-search")
            << "\",\"entropy\":\"ans\""
            << ",\"ans_selection\":\"" << (estimated ? "estimated" : "exact") << '"'
            << ",\"search_evaluation\":\"" << (shortlist ? "shortlist" : measure ? "measure-all" : "encode-all") << '"'
            << ",\"adaptive_populations\":" << (populations ? "true" : "false")
            << ",\"candidate_ledger_enabled\":" << (ledger ? "true" : "false")
            << ",\"candidate_emissions\":" << profile.candidate_emissions
            << ",\"screened_candidates\":" << profile.screened_candidates
            << ",\"exactly_measured_candidates\":" << profile.exactly_measured_candidates
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
            << ",\"cpu_fallback_candidates\":" << g.cpu_fallback_candidates
            << ",\"ans_gpu_groups\":" << g.ans_gpu_groups
            << ",\"ans_gpu_states\":" << g.ans_gpu_states
            << ",\"ans_cpu_fallback_groups\":" << g.ans_cpu_fallback_groups
            << ",\"ans_cpu_small_groups\":" << g.ans_cpu_small_groups
            << ",\"tree_prepare_seconds\":" << g.tree_prepare_seconds
            << ",\"tree_evaluate_seconds\":" << g.tree_evaluate_seconds
            << ",\"tree_gpu_preparations\":" << g.tree_gpu_preparations
            << ",\"tree_gpu_groups\":" << g.tree_gpu_groups
            << ",\"tree_gpu_samples\":" << g.tree_gpu_samples
            << ",\"tree_cpu_fallback_properties\":" << g.tree_cpu_fallback_properties
            << ",\"tree_cpu_small_properties\":" << g.tree_cpu_small_properties
            << ",\"tree_cpu_unsupported_learning_calls\":" << g.tree_cpu_unsupported_learning_calls
            << ",\"population_seconds\":" << g.population_seconds
            << ",\"population_gpu_calls\":" << g.population_gpu_calls
            << ",\"population_tokens\":" << g.population_tokens
            << ",\"population_cpu_small\":" << g.population_cpu_small
            << ",\"population_cpu_unsupported\":" << g.population_cpu_unsupported
            << ",\"population_cpu_wide\":" << g.population_cpu_wide
            << ",\"population_screen_skips\":" << g.population_screen_skips << '}';
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
      {"prefix_section_measurements", e.prefix_section_measurement_count},
      {"prefix_population_merge_ns", e.prefix_population_merge_nanoseconds},
      {"prefix_population_models", e.prefix_population_model_count},
      {"prefix_histogram_tokens", e.prefix_histogram_token_count},
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
  std::cout << "},\"candidate_ledger\":[";
  for (size_t i = 0; i < candidates.rows.size(); ++i)
    std::cout << (i ? "," : "") << candidates.rows[i];
  std::cout << "],\"policy\":";
  if (profiling)
    Policy(resolved);
  else
    std::cout << "null";
  std::cout << "}\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
