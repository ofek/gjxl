// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codec/modular/tree_learning.h"
#include "codestream/modular/gpu_tree_statistics.h"
#include "core/resource_context.h"
#include "gpu/metal/metal_backend.h"
#include <charconv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
// A deterministic production-learner probe, not an encoder throughput test.
// Sorting, CPU parent histograms, threshold enumeration, first preparation,
// uploads, waits, tree control and checkpoint publication all remain timed.
std::vector<TrainingSample> Samples(size_t n) {
  std::vector<TrainingSample> samples(n);
  uint32_t random = 0x938af102;
  for (size_t i = 0; i < n; ++i) {
    auto &s = samples[i];
    for (size_t p = 0; p < 16; ++p) {
      random = random * 1664525u + 1013904223u;
      s.properties[p] = p == 0 ? static_cast<int64_t>(i * 64 / n)
                              : static_cast<int64_t>(random % 65536);
    }
    for (size_t p = 0; p < 14; ++p)
      s.residuals[p] = p == (i * 64 / n) % 14
                           ? 0
                           : static_cast<int32_t>(1 + ((i * 7919 + p * 17) % 4095));
  }
  return samples;
}
void Run(GpuBackend &gpu, std::span<const TrainingSample> samples, size_t leaves,
         size_t minimum, unsigned repetition, const DecisionTree &expected,
         const TreeLearningCheckpoints &expected_checkpoints) {
  DecisionTree tree;
  TreeLearningCheckpoints checkpoints;
  ModularProfile profile;
  const auto before = gpu.stats();
  const auto begin = std::chrono::steady_clock::now();
  {
    ProfileSession session(&profile);
    const TreeLearningOptions options{samples.size(), leaves};
    if (minimum) {
      ModularGpuTreeStatistics evaluator(gpu, minimum);
      Ok(LearnEntropyTreeWithCostEvaluator(samples, &tree, evaluator, options,
                                           samples.size(), &checkpoints));
    } else {
      Ok(LearnEntropyTree(samples, &tree, options, samples.size(), &checkpoints));
    }
  }
  const double seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - begin).count();
  if (tree != expected || checkpoints != expected_checkpoints ||
      gpu.stats().committed_submissions - before.committed_submissions !=
          profile.adaptive_gpu.tree_gpu_groups)
    throw std::runtime_error("Benchmark changed exact tree/checkpoints or submissions");
  const auto &g = profile.adaptive_gpu;
  const auto stage = [&](TrainingProfileStage s) {
    return profile.training_work.seconds[static_cast<size_t>(s)];
  };
  std::cout << std::setprecision(12)
            << "{\"samples\":" << samples.size() << ",\"leaves\":" << leaves
            << ",\"minimum_gpu_leaf\":" << minimum
            << ",\"repetition\":" << repetition << ",\"seconds\":" << seconds
            << ",\"sorting\":" << stage(TrainingProfileStage::kPropertySorting)
            << ",\"scanning\":" << stage(TrainingProfileStage::kSplitScanning)
            << ",\"gpu_preparation\":" << g.tree_prepare_seconds
            << ",\"gpu_evaluation\":" << g.tree_evaluate_seconds
            << ",\"gpu_groups\":" << g.tree_gpu_groups
            << ",\"cpu_small_properties\":" << g.tree_cpu_small_properties
            << ",\"nodes\":" << tree.size
            << ",\"checkpoints\":" << checkpoints.size
            << ",\"exact\":true}" << std::endl;
}
} // namespace
int main(int argc, char **argv) try {
  if (argc != 3 || (std::string_view(argv[1]) != "root" &&
                    std::string_view(argv[1]) != "tree"))
    throw std::runtime_error("Usage: tree_statistics_benchmark root|tree repetitions");
  unsigned repetitions = 0;
  const std::string_view text(argv[2]);
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), repetitions);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      !repetitions || repetitions > 100)
    throw std::runtime_error("Repetitions must be in [1,100]");
  std::unique_ptr<GpuBackend> gpu;
  Ok(CreateEmbeddedMetalBackend({}, &gpu));
  const bool root = std::string_view(argv[1]) == "root";
  const std::vector<size_t> counts = root
      ? std::vector<size_t>{256, 1024, 2048, 4096, 8192, 16384, 65536, 262144}
      : std::vector<size_t>{16384, 65536, 262144};
  const std::vector<size_t> minima = root ? std::vector<size_t>{0, 16}
      : std::vector<size_t>{0, 1024, 4096, 16384};
  for (const size_t n : counts) {
    const auto samples = Samples(n);
    const size_t leaves = root ? 1 : 64;
    DecisionTree expected;
    TreeLearningCheckpoints checkpoints;
    Ok(LearnEntropyTree(samples, &expected, {n, leaves}, n, &checkpoints));
    for (unsigned repetition = 1; repetition <= repetitions; ++repetition)
      for (size_t index = 0; index < minima.size(); ++index)
        Run(*gpu, samples, leaves,
            minima[(index + repetition - 1) % minima.size()], repetition,
            expected, checkpoints);
  }
  const auto usage = gjxl::resource_budget_internal::DefaultResourceBudget().snapshot();
  if (usage.total.live_capacity_bytes || usage.total.pending_count)
    throw std::runtime_error("Benchmark leaked managed backing");
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
