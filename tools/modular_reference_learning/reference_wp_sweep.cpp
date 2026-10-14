// Copyright (c) the JPEG XL Project Authors.
// Copyright (c) 2026 Yunho Cho
// SPDX-License-Identifier: BSD-3-Clause
// Research-only complete-file WP control. Selection, transformed pixels,
// grouping, sampling and entropy options stay fixed. Refit the selected WP
// vector, then each uniform preset on eligible streams. This is not exhaustive
// over independent per-stream assignments, RCTs, palettes or tree scopes.
#include "codestream/modular/reference_workflow.h"
#include "io/pnm.h"
#include "modular_reference.h"
#include <algorithm>
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
using namespace gjxl::resource_budget_internal;
namespace reference = gjxl::test::modular_reference;
void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
struct Timer {
  std::chrono::steady_clock::time_point wall = std::chrono::steady_clock::now();
  std::clock_t cpu = std::clock();
  void Write(std::ostream &out) const {
    out << "{\"cpu\":" << double(std::clock() - cpu) / CLOCKS_PER_SEC
        << ",\"wall\":"
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - wall).count()
        << '}';
  }
};
template <class Function>
void Stage(const ResourceBudget &budget, size_t bytes, Function &&function) {
  ResourceReservation reservation;
  Ok(budget.TryReserve(std::max(size_t{1}, bytes), &reservation));
  ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
  function();
}
struct FitStats {
  size_t calls = 0, fits = 0, empty = 0, scanned = 0, selected = 0,
         distinct = 0, nodes = 0;
};
void WriteStats(std::ostream &out, const FitStats &s) {
  out << "{\"learning_calls\":" << s.calls << ",\"fits\":" << s.fits
      << ",\"empty_learning_calls\":" << s.empty << ",\"scanned_pixels\":" << s.scanned
      << ",\"selected_samples\":" << s.selected << ",\"distinct_samples\":" << s.distinct
      << ",\"tree_nodes\":" << s.nodes << '}';
}
FitStats Stats(const ReferenceWorkflowStats &s) {
  return {s.learning_calls, s.fits, s.empty_learning_calls, s.scanned_pixels,
          s.selected_samples, s.distinct_samples, s.tree_nodes};
}
void Fit(std::span<const HistogramTrainingImage> images, size_t leaves,
         const ReferenceWorkflowOptions &options, const ResourceBudget &budget,
         DecisionTree *tree, FitStats *stats) {
  HistogramSamplingOptions sampling;
  sampling.maximum_samples = options.maximum_samples;
  sampling.maximum_channel_size = kGroupDimension;
  if (options.predictors == ReferencePredictors::kAll) {
    sampling.predictor_count = kHistogramPredictors;
    for (size_t i = 0; i < kHistogramPredictors; ++i)
      sampling.predictors[i] = static_cast<Predictor>(i);
    std::swap(sampling.predictors[0], sampling.predictors[size_t(Predictor::kWeighted)]);
    std::swap(sampling.predictors[1], sampling.predictors[size_t(Predictor::kGradient)]);
  }
  HostStorageBound bound;
  Ok(ComputeHistogramSamplePreparationStorageBound(images, sampling, &bound));
  PreparedHistogramSamples prepared;
  Stage(budget, bound.peak_bytes,
        [&] { Ok(PrepareHistogramSamples(images, &prepared, sampling)); });
  ++stats->calls;
  stats->scanned += prepared.stats().scanned_pixels;
  stats->selected += prepared.stats().selected_samples;
  stats->distinct += prepared.samples().weights.size();
  if (prepared.samples().total_samples == 0) {
    tree->nodes[0].predictor = sampling.predictors[0];
    ++stats->empty;
  } else {
    auto fitting = prepared.tree_options();
    fitting.maximum_leaves = leaves;
    Ok(ComputeHistogramTreeLearningStorageBound(prepared.samples(), fitting, &bound));
    Stage(budget, bound.peak_bytes,
          [&] { Ok(LearnPreparedHistogramTree(prepared, tree, leaves)); });
    ++stats->fits;
  }
  prepared = {};
  if (tree->nodes.size() > kInlineTreeNodes && tree->nodes.size() != tree->size) {
    HostStorageBound compact;
    Check(compact.AddVector<DecisionNode>(tree->size, VectorCapacityPolicy::kFreshExact),
          "Tree storage overflow");
    Stage(budget, compact.peak_bytes, [&] { tree->nodes.resize(tree->size); });
  }
  stats->nodes += tree->size;
}
struct Candidate {
  ManagedVector<PreparedModularStreamView> streams;
  ManagedVector<DecisionTree> trees;
  FitStats stats;
  size_t changed = 0;
};
Candidate Refit(const ReferenceModularFrame &frame, int mode, size_t population,
                const ReferenceWorkflowOptions &options, const ResourceBudget &budget) {
  const auto records = frame.streams();
  Candidate candidate;
  HostStorageBound bound;
  Check(bound.AddVector<PreparedModularStreamView>(records.size(), VectorCapacityPolicy::kFreshExact) &&
            bound.AddVector<DecisionTree>(records.size(), VectorCapacityPolicy::kFreshExact),
        "Candidate storage overflow");
  Stage(budget, bound.peak_bytes, [&] {
    candidate.streams.assign(frame.view().streams.begin(), frame.view().streams.end());
    candidate.trees.resize(records.size());
  });
  for (size_t i = 0; i < records.size(); ++i) {
    auto &stream = candidate.streams[i];
    if (mode >= 0 && records[i].selection.trace.wp_count) {
      Check(records[i].selection.trace.wp_count == kSelectionWeightedPresets.size(),
            "Unexpected WP candidate domain");
      const auto &weighted = kSelectionWeightedPresets[size_t(mode)];
      candidate.changed += stream.weighted != weighted;
      stream.weighted = weighted;
    }
    if (stream.local_tree)
      stream.local_tree = &candidate.trees[i];
  }
  if (frame.stats().local_trees) {
    for (size_t i = 0; i < records.size(); ++i) {
      const auto &record = records[i];
      if (!record.channel_count)
        continue;
      HistogramTrainingImage image{record.learning_extent, record.training_channels(),
                                   record.metadata_channels, candidate.streams[i].weighted};
      size_t coded_samples = 0;
      for (auto view : candidate.streams[i].channels)
        coded_samples += view.descriptor.extent.width * view.descriptor.extent.height;
      Fit({&image, 1}, std::min(kMaximumTreeLeaves, (1024 + coded_samples + 1) / 2),
          options, budget, &candidate.trees[i], &candidate.stats);
    }
  } else {
    size_t stop = 1;
    for (const auto &record : records)
      if (record.channel_count)
        stop = std::max(stop, record.id + 1);
    ManagedVector<HistogramTrainingImage> images;
    HostStorageBound image_bound;
    Check(image_bound.AddVector<HistogramTrainingImage>(stop, VectorCapacityPolicy::kFreshExact),
          "Learning image storage overflow");
    Stage(budget, image_bound.peak_bytes, [&] { images.resize(stop); });
    for (size_t i = 0; i < records.size(); ++i)
      if (records[i].id < stop)
        images[records[i].id] = {records[i].learning_extent, records[i].training_channels(),
                                records[i].metadata_channels, candidate.streams[i].weighted};
    Fit(images, MaximumGlobalTreeLeaves(population), options, budget,
        &candidate.trees[0], &candidate.stats);
  }
  if (mode < 0) {
    for (size_t i = 0; i < records.size(); ++i)
      if (records[i].learning.stop_group)
        Check(candidate.trees[i] == records[i].tree, "Selected-WP replay changed a raw tree");
    const auto &s = candidate.stats;
    const auto expected = Stats(frame.stats());
    Check(s.calls == expected.calls && s.fits == expected.fits && s.empty == expected.empty &&
              s.scanned == expected.scanned && s.selected == expected.selected &&
              s.distinct == expected.distinct && s.nodes == expected.nodes,
          "Selected-WP replay changed learning counts");
  }
  return candidate;
}
codestream_internal::CodestreamBuffer Encode(PreparedModularStreamsView view,
    const ReferenceWorkflowOptions &options, const ResourceBudget &budget) {
  ModularWorkflowStoragePlan plan;
  Ok(ComputePreparedModularStreamsStoragePlan(view, options.entropy, &plan));
  codestream_internal::CodestreamBuffer bytes;
  Stage(budget, plan.working.peak_bytes,
        [&] { Ok(EncodePreparedModularStreams(view, options.entropy, &bytes)); });
  return bytes;
}
void Save(const std::filesystem::path &path, std::span<const uint8_t> bytes) {
  std::ofstream out(path, std::ios::binary);
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  out.close();
}
reference::IntegerImage Unpack(const io::IntegerImage &input, const ModularInputProfile &profile) {
  reference::IntegerImage result{input.extent.width, input.extent.height,
                                 uint32_t(profile.channel_count), profile.metadata.bits, {}};
  for (size_t y = 0; y < input.extent.height; ++y)
    for (size_t x = 0; x < input.extent.width * profile.channel_count; ++x) {
      const size_t offset = y * input.row_stride + x * profile.bytes_per_sample;
      result.samples.push_back(profile.bytes_per_sample == 1 ? input.bytes[offset]
          : (uint16_t(input.bytes[offset]) << 8) | input.bytes[offset + 1]);
    }
  return result;
}
} // namespace

int main(int argc, char **argv) try {
  Check(argc == 5, "reference_wp_sweep INPUT OUT auto|buffered best|all");
  const std::string buffering = argv[3], predictors = argv[4];
  Check(buffering == "auto" || buffering == "buffered", "Invalid buffering");
  Check(predictors == "best" || predictors == "all", "Invalid predictors");
  const std::filesystem::path directory = argv[2];
  Check(!std::filesystem::exists(directory), "Output directory already exists");
  io::IntegerImage input;
  Ok(io::ReadPnm(argv[1], &input));
  const PackedModularImageView packed{input.bytes, input.extent, input.row_stride,
      static_cast<PackedModularFormat>(input.format), SampleByteOrder::kBigEndian};
  ModularInputProfile profile;
  Ok(ResolveModularInput(packed.extent, packed.format, &profile));
  const auto original = Unpack(input, profile);
  ReferenceWorkflowOptions options;
  options.buffering = buffering == "auto" ? ReferenceBuffering::kAutomatic : ReferenceBuffering::kBuffered;
  options.predictors = predictors == "best" ? ReferencePredictors::kBest : ReferencePredictors::kAll;
  ResourceBudget budget(size_t{8} << 30);
  std::filesystem::create_directories(directory);
  std::ofstream out(directory / "result.json");
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out << std::setprecision(17) << std::boolalpha
      << "{\"schema\":1,\"kind\":\"uniform-wp-ablation\",\"buffering\":\"" << buffering
      << "\",\"predictors\":\"" << predictors << "\",\"baseline\":{\"prepare\":";
  ReferenceModularFrame frame;
  {
    Timer timer;
    Ok(PrepareReferenceModularFrame(packed, options, budget, &frame));
    timer.Write(out);
  }
  out << ",\"encode_prepared\":";
  codestream_internal::CodestreamBuffer baseline;
  {
    Timer timer;
    baseline = Encode(frame.view(), options, budget);
    timer.Write(out);
  }
  Check(reference::DecodeLossless(baseline.view()) == original, "Baseline decode differs");
  Save(directory / "baseline.jxl", baseline.view());
  out << ",\"bytes\":" << baseline.size() << ",\"stats\":";
  WriteStats(out, Stats(frame.stats()));
  out << "},\"local_trees\":" << frame.stats().local_trees << ",\"streams\":[";
  for (size_t i = 0; i < frame.streams().size(); ++i) {
    const auto &record = frame.streams()[i];
    if (i) out << ',';
    out << "{\"id\":" << record.id << ",\"selected_wp\":" << unsigned(record.selection.wp_mode)
        << ",\"eligible\":" << (record.selection.trace.wp_count != 0) << ",\"wp_costs\":[";
    for (size_t j = 0; j < record.selection.trace.wp_count; ++j) {
      if (j) out << ',';
      out << record.selection.trace.wp_costs[j];
    }
    out << "]}";
  }
  out << "],\"candidates\":[";
  for (int mode = -1; mode < int(kSelectionWeightedPresets.size()); ++mode) {
    const std::string name = mode < 0 ? "selected-replay" : "wp-" + std::to_string(mode);
    if (mode >= 0) out << ',';
    out << "{\"name\":\"" << name << "\",\"mode\":" << mode << ",\"prepare\":";
    Candidate candidate;
    {
      Timer timer;
      candidate = Refit(frame, mode, original.samples.size(), options, budget);
      timer.Write(out);
    }
    auto view = frame.view();
    view.streams = candidate.streams;
    if (!frame.stats().local_trees) view.global_tree = &candidate.trees[0];
    out << ",\"encode_prepared\":";
    codestream_internal::CodestreamBuffer bytes;
    {
      Timer timer;
      bytes = Encode(view, options, budget);
      timer.Write(out);
    }
    Check(reference::DecodeLossless(bytes.view()) == original, "Candidate decode differs");
    if (mode < 0)
      Check(std::ranges::equal(bytes.view(), baseline.view()), "Selected-WP replay changed bytes");
    Save(directory / (name + ".jxl"), bytes.view());
    out << ",\"bytes\":" << bytes.size() << ",\"changed_streams\":" << candidate.changed
        << ",\"stats\":";
    WriteStats(out, candidate.stats);
    out << ",\"exact_decode\":true}";
    out.flush();
    std::cout << name << ": " << bytes.size() << " bytes, " << candidate.changed
              << " changed streams, " << candidate.stats.fits << " fits; exact decode\n" << std::flush;
  }
  const auto peak = budget.snapshot();
  frame = {};
  baseline = {};
  const auto released = budget.snapshot();
  Check(!released.committed_bytes() && !released.total.backing_count &&
            !released.total.pending_count && !released.open_reservations,
        "Sweep leaked managed capacity");
  out << "],\"exact_decode\":true,\"selected_replay_exact\":true,\"peak_backing_bytes\":"
      << peak.peak_backing_bytes << ",\"peak_committed_bytes\":" << peak.peak_committed_bytes << "}\n";
  out.close();
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
