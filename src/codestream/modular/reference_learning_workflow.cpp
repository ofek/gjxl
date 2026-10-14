// Copyright (c) the JPEG XL Project Authors.
// Copyright (c) 2026 Yunho Cho
// SPDX-License-Identifier: BSD-3-Clause
#include "codec/modular/profile.h"
#include "codestream/modular/reference_workflow_internal.h"

namespace gjxl::modular_internal::reference_workflow_internal {
namespace {
using namespace resource_budget_internal;
using namespace modular_gpu_internal;
Status Invalid() { return Status::InvalidArgument("Invalid reference learning workflow"); }
HistogramSamplingOptions SamplingOptions(const ReferenceWorkflowOptions &options) {
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
  return sampling;
}
Status Learn(std::span<const HistogramTrainingImage> images, size_t maximum_leaves,
             const ReferenceWorkflowOptions &options, const ResourceBudget &budget,
             ReferenceWorkflowStats *stats, ReferenceModularStream *record) {
  ProfileScope profile(ProfileStage::kTraining);
  const auto sampling = SamplingOptions(options);
  HostStorageBound bound;
  PreparedHistogramSamples prepared;
  {
    Seconds timer(stats->sample_preparation_seconds);
    if (auto s = ComputeHistogramSamplePreparationStorageBound(images, sampling, &bound); !s.ok())
      return s;
    if (auto s = Stage(budget, bound.peak_bytes, stats,
                       [&] { return PrepareHistogramSamples(images, &prepared, sampling); });
        !s.ok())
      return s;
  }
  Seconds fitting(stats->fitting_seconds);
  record->learning = {0, images.size(), prepared.stats(), prepared.samples().weights.size(), 1};
  ++stats->learning_calls;
  stats->scanned_pixels += prepared.stats().scanned_pixels;
  stats->selected_samples += prepared.stats().selected_samples;
  stats->distinct_samples += prepared.samples().weights.size();
  if (prepared.samples().total_samples == 0) {
    record->tree.nodes[0].predictor = sampling.predictors[0];
    ++stats->empty_learning_calls;
  } else {
    auto fitting = prepared.tree_options();
    fitting.maximum_leaves = maximum_leaves;
    if (auto s = ComputeHistogramTreeLearningStorageBound(prepared.samples(), fitting, &bound);
        !s.ok())
      return s;
    if (auto s = Stage(
            budget, bound.peak_bytes, stats,
            [&] { return LearnPreparedHistogramTree(prepared, &record->tree, maximum_leaves); });
        !s.ok())
      return s;
    ++stats->fits;
    record->learning.tree_nodes = record->tree.size;
  }
  // The fit reserves its maximum tree size. Keep only live nodes between
  // scopes, after releasing the preparation columns. No refit or pruning.
  prepared = {};
  if (record->tree.nodes.size() > kInlineTreeNodes &&
      record->tree.nodes.size() != record->tree.size) {
    HostStorageBound compact;
    if (!compact.AddVector<DecisionNode>(record->tree.size, VectorCapacityPolicy::kFreshExact))
      return Invalid();
    if (auto s = Stage(budget, compact.peak_bytes, stats,
                       [&] {
                         record->tree.nodes.resize(record->tree.size);
                         return Status::Ok();
                       });
        !s.ok())
      return s;
  }
  stats->tree_nodes += record->tree.size;
  return Status::Ok();
}

using ChannelFeatures = std::array<WeightedFeatureView, kReferenceStreamChannels>;
Status LearnLocal(ReferenceModularStream &record, const PreparedModularStreamView &stream,
                  const ReferenceWorkflowOptions &options, const ResourceBudget &budget,
                  ReferenceWorkflowStats *stats,
                  std::span<const WeightedFeatureView> features = {}) {
  if (!record.channel_count)
    return Status::Ok();
  const HistogramTrainingImage image{record.learning_extent, record.training_channels(),
                                     record.metadata_channels, stream.weighted, features};
  size_t coded_samples = 0;
  for (auto v : stream.channels)
    coded_samples += v.descriptor.extent.width * v.descriptor.extent.height;
  const size_t leaves = std::min(kMaximumTreeLeaves, (1024 + coded_samples + 1) / 2);
  return Learn({&image, 1}, leaves, options, budget, stats, &record);
}
Status LearnGlobal(std::span<ReferenceModularStream> records,
                   std::span<const PreparedModularStreamView> streams, uint64_t population,
                   const ReferenceWorkflowOptions &options, const ResourceBudget &budget,
                   ReferenceWorkflowStats *stats, std::span<const ChannelFeatures> features = {}) {
  // Preserve excluded global descriptors, canonical ID holes, and the first
  // image's population contribution, exactly as in the scalar workflow.
  size_t stop = 1;
  for (const auto &record : records)
    if (record.channel_count)
      stop = std::max(stop, record.id + 1);
  HostStorageBound bound;
  if (!bound.AddVector<HistogramTrainingImage>(stop, VectorCapacityPolicy::kFreshExact))
    return Invalid();
  ManagedVector<HistogramTrainingImage> images;
  if (auto s = Stage(budget, bound.peak_bytes, stats,
                     [&] {
                       images.resize(stop);
                       return Status::Ok();
                     });
      !s.ok())
    return s;
  for (size_t i = 0; i < records.size(); ++i) {
    const auto &record = records[i];
    if (record.id < stop)
      images[record.id] = {record.learning_extent, record.training_channels(),
                           record.metadata_channels, streams[i].weighted,
                           features.empty() ? std::span<const WeightedFeatureView>{}
                                            : std::span(features[i]).first(record.channel_count)};
  }
  return Learn(images, MaximumGlobalTreeLeaves(population), options, budget, stats, &records[0]);
}

// Batching is independent of compression decisions. In the default local-tree
// policy only one feature owner is live at a time. Explicit buffered/global
// learning retains its entire scope until the one fit, charging every batch.
constexpr size_t kFeatureBatchSamples = 8 * 1024 * 1024;
struct FeatureBatch {
  size_t first = 0, stop = 0, slices = 0, samples = 0, retained_bytes = 0;
  std::unique_ptr<PreparedGpuModularStreamFeatures> owner;
};
struct HybridLearning {
  std::span<ReferenceModularStream> records;
  std::span<PreparedModularStreamView> streams;
  const ReferenceWorkflowOptions &options;
  const ResourceBudget &budget;
  ReferenceWorkflowStats *stats;
  GpuModularStreamProcessing &op;
  GpuTokenOwners &tokens;
  ManagedVector<ChannelFeatures> features;
  ManagedVector<ModularStreamFeatureSlice> jobs;
  ManagedVector<ModularStreamTokenTree> trees;
  ManagedVector<FeatureBatch> batches;
  size_t live_samples = 0, live_bytes = 0, token_owners = 0;

  Status Initialize() {
    if (records.size() > SIZE_MAX / kReferenceStreamChannels)
      return Invalid();
    const size_t job_capacity =
        std::min(kMaximumModularFeatureBatchSlices, records.size() * kReferenceStreamChannels);
    const size_t batch_capacity = stats->local_trees ? 1 : records.size();
    HostStorageBound bound;
    if (!bound.AddVector<ChannelFeatures>(records.size(), VectorCapacityPolicy::kFreshExact) ||
        !bound.AddVector<ModularStreamFeatureSlice>(job_capacity,
                                                    VectorCapacityPolicy::kFreshExact) ||
        !bound.AddVector<ModularStreamTokenTree>(job_capacity, VectorCapacityPolicy::kFreshExact) ||
        !bound.AddVector<FeatureBatch>(batch_capacity, VectorCapacityPolicy::kFreshExact) ||
        !bound.AddVector<std::unique_ptr<PreparedGpuModularTokens>>(
            records.size(), VectorCapacityPolicy::kFreshExact))
      return Invalid();
    return Stage(budget, bound.peak_bytes, stats, [&] {
      features.resize(records.size());
      jobs.reserve(job_capacity);
      trees.reserve(job_capacity);
      batches.resize(batch_capacity);
      tokens.resize(records.size());
      return Status::Ok();
    });
  }
  Status Prepare(size_t first, FeatureBatch *batch) {
    ProfileScope profile(ProfileStage::kTraining);
    Seconds timer(stats->feature_seconds);
    jobs.clear();
    FeatureBatch result;
    result.first = first;
    bool wavefront = true;
    size_t stop = first;
    for (; stop < streams.size(); ++stop) {
      const auto &stream = streams[stop];
      size_t samples = 0, slices = 0;
      bool supported = true;
      for (const auto &channel : stream.channels) {
        size_t area;
        if (!channel.descriptor.extent.try_area(&area) || area > SIZE_MAX - samples)
          return Invalid();
        samples += area;
        slices += area != 0;
        supported &=
            channel.descriptor.extent.width <= 256 && channel.descriptor.extent.height <= 256;
      }
      if (!samples)
        continue;
      if (samples > kFeatureBatchSamples || slices > kMaximumModularFeatureBatchSlices)
        return Status::Unsupported("Selected stream exceeds histogram feature batch admission");
      if (!jobs.empty() &&
          (samples > kFeatureBatchSamples - result.samples ||
           slices > kMaximumModularFeatureBatchSlices - jobs.size() || supported != wavefront))
        break;
      wavefront = supported;
      for (size_t c = 0; c < stream.channels.size(); ++c) {
        const auto &channel = stream.channels[c];
        if (!channel.descriptor.extent.empty())
          jobs.push_back({channel, stream.weighted, static_cast<uint32_t>(records[stop].id),
                          static_cast<uint32_t>(c)});
      }
      result.samples += samples;
    }
    result.stop = stop;
    result.slices = jobs.size();
    if (jobs.empty()) {
      *batch = std::move(result);
      return Status::Ok();
    }
    ModularStreamFeatureStoragePlan plan;
    const auto limits = op.modular_stream_limits();
    if (auto s = ComputeModularStreamFeatureStoragePlan(jobs, options.feature_cpu_participants,
                                                        &plan, &limits);
        !s.ok())
      return s;
    const size_t bytes = std::max(plan.cpu_preparation.peak_bytes, plan.gpu_preparation.peak_bytes);
    if (auto s = Stage(budget, bytes, stats,
                       [&] {
                         return op.PrepareModularStreamFeatures(jobs, &result.owner,
                                                                options.feature_cpu_participants,
                                                                options.feature_producer);
                       });
        !s.ok())
      return s;
    if (!result.owner || result.owner->backend_id() != options.gpu->id() ||
        result.owner->sample_count() != result.samples ||
        result.owner->weighted_views().size() != jobs.size())
      return Status::Internal("Histogram feature producer returned an invalid owner");
    const auto views = result.owner->weighted_views();
    size_t next = 0;
    for (size_t i = first; i < stop; ++i)
      for (size_t c = 0; c < streams[i].channels.size(); ++c) {
        const auto &channel = streams[i].channels[c];
        if (channel.descriptor.extent.empty())
          continue;
        if (views[next].values.empty())
          return Status::Internal("Histogram feature map is empty");
        if (auto s = ValidateWeightedFeatureView(views[next], channel, streams[i].weighted);
            !s.ok())
          return s;
        features[i][c] = views[next++];
      }
    result.retained_bytes = plan.retained.retained_bytes;
    if (result.samples > SIZE_MAX - live_samples || result.retained_bytes > SIZE_MAX - live_bytes)
      return Invalid();
    live_samples += result.samples;
    live_bytes += result.retained_bytes;
    stats->peak_feature_samples = std::max(stats->peak_feature_samples, live_samples);
    stats->peak_feature_bytes = std::max(stats->peak_feature_bytes, live_bytes);
    ++stats->feature_batches;
    stats->feature_samples += result.samples;
    if (result.owner->producer() == WeightedFeatureBackend::kGpu) {
      ++stats->gpu_feature_batches;
      stats->gpu_feature_samples += result.samples;
    } else if (result.owner->producer() == WeightedFeatureBackend::kCpu) {
      ++stats->cpu_feature_batches;
    } else {
      return Status::Internal("Histogram feature producer returned an invalid route");
    }
    *batch = std::move(result);
    return Status::Ok();
  }
  Status Tokenize(FeatureBatch &batch) {
    if (!batch.owner)
      return Status::Ok();
    ProfileScope profile(ProfileStage::kTokens);
    Seconds timer(stats->token_seconds);
    trees.clear();
    if (stats->local_trees) {
      size_t next = 0;
      for (size_t i = batch.first; i < batch.stop; ++i) {
        size_t count = 0;
        for (const auto &channel : streams[i].channels)
          count += !channel.descriptor.extent.empty();
        if (count) {
          trees.push_back({next, count, &records[i].tree});
          next += count;
        }
      }
    } else {
      trees.push_back({0, batch.slices, &records[0].tree});
    }
    ModularStreamTokenStoragePlan plan;
    const auto limits = op.modular_stream_limits();
    if (auto s = ComputeModularStreamTokenStoragePlan(batch.samples, batch.slices, trees, &plan,
                                                      &limits);
        !s.ok())
      return s;
    std::unique_ptr<PreparedGpuModularTokens> owner;
    if (auto s = Stage(budget, plan.generation.peak_bytes, stats,
                       [&] { return op.GenerateModularStreamTokens(*batch.owner, trees, &owner); });
        !s.ok())
      return s;
    if (!owner || owner->backend_id() != options.gpu->id() ||
        owner->token_count() != batch.samples || owner->context_count() != plan.maximum_contexts)
      return Status::Internal("Histogram token producer returned an invalid owner");
    ModularTokenHostView host;
    if (auto s = owner->BorrowHostView(&host); !s.ok())
      return s;
    if (host.values.size() != batch.samples || host.contexts.size() != batch.samples ||
        !host.values.data() || !host.contexts.data())
      return Status::Internal("Histogram token producer returned an invalid host view");
    size_t next = 0;
    for (size_t i = batch.first; i < batch.stop; ++i) {
      auto &stream = streams[i];
      size_t count = 0;
      for (const auto &channel : stream.channels)
        count += channel.descriptor.extent.width * channel.descriptor.extent.height;
      if (count) {
        stream.completed_tokens = PreparedModularStreamTokenView{
            EntropyTokenStreamView::Split(host.values.subspan(next, count),
                                          host.contexts.subspan(next, count)),
            stream.channels, stats->local_trees ? &records[i].tree : &records[0].tree,
            stream.weighted, records[i].id};
        next += count;
      }
      features[i] = {};
    }
    if (next != batch.samples || token_owners >= tokens.size())
      return Status::Internal("Histogram batch token coverage mismatch");
    tokens[token_owners++] = std::move(owner);
    ++stats->token_batches;
    stats->token_samples += batch.samples;
    live_samples -= batch.samples;
    live_bytes -= batch.retained_bytes;
    batch.owner.reset();
    return Status::Ok();
  }
  Status Run(uint64_t population) {
    if (auto s = Initialize(); !s.ok())
      return s;
    size_t next = 0, count = 0;
    while (next < records.size()) {
      auto &batch = batches[stats->local_trees ? 0 : count];
      if (auto s = Prepare(next, &batch); !s.ok())
        return s;
      if (batch.stop <= next)
        return Status::Internal("Histogram batch did not advance");
      next = batch.stop;
      ++count;
      if (stats->local_trees) {
        for (size_t i = batch.first; i < batch.stop; ++i)
          if (auto s = LearnLocal(records[i], streams[i], options, budget, stats,
                                  std::span(features[i]).first(records[i].channel_count));
              !s.ok())
            return s;
        if (auto s = Tokenize(batch); !s.ok())
          return s;
      }
    }
    if (!stats->local_trees) {
      if (auto s = LearnGlobal(records, streams, population, options, budget, stats, features);
          !s.ok())
        return s;
      for (size_t i = 0; i < count; ++i)
        if (auto s = Tokenize(batches[i]); !s.ok())
          return s;
    }
    if (live_samples || live_bytes)
      return Status::Internal("Histogram features survived tokenization");
    return Status::Ok();
  }
};
} // namespace

Status LearnReferenceModularScopes(std::span<ReferenceModularStream> records,
                                   std::span<PreparedModularStreamView> streams,
                                   uint64_t population, const ReferenceWorkflowOptions &options,
                                   const ResourceBudget &budget, ReferenceWorkflowStats *stats,
                                   GpuTokenOwners *tokens) {
  if (options.gpu) {
    auto *op = QueryGpuModularStreamProcessing(*options.gpu);
    if (!op)
      return Status::Unsupported("Backend does not support histogram stream processing");
    return HybridLearning{records, streams, options, budget, stats, *op, *tokens}.Run(population);
  }
  if (stats->local_trees) {
    for (size_t i = 0; i < records.size(); ++i)
      if (auto s = LearnLocal(records[i], streams[i], options, budget, stats); !s.ok())
        return s;
    return Status::Ok();
  }
  return LearnGlobal(records, streams, population, options, budget, stats);
}
} // namespace gjxl::modular_internal::reference_workflow_internal
