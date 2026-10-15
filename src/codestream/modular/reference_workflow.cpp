// Copyright (c) the JPEG XL Project Authors.
// Copyright (c) 2026 Yunho Cho
// SPDX-License-Identifier: BSD-3-Clause
// Lossless scope policy follows pinned enc_frame.cc and enc_modular.cc.
#include "codestream/modular/reference_workflow.h"
#include "codec/modular/profile.h"
#include <algorithm>
#include <stdexcept>

namespace gjxl::modular_internal {
namespace {
using namespace resource_budget_internal;
Status Invalid() {
  return Status::InvalidArgument("Invalid reference Modular workflow");
}
template <typename Function>
Status Stage(const ResourceBudget &budget, size_t bytes,
             ReferenceWorkflowStats *stats, Function &&function) {
  ResourceReservation reservation;
  if (auto s = budget.TryReserve(std::max(size_t{1}, bytes), &reservation);
      !s.ok())
    return s;
  stats->maximum_stage_bytes = std::max(stats->maximum_stage_bytes, bytes);
  auto context = CurrentResourceContext();
  context.reservation = &reservation;
  context.resource_class = ResourceClass::kPreparation;
  ResourceContextScope scope(context);
  return function();
}
Status Fill(const ModularImage &image, ReferenceModularStream *record) {
  if (image.channel_count() > record->channels.size())
    return Invalid();
  record->channel_count = image.channel_count();
  record->metadata_channels = image.metadata_channels();
  for (size_t c = 0; c < image.channel_count(); ++c)
    record->channels[c] = image.view(c);
  return Status::Ok();
}
Status Select(ReferenceSelectionOptions options, const ResourceBudget &budget,
              ReferenceWorkflowStats *stats, ReferenceModularStream *record) {
  ProfileScope profile(ProfileStage::kTransforms);
  HostStorageBound bound;
  if (auto s = ComputeReferenceSelectionStorageBound(
          record->training_channels(), record->metadata_channels, options,
          &bound);
      !s.ok())
    return s;
  if (auto s = Stage(budget, bound.peak_bytes, stats,
                     [&] {
                       return SelectReferenceModularEncoding(
                           record->training_channels(),
                           record->metadata_channels, options,
                           &record->selection);
                     });
      !s.ok())
    return s;
  ++stats->selection_calls;
  if (record->selection.image.channel_count())
    return Fill(record->selection.image, record);
  return Status::Ok();
}
HistogramSamplingOptions
SamplingOptions(const ReferenceWorkflowOptions &options) {
  HistogramSamplingOptions sampling;
  sampling.maximum_samples = options.maximum_samples;
  sampling.maximum_channel_size = kGroupDimension;
  if (options.predictors == ReferencePredictors::kAll) {
    sampling.predictor_count = kHistogramPredictors;
    for (size_t i = 0; i < kHistogramPredictors; ++i)
      sampling.predictors[i] = static_cast<Predictor>(i);
    std::swap(sampling.predictors[0],
              sampling.predictors[size_t(Predictor::kWeighted)]);
    std::swap(sampling.predictors[1],
              sampling.predictors[size_t(Predictor::kGradient)]);
  }
  return sampling;
}
Status Learn(std::span<const HistogramTrainingImage> images,
             size_t maximum_leaves, const ReferenceWorkflowOptions &options,
             const ResourceBudget &budget, ReferenceWorkflowStats *stats,
             ReferenceModularStream *record) {
  ProfileScope profile(ProfileStage::kTraining);
  const auto sampling = SamplingOptions(options);
  HostStorageBound bound;
  if (auto s = ComputeHistogramSamplePreparationStorageBound(images, sampling,
                                                             &bound);
      !s.ok())
    return s;
  PreparedHistogramSamples prepared;
  if (auto s = Stage(
          budget, bound.peak_bytes, stats,
          [&] { return PrepareHistogramSamples(images, &prepared, sampling); });
      !s.ok())
    return s;
  record->learning = {0, images.size(), prepared.stats(),
                      prepared.samples().weights.size(), 1};
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
    if (auto s = ComputeHistogramTreeLearningStorageBound(prepared.samples(),
                                                          fitting, &bound);
        !s.ok())
      return s;
    if (auto s = Stage(budget, bound.peak_bytes, stats,
                       [&] {
                         return LearnPreparedHistogramTree(
                             prepared, &record->tree, maximum_leaves);
                       });
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
    if (!compact.AddVector<DecisionNode>(record->tree.size,
                                         VectorCapacityPolicy::kFreshExact))
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
} // namespace

PreparedModularStreamsView ReferenceModularFrame::view() const {
  return {extent_, format_, records_.empty() || stats_.local_trees ? nullptr : &records_[0].tree,
          streams_, token_views_};
}
Status PrepareReferenceModularFrame(PackedModularImageView input,
                                    const ReferenceWorkflowOptions &options,
                                    const ResourceBudget &budget,
                                    ReferenceModularFrame *out) try {
  if (!out ||
      (options.buffering != ReferenceBuffering::kAutomatic &&
       options.buffering != ReferenceBuffering::kBuffered) ||
      (options.predictors != ReferencePredictors::kBest &&
       options.predictors != ReferencePredictors::kAll) ||
      !options.maximum_samples || options.maximum_samples > kHistogramMaximumSamples ||
      (options.weighted_feature_backend != WeightedFeatureBackend::kAuto &&
       options.weighted_feature_backend != WeightedFeatureBackend::kCpu &&
       options.weighted_feature_backend != WeightedFeatureBackend::kGpu))
    return Invalid();
  auto *gpu =
      options.gpu ? modular_gpu_internal::QueryGpuModularStreamGeneration(*options.gpu) : nullptr;
  if (options.gpu && !gpu)
    return Status::Unsupported("Backend lacks Modular stream generation");
  if (auto s = ValidatePreparedModularEntropyOptions(options.entropy); !s.ok())
    return s;
  if (auto s = input.Validate(); !s.ok())
    return s;
  ModularInputProfile source;
  if (auto s = ResolveModularInput(input.extent, input.format, &source);
      !s.ok())
    return s;
  const uint64_t population =
      uint64_t{input.extent.width} * input.extent.height * source.channel_count;
  if (population > kHistogramMaximumPopulation)
    return Status::Unsupported(
        "Reference workflow source population exceeds private admission");
  ModularFrameGeometry geometry;
  if (auto s = ModularFrameGeometry::Create(input.extent, &geometry); !s.ok())
    return s;
  size_t final_id;
  if (auto s = ModularStreamId(geometry, StreamRole::kGroup,
                               geometry.group_count() - 1, &final_id);
      !s.ok())
    return s;
  if (final_id >= 65535)
    return Status::Unsupported(
        "Reference workflow group IDs exceed learning admission");
  ReferenceModularFrame result;
  result.extent_ = input.extent;
  result.format_ = input.format;
  auto &stats = result.stats_;
  stats.local_trees = options.buffering == ReferenceBuffering::kAutomatic &&
                      geometry.group_count() > 8;
  HostStorageBound bound;
  const size_t count = 1 + geometry.dc_group_count() + geometry.group_count();
  if (!bound.AddVector<ReferenceModularStream>(
          count, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<PreparedModularStreamView>(
          count, VectorCapacityPolicy::kFreshExact))
    return Invalid();
  if (auto s = Stage(budget, bound.peak_bytes, &stats,
                     [&] {
                       result.records_.resize(count);
                       result.streams_.resize(count);
                       return Status::Ok();
                     });
      !s.ok())
    return s;
  if (auto s = ComputeModularImageStorageBound(source.channels(), 0, &bound);
      !s.ok())
    return s;
  if (auto s = Stage(
          budget, bound.peak_bytes, &stats,
          [&] { return ModularEncoderFrame::Prepare(input, &result.source_); });
      !s.ok())
    return s;
  auto &global = result.records_[0];
  if (auto s = Fill(result.source_.image(), &global); !s.ok())
    return s;
  global.learning_extent = input.extent;
  if (!stats.local_trees) {
    const bool single = geometry.group_count() == 1;
    if (auto s = Select({ReferencePaletteStage::kGlobal, source.metadata.bits,
                         1024, single, single},
                        budget, &stats, &global);
        !s.ok())
      return s;
  }
  const ModularImage *base = &result.source_.image();
  if (global.selection.image.channel_count()) {
    base = &global.selection.image;
    result
        .source_ = {}; // All future group views now borrow the selected image.
  }
  ChannelShape shape;
  if (auto s = DescribeImage(*base, &shape); !s.ok())
    return s;
  ModularStreamStoragePlan placement;
  if (auto s = ComputeModularStreamStoragePlan(geometry, shape.channels(),
                                               shape.metadata, &placement);
      !s.ok())
    return s;
  ModularStreamPlan plan;
  if (auto s = Stage(budget, placement.owned.peak_bytes, &stats,
                     [&] {
                       return BuildModularStreamPlan(geometry, shape.channels(),
                                                     shape.metadata, &plan);
                     });
      !s.ok())
    return s;
  if (stats.local_trees) {
    // The global header is produced from the first streaming DC chunk. Its
    // normal planes remain present as excluded learning descriptors.
    global.learning_extent = {std::min(input.extent.width, size_t{2048}),
                              std::min(input.extent.height, size_t{2048})};
    for (size_t c = 0; c < global.channel_count; ++c) {
      ModularChannelView view;
      if (auto s = BorrowChannelSlice(base->view(c),
                                      {0, 0, global.learning_extent}, &view);
          !s.ok())
        return s;
      global.channels[c] = view;
    }
  }
  for (size_t i = 0; i < plan.streams.size(); ++i) {
    const auto &entry = plan.streams[i];
    auto &record = result.records_[i];
    record.id = entry.id;
    if (i != 0) {
      record.learning_extent = entry.role == StreamRole::kDcGroup
                                   ? Extent2D{2048, 2048}
                                   : Extent2D{256, 256};
      for (auto slice : std::span(plan.slices)
                            .subspan(entry.slice_begin, entry.slice_count)) {
        if (record.channel_count == record.channels.size())
          return Invalid();
        if (auto s =
                BorrowChannelSlice(base->view(slice.channel), slice.rect,
                                   &record.channels[record.channel_count++]);
            !s.ok())
          return s;
      }
      if (record.channel_count)
        if (auto s =
                Select({ReferencePaletteStage::kLocal, source.metadata.bits},
                       budget, &stats, &record);
            !s.ok())
          return s;
    }
    auto &stream = result.streams_[i];
    stream.channels = record.training_channels().first(
        i == 0 ? entry.slice_count : record.channel_count);
    stream.transforms = record.selection.transforms;
    stream.weighted = kSelectionWeightedPresets[record.selection.wp_mode];
    if (stats.local_trees && (i == 0 || record.channel_count))
      stream.local_tree = &record.tree;
  }
  using namespace modular_gpu_internal;
  ManagedVector<ModularStreamChannel> jobs;
  ManagedVector<const DecisionTree *> trees;
  std::unique_ptr<PreparedGpuModularStreamFeatures> features;
  if (gpu) {
    ProfileScope profile(ProfileStage::kTraining);
    const auto start = std::chrono::steady_clock::now();
    size_t channels = 0;
    for (const auto &stream : result.streams_)
      for (const auto &channel : stream.channels)
        channels += !channel.descriptor.extent.empty();
    const size_t tree_count = stats.local_trees ? count : 1;
    HostStorageBound headers;
    if (!headers.AddVector<ModularStreamChannel>(channels, VectorCapacityPolicy::kFreshExact) ||
        !headers.AddVector<const DecisionTree *>(tree_count, VectorCapacityPolicy::kFreshExact))
      return Invalid();
    if (auto s = Stage(budget, headers.peak_bytes, &stats,
                       [&] {
                         jobs.resize(channels);
                         trees.resize(tree_count);
                         return Status::Ok();
                       });
        !s.ok())
      return s;
    size_t next = 0;
    for (size_t i = 0; i < count; ++i) {
      const auto &stream = result.streams_[i];
      if (stats.local_trees || i == 0)
        trees[i] = &result.records_[i].tree;
      for (size_t c = 0; c < stream.channels.size(); ++c)
        if (!stream.channels[c].descriptor.extent.empty())
          jobs[next++] = {stream.channels[c], stream.weighted,
                          static_cast<uint32_t>(result.records_[i].id), static_cast<uint32_t>(c),
                          static_cast<uint32_t>(stats.local_trees ? i : 0)};
    }
    const auto limits = gpu->modular_stream_limits();
    ModularStreamBatchPlan batch;
    if (auto s = ComputeModularStreamBatchPlan(jobs, 1, &batch, &limits); !s.ok())
      return s;
    if (auto s = Stage(budget, batch.preparation.peak_bytes, &stats,
                       [&] {
                         return gpu->PrepareModularStreamFeatures(jobs, &features,
                                                                  options.weighted_feature_backend);
                       });
        !s.ok())
      return s;
    std::span<const AdaptiveModularWeightedFeature> host;
    if (auto s = features->BorrowWeightedHostView(&host); !s.ok())
      return s;
    if (host.size() != batch.tokens.token_count)
      return Status::Internal("Incomplete Modular weighted features");
    size_t offset = 0;
    for (size_t i = 0; i < count; ++i) {
      size_t samples = 0;
      for (const auto &channel : result.streams_[i].channels)
        samples += channel.descriptor.extent.width * channel.descriptor.extent.height;
      result.records_[i].weighted_features = host.subspan(offset, samples);
      offset += samples;
    }
    stats.reused_feature_samples = host.size();
    if (active_modular_profile) {
      auto &p = active_modular_profile->adaptive_gpu;
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      p.feature_seconds += elapsed;
      p.learning_feature_seconds += elapsed;
      ++p.feature_preparations;
      ++p.learning_feature_uses;
      p.feature_samples += host.size();
    }
  }
  if (stats.local_trees) {
    for (size_t i = 0; i < result.records_.size(); ++i) {
      auto &record = result.records_[i];
      if (record.channel_count == 0)
        continue;
      const HistogramTrainingImage image{record.learning_extent, record.training_channels(),
                                         record.metadata_channels, result.streams_[i].weighted,
                                         record.weighted_features};
      size_t coded_samples = 0;
      for (auto v : result.streams_[i].channels)
        coded_samples += v.descriptor.extent.width * v.descriptor.extent.height;
      const size_t leaves =
          std::min(kMaximumTreeLeaves, (1024 + coded_samples + 1) / 2);
      if (auto s = Learn({&image, 1}, leaves, options, budget, &stats, &record);
          !s.ok())
        return s;
    }
  } else {
    // Image.empty() in the reference tests dimensions, including excluded
    // global planes. Thus the first image is still zero; trim trailing holes.
    size_t stop = 1;
    for (const auto &record : result.records_)
      if (record.channel_count)
        stop = std::max(stop, record.id + 1);
    HostStorageBound images_bound;
    if (!images_bound.AddVector<HistogramTrainingImage>(
            stop, VectorCapacityPolicy::kFreshExact))
      return Invalid();
    ManagedVector<HistogramTrainingImage> images;
    if (auto s = Stage(budget, images_bound.peak_bytes, &stats,
                       [&] {
                         images.resize(stop);
                         return Status::Ok();
                       });
        !s.ok())
      return s;
    for (size_t i = 0; i < result.records_.size(); ++i) {
      const auto &record = result.records_[i];
      if (record.id < stop)
        images[record.id] = {record.learning_extent, record.training_channels(),
                             record.metadata_channels, result.streams_[i].weighted,
                             record.weighted_features};
    }
    if (auto s = Learn(images, MaximumGlobalTreeLeaves(population), options,
                       budget, &stats, &global);
        !s.ok())
      return s;
  }
  // Reuse the coding planner's full shape/tree/header validation before
  // publishing this self-contained result, without allocating coding backing.
  ModularWorkflowStoragePlan coding;
  if (auto s = ComputePreparedModularStreamsStoragePlan(
          result.view(), options.entropy, &coding);
      !s.ok())
    return s;
  if (gpu) {
    ProfileScope profile(ProfileStage::kTokens);
    const auto start = std::chrono::steady_clock::now();
    size_t nodes = 0;
    for (const auto *tree : trees) {
      if (tree->size > UINT32_MAX - nodes)
        return Status::Unsupported("Modular stream tree table exceeds limits");
      nodes += tree->size;
    }
    const auto limits = gpu->modular_stream_limits();
    ModularStreamBatchPlan batch;
    if (auto s = ComputeModularStreamBatchPlan(jobs, nodes, &batch, &limits); !s.ok())
      return s;
    if (!batch.generation.AddVector<EntropyTokenStreamView>(count,
                                                            VectorCapacityPolicy::kFreshExact))
      return Invalid();
    if (auto s = Stage(budget, batch.generation.peak_bytes, &stats,
                       [&] {
                         result.token_views_.resize(count);
                         return gpu->GenerateModularStreamTokens(*features, trees,
                                                                 &result.gpu_tokens_);
                       });
        !s.ok())
      return s;
    ModularTokenHostView host;
    if (auto s = result.gpu_tokens_->BorrowHostView(&host); !s.ok())
      return s;
    if (host.values.size() != batch.tokens.token_count ||
        host.contexts.size() != host.values.size())
      return Status::Internal("Incomplete Modular stream tokens");
    size_t offset = 0;
    for (size_t i = 0; i < count; ++i) {
      const size_t samples = result.records_[i].weighted_features.size();
      result.token_views_[i] = EntropyTokenStreamView::Split(
          host.values.subspan(offset, samples), host.contexts.subspan(offset, samples));
      result.records_[i].weighted_features = {};
      offset += samples;
    }
    stats.gpu_token_samples = host.values.size();
    if (active_modular_profile) {
      auto &p = active_modular_profile->adaptive_gpu;
      p.token_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      ++p.feature_reuses;
      ++p.encoded_candidates;
    }
  }
  *out = std::move(result);
  return Status::Ok();
} catch (const ManagedAllocationFailure &error) {
  return error.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Reference Modular workflow allocation failed");
} catch (const std::length_error &) {
  return Invalid();
}

Status EncodeReferenceModularImage(PackedModularImageView input,
                                   const ReferenceWorkflowOptions &options,
                                   const ResourceBudget &budget,
                                   codestream_internal::CodestreamBuffer *out,
                                   ReferenceWorkflowStats *stats) {
  if (!out)
    return Invalid();
  ReferenceModularFrame prepared;
  if (auto s = PrepareReferenceModularFrame(input, options, budget, &prepared);
      !s.ok())
    return s;
  ModularWorkflowStoragePlan plan;
  if (auto s = ComputePreparedModularStreamsStoragePlan(prepared.view(),
                                                        options.entropy, &plan);
      !s.ok())
    return s;
  auto result_stats = prepared.stats();
  codestream_internal::CodestreamBuffer result;
  if (auto s = Stage(budget, plan.working.peak_bytes, &result_stats,
                     [&] {
                       return EncodePreparedModularStreams(
                           prepared.view(), options.entropy, &result);
                     });
      !s.ok())
    return s;
  *out = std::move(result);
  if (stats)
    *stats = result_stats;
  return Status::Ok();
}
} // namespace gjxl::modular_internal
