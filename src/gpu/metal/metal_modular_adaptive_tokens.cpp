// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/metal/metal_backend_internal.h"
#include "gpu/ops/modular_adaptive_layout.h"
#include <cstring>

namespace gjxl::metal_internal {
using namespace modular_gpu_internal;
using namespace modular_internal;
using namespace resource_budget_internal;
namespace {
using Params = gjxl_modular_adaptive::Params;
using Slice = gjxl_modular_adaptive::Slice;
using Feature = gjxl_modular_adaptive::WeightedFeature;
using Node = gjxl_modular_adaptive::Node;
constexpr size_t kGuard = 256;
static_assert(sizeof(Params) == 40 && alignof(Params) == 8);
static_assert(sizeof(Slice) == sizeof(AdaptiveModularTokenSlice) && alignof(Slice) == 8);
static_assert(sizeof(Feature) == sizeof(AdaptiveModularWeightedFeature) && alignof(Feature) == 8);
static_assert(sizeof(Node) == sizeof(ModularTokenNode));
static_assert(sizeof(AdaptiveModularDiagnostic) == 272 &&
              offsetof(AdaptiveModularDiagnostic, leaf) == 248);
#define OFFSET(A, B, f) static_assert(offsetof(A, f) == offsetof(B, f))
OFFSET(Slice, AdaptiveModularTokenSlice, token_offset);
OFFSET(Slice, AdaptiveModularTokenSlice, plane_offset_elements);
OFFSET(Slice, AdaptiveModularTokenSlice, plane_stride_elements);
OFFSET(Slice, AdaptiveModularTokenSlice, stream_id);
OFFSET(Slice, AdaptiveModularTokenSlice, property_channel);
OFFSET(Slice, AdaptiveModularTokenSlice, width);
OFFSET(Slice, AdaptiveModularTokenSlice, height);
OFFSET(Slice, AdaptiveModularTokenSlice, reserved);
OFFSET(Feature, AdaptiveModularWeightedFeature, prediction);
OFFSET(Feature, AdaptiveModularWeightedFeature, property15);
#undef OFFSET
bool GuardIntact(const std::byte *p, size_t count) {
  return std::all_of(p, p + count, [](std::byte b) { return b == std::byte{0xa5}; });
}
class CompletedFeatures final : public PreparedGpuAdaptiveModularFeatures {
public:
  BackendId backend_id() const noexcept override { return backend; }
  const codec_internal::ImageMetadata &metadata() const noexcept override { return source; }
  size_t token_count() const noexcept override { return layout.tokens; }
  std::span<const AdaptiveModularTokenSlice> slices() const noexcept override {
    return descriptors;
  }
  bool has_weighted_features() const noexcept override { return weighted != nullptr; }
  Status
  BorrowWeightedHostView(std::span<const AdaptiveModularWeightedFeature> *out) const override {
    if (!out)
      return Status::InvalidArgument("Null adaptive Modular weighted view");
    *out = host_weighted;
    return Status::Ok();
  }
  BackendId backend;
  codec_internal::ImageMetadata source;
  AdaptiveModularLayout layout;
  ManagedVector<AdaptiveModularTokenSlice> descriptors;
  std::unique_ptr<DeviceBuffer> planes, weighted;
  std::span<const AdaptiveModularWeightedFeature> host_weighted;
};
class CompletedTokens final : public PreparedGpuModularTokens {
public:
  CompletedTokens(BackendId backend, AdaptiveModularTokenStoragePlan plan, size_t contexts)
      : backend_(backend), plan_(plan), contexts_(contexts) {}
  BackendId backend_id() const noexcept override { return backend_; }
  size_t token_count() const noexcept override { return plan_.token_count; }
  size_t context_count() const noexcept override { return contexts_; }
  DeviceMemoryRange values() const noexcept override {
    return {buffer.get(), 0, plan_.values_bytes};
  }
  DeviceMemoryRange contexts() const noexcept override {
    return {buffer.get(), plan_.contexts_offset, plan_.token_count * sizeof(uint16_t)};
  }
  Status BorrowHostView(ModularTokenHostView *out) const override {
    if (!out)
      return Status::InvalidArgument("Null adaptive Modular token view");
    *out = host;
    return Status::Ok();
  }
  std::unique_ptr<DeviceBuffer> buffer;
  ModularTokenHostView host;

private:
  BackendId backend_;
  AdaptiveModularTokenStoragePlan plan_;
  size_t contexts_;
};
struct Pending {
  std::unique_ptr<GpuSubmission> submission;
  bool completed = false;
  ~Pending() {
    if (submission && !completed)
      DrainMetalSubmission(*submission);
  }
  Status Wait() {
    const auto s = submission->Wait();
    completed = true;
    return s;
  }
};
struct EncodeContext {
  MTL::Buffer *planes, *weighted, *output, *dispatch, *diagnostics = nullptr;
  MTL::ComputePipelineState *pipeline;
  Params params;
  size_t maximum_slice_tokens;
};
void Encode(MetalBackend &, MTL::ComputeCommandEncoder *encoder, const void *opaque) {
  const auto &c = *static_cast<const EncodeContext *>(opaque);
  encoder->setComputePipelineState(c.pipeline);
  RecordMetalComputePipelineState(c.pipeline);
  encoder->setBuffer(c.planes, 0, 0);
  encoder->setBuffer(c.weighted, 0, 1);
  encoder->setBuffer(c.output, 0, 2);
  encoder->setBuffer(c.dispatch, 0, 3);
  encoder->setBytes(&c.params, sizeof(Params), 4);
  if (c.diagnostics)
    encoder->setBuffer(c.diagnostics, kGuard, 5);
  DispatchMetalThreads(encoder, MTL::Size(c.maximum_slice_tokens, c.params.slice_count, 1),
                       MTL::Size(64, 1, 1));
}
} // namespace
ModularTokenDeviceLimits MetalBackend::adaptive_modular_limits() const noexcept {
  return {size_t(PTRDIFF_MAX) / sizeof(uint32_t), UINT32_MAX,
          static_cast<size_t>(device_->maxBufferLength())};
}
Status MetalBackend::PrepareAdaptiveModularFeatures(
    const ModularEncoderFrame &frame, std::span<const AdaptiveModularTokenSlice> slices,
    const WeightedPredictorParameters &parameters, bool weighted,
    std::unique_ptr<PreparedGpuAdaptiveModularFeatures> *out) try {
  if (!out || !parameters.valid())
    return Status::InvalidArgument("Invalid adaptive Modular feature output/parameters");
  AdaptiveModularLayout layout;
  if (auto s = ComputeAdaptiveModularLayout(frame, &layout); !s.ok())
    return s;
  if (auto s = ValidateAdaptiveModularSlices(layout, slices); !s.ok())
    return s;
  AdaptiveModularTokenStoragePlan storage;
  const auto limits = adaptive_modular_limits();
  if (auto s =
          ComputeAdaptiveModularTokenStoragePlan(layout.shape.channels(), layout.shape.metadata,
                                                 slices.size(), 1, weighted, 1, &storage, &limits);
      !s.ok())
    return s;
  const ManagedHostScope resources(ResourceClass::kPreparation);
  auto candidate = std::make_unique<CompletedFeatures>();
  candidate->backend = id();
  candidate->source = frame.metadata();
  candidate->layout = layout;
  candidate->descriptors.assign(slices.begin(), slices.end());
  {
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    if (auto s = Allocate(storage.plane_bytes, &candidate->planes); !s.ok())
      return s;
    if (weighted)
      if (auto s = Allocate(storage.weighted_bytes, &candidate->weighted); !s.ok())
        return s;
  }
  auto *planes = static_cast<std::byte *>(AsMetalBuffer(*candidate->planes)->contents());
  std::memset(planes, 0, storage.plane_bytes);
  for (size_t c = 0; c < layout.shape.count; ++c) {
    const auto source = frame.image().view(c);
    const size_t bytes =
        source.descriptor.extent.width * source.descriptor.extent.height * sizeof(int32_t);
    std::memcpy(planes + layout.plane_offsets[c], source.backing.data(), bytes);
  }
  if (weighted) {
    // Shared storage is initialized directly: no full-image feature staging or
    // hidden tokenization. Each scalar weighted recurrence resets at its slice.
    auto *features = static_cast<AdaptiveModularWeightedFeature *>(
        AsMetalBuffer(*candidate->weighted)->contents());
    if (auto s = VisitAdaptiveModularSlices(
            layout,
            [&](const auto &slice, size_t c, ModularRect rect) {
              auto view = frame.image().view(c);
              view.backing = view.backing.subspan(rect.y * view.stride + rect.x);
              view.descriptor.extent = rect.extent;
              WeightedPredictor<ResourceClass::kPreparation> wp(slice.width, parameters);
              for (size_t y = 0; y < slice.height; ++y)
                for (size_t x = 0; x < slice.width; ++x) {
                  const auto n = Neighbors(view, x, y);
                  const auto p =
                      wp.Predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
                  features[slice.token_offset + y * slice.width + x] = {p.first, p.second};
                  if (!wp.Update(view.Row(y)[x], x, y))
                    return Status::Unsupported("Adaptive Modular weighted "
                                               "state exceeds supported range");
                }
              return Status::Ok();
            });
        !s.ok())
      return s;
    std::span<const std::byte> bytes;
    if (auto s = BorrowCompletedReadOnly(*candidate->weighted, storage.weighted_bytes, 0, &bytes);
        !s.ok())
      return s;
    candidate->host_weighted = {
        reinterpret_cast<const AdaptiveModularWeightedFeature *>(bytes.data()),
        storage.token_count};
  }
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Adaptive Modular feature allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Adaptive Modular feature storage overflow");
}

Status
MetalBackend::GenerateAdaptiveModularTokens(const PreparedGpuAdaptiveModularFeatures &features,
                                            const DecisionTree &tree,
                                            std::unique_ptr<PreparedGpuModularTokens> *out) {
  return GenerateAdaptiveModularTokensImpl(features, tree, out);
}
Status MetalBackend::GenerateAdaptiveModularTokensImpl(
    const PreparedGpuAdaptiveModularFeatures &features, const DecisionTree &tree,
    std::unique_ptr<PreparedGpuModularTokens> *out, DeviceBuffer *diagnostics) try {
  const auto *input = dynamic_cast<const CompletedFeatures *>(&features);
  if (!out || !input || input->backend_id() != id())
    return Status::InvalidArgument("Invalid adaptive Modular feature owner/output");
  TreeLayout tree_layout;
  if (auto s = ValidateTree(tree, &tree_layout); !s.ok())
    return s;
  const auto &metadata = input->metadata();
  const size_t channels = (metadata.color == codec_internal::SourceColor::kGraySrgb ? 1 : 3) +
                          static_cast<size_t>(metadata.alpha.has_value());
  if (tree_layout.leaves >
      MaximumGlobalTreeLeaves(uint64_t{metadata.extent.width} * metadata.extent.height * channels))
    return Status::Unsupported("Adaptive Modular tree exceeds the image-dependent decoder limit");
  if (tree_layout.weighted && !input->has_weighted_features())
    return Status::Unsupported("Adaptive Modular tree needs weighted features");
  AdaptiveModularTokenStoragePlan plan;
  const auto limits = adaptive_modular_limits();
  if (auto s = ComputeAdaptiveModularTokenStoragePlan(
          input->layout.shape.channels(), input->layout.shape.metadata, input->slices().size(),
          tree.size, input->has_weighted_features(), 1, &plan, &limits);
      !s.ok())
    return s;
  EncodeContext context{};
  context.planes = AsMetalBuffer(*input->planes)->handle();
  context.weighted = input->weighted ? AsMetalBuffer(*input->weighted)->handle() : nullptr;
  context.params = {
      plan.contexts_offset,
      plan.nodes_offset,
      plan.status_offset,
      static_cast<uint32_t>(plan.slice_count),
      static_cast<uint32_t>(tree.size),
      static_cast<uint32_t>(diagnostics ? input->has_weighted_features() : tree_layout.weighted),
      0};
  context.maximum_slice_tokens = input->layout.maximum_slice_tokens;
  context.pipeline = modular_token_pipelines_[diagnostics ? 4 : 3].get();
  if (diagnostics) {
    if (plan.token_count > (size_t(PTRDIFF_MAX) - 2 * kGuard) / sizeof(AdaptiveModularDiagnostic) ||
        plan.output_bytes >
            limits.maximum_buffer_bytes - std::min(kGuard, limits.maximum_buffer_bytes))
      return Status::Unsupported("Adaptive Modular diagnostic storage exceeds limits");
    MetalBuffer *native = nullptr;
    if (auto s = RequireMetalBuffer(
            diagnostics, plan.token_count * sizeof(AdaptiveModularDiagnostic) + 2 * kGuard, 0,
            "adaptive Modular diagnostics", &native);
        !s.ok())
      return s;
    context.diagnostics = native->handle();
  }
  const ManagedHostScope resources(ResourceClass::kPreparation);
  auto candidate = std::make_unique<CompletedTokens>(id(), plan, tree_layout.leaves);
  std::unique_ptr<DeviceBuffer> dispatch;
  Pending pending;
  {
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    if (auto s = Allocate(plan.output_bytes + (diagnostics ? kGuard : 0), &candidate->buffer);
        !s.ok())
      return s;
  }
  if (diagnostics)
    std::memset(AsMetalBuffer(*candidate->buffer)->contents(), 0xa5,
                candidate->buffer->size_bytes());
  if (auto s = Allocate(plan.dispatch_bytes, &dispatch); !s.ok())
    return s;
  auto *bytes = static_cast<std::byte *>(AsMetalBuffer(*dispatch)->contents());
  std::memset(bytes, 0, plan.dispatch_bytes);
  std::memcpy(bytes, input->slices().data(), plan.slices_bytes);
  auto *nodes = reinterpret_cast<ModularTokenNode *>(bytes + plan.nodes_offset);
  for (size_t i = 0; i < tree.size; ++i) {
    const auto &n = tree.nodes[i];
    nodes[i] = {n.property,
                n.split,
                n.left,
                n.right,
                static_cast<uint32_t>(n.predictor),
                n.offset,
                n.multiplier,
                tree_layout.context[i]};
  }
  context.output = AsMetalBuffer(*candidate->buffer)->handle();
  context.dispatch = AsMetalBuffer(*dispatch)->handle();
  if (!context.weighted)
    context.weighted = context.dispatch; // Unused, safely sized binding.
  if (auto s = SubmitCompute("modular.adaptive_tokens", Encode, &context, &pending.submission);
      !s.ok())
    return s;
  if (!pending.submission)
    return Status::Internal("Adaptive Modular operation returned no submission");
  if (auto s = pending.Wait(); !s.ok())
    return s;
  uint32_t status;
  std::memcpy(&status, bytes + plan.status_offset, sizeof(status));
  if (auto s = CheckModularTokenDeviceStatus(status); !s.ok())
    return s;
  if (diagnostics) {
    const auto *p = static_cast<const std::byte *>(AsMetalBuffer(*candidate->buffer)->contents());
    if (!GuardIntact(p + plan.values_bytes, plan.contexts_offset - plan.values_bytes) ||
        !GuardIntact(p + plan.output_bytes, kGuard))
      return Status::DeviceError("Adaptive Modular token guard overwritten");
  }
  std::span<const std::byte> output;
  if (auto s = BorrowCompletedReadOnly(*candidate->buffer, plan.output_bytes, 0, &output); !s.ok())
    return s;
  candidate->host = {
      {reinterpret_cast<const uint32_t *>(output.data()), plan.token_count},
      {reinterpret_cast<const uint16_t *>(output.data() + plan.contexts_offset), plan.token_count}};
  pending.submission.reset();
  dispatch.reset();
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Adaptive Modular token allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Adaptive Modular token storage overflow");
}

Status
MetalBackend::DiagnoseAdaptiveModularTokens(const PreparedGpuAdaptiveModularFeatures &features,
                                            const DecisionTree &tree,
                                            ManagedVector<AdaptiveModularDiagnostic> *out) try {
  if (!out || !dynamic_cast<const CompletedFeatures *>(&features) ||
      features.backend_id() != id() ||
      features.token_count() >
          (size_t(PTRDIFF_MAX) - 2 * kGuard) / sizeof(AdaptiveModularDiagnostic))
    return Status::InvalidArgument("Invalid adaptive Modular diagnostics");
  const ManagedHostScope resources(ResourceClass::kDiagnostics);
  const size_t bytes = features.token_count() * sizeof(AdaptiveModularDiagnostic);
  std::unique_ptr<DeviceBuffer> diagnostic;
  if (auto s = Allocate(bytes + 2 * kGuard, &diagnostic); !s.ok())
    return s;
  auto *data = static_cast<std::byte *>(AsMetalBuffer(*diagnostic)->contents());
  std::memset(data, 0xa5, bytes + 2 * kGuard);
  std::unique_ptr<PreparedGpuModularTokens> tokens;
  if (auto s = GenerateAdaptiveModularTokensImpl(features, tree, &tokens, diagnostic.get());
      !s.ok())
    return s;
  if (!GuardIntact(data, kGuard) || !GuardIntact(data + kGuard + bytes, kGuard))
    return Status::DeviceError("Adaptive Modular diagnostic guard overwritten");
  ManagedVector<AdaptiveModularDiagnostic> candidate(features.token_count());
  std::memcpy(candidate.data(), data + kGuard, bytes);
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Adaptive Modular diagnostic allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Adaptive Modular diagnostic storage overflow");
}
Status DiagnoseMetalAdaptiveModularTokens(GpuBackend &backend,
                                          const PreparedGpuAdaptiveModularFeatures &features,
                                          const DecisionTree &tree,
                                          ManagedVector<AdaptiveModularDiagnostic> *out) {
  auto *metal = dynamic_cast<MetalBackend *>(&backend);
  if (!metal)
    return Status::Unsupported("Adaptive Modular diagnostics require Metal");
  return metal->DiagnoseAdaptiveModularTokens(features, tree, out);
}
} // namespace gjxl::metal_internal
