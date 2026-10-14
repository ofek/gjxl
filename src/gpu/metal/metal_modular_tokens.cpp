// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/geometry.h"
#include "gpu/metal/kernels/modular_token_params.h"
#include "gpu/metal/metal_backend_internal.h"
#include "gpu/ops/modular_token_storage_plan.h"
#include <algorithm>
#include <cstring>

namespace gjxl::metal_internal {
using namespace modular_gpu_internal;
using namespace modular_internal;
using namespace resource_budget_internal;
namespace {
using Params = gjxl_modular_tokens::Params;
using Slice = gjxl_modular_tokens::Slice;
using Node = gjxl_modular_tokens::Node;
constexpr size_t kDiagnosticGuard = 256;
static_assert(sizeof(ModularTokenDiagnostic) == 140 && alignof(ModularTokenDiagnostic) == 4);
static_assert(offsetof(ModularTokenDiagnostic, leaf) == 116 &&
              offsetof(ModularTokenDiagnostic, context) == 136);
bool GuardIntact(const std::byte *bytes, size_t count) {
  return std::all_of(bytes, bytes + count, [](std::byte b) { return b == std::byte{0xa5}; });
}
static_assert(std::is_standard_layout_v<Params> && std::is_trivially_copyable_v<Params>);
static_assert(sizeof(Params) == 72 && alignof(Params) == 8 && offsetof(Params, source_width) == 56);
static_assert(sizeof(Slice) == sizeof(ModularTokenSlice) &&
              alignof(Slice) == alignof(ModularTokenSlice));
static_assert(sizeof(Node) == sizeof(ModularTokenNode) &&
              alignof(Node) == alignof(ModularTokenNode));
#define MATCH_OFFSET(A, B, field) static_assert(offsetof(A, field) == offsetof(B, field))
MATCH_OFFSET(Slice, ModularTokenSlice, token_offset);
MATCH_OFFSET(Slice, ModularTokenSlice, stream_id);
MATCH_OFFSET(Slice, ModularTokenSlice, channel);
MATCH_OFFSET(Slice, ModularTokenSlice, x);
MATCH_OFFSET(Slice, ModularTokenSlice, y);
MATCH_OFFSET(Slice, ModularTokenSlice, width);
MATCH_OFFSET(Slice, ModularTokenSlice, height);
MATCH_OFFSET(Node, ModularTokenNode, property);
MATCH_OFFSET(Node, ModularTokenNode, split);
MATCH_OFFSET(Node, ModularTokenNode, left);
MATCH_OFFSET(Node, ModularTokenNode, right);
MATCH_OFFSET(Node, ModularTokenNode, predictor);
MATCH_OFFSET(Node, ModularTokenNode, offset);
MATCH_OFFSET(Node, ModularTokenNode, multiplier);
MATCH_OFFSET(Node, ModularTokenNode, context);
#undef MATCH_OFFSET
static_assert(gjxl_modular_tokens::kResidualError == kModularTokenResidualError);

class CompletedTokens final : public PreparedGpuModularTokens {
public:
  CompletedTokens(BackendId backend, ModularTokenStoragePlan plan, size_t contexts)
      : backend_(backend), plan_(plan), contexts_(contexts) {}
  BackendId backend_id() const noexcept override { return backend_; }
  size_t token_count() const noexcept override { return plan_.token_count; }
  size_t context_count() const noexcept override { return contexts_; }
  DeviceMemoryRange values() const noexcept override {
    return {buffer.get(), 0, plan_.values_bytes};
  }
  DeviceMemoryRange contexts() const noexcept override {
    return {buffer.get(), plan_.contexts_offset, plan_.contexts_bytes};
  }
  Status BorrowHostView(ModularTokenHostView *out) const override {
    if (!out)
      return Status::InvalidArgument("Null Metal Modular token host view");
    *out = host;
    return Status::Ok();
  }
  // Filled after completion/status validation, before publication. No input,
  // dispatch scratch, submission or duplicate charge survives in this owner.
  std::unique_ptr<DeviceBuffer> buffer;
  ModularTokenHostView host;

private:
  BackendId backend_;
  ModularTokenStoragePlan plan_;
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
    const auto status = submission->Wait();
    completed = true;
    return status;
  }
};
struct EncodeContext {
  std::array<MTL::Buffer *, 4> input;
  MTL::Buffer *output, *dispatch;
  MTL::ComputePipelineState *pipeline;
  Params params;
  size_t maximum_slice_tokens;
  bool prepare = false;
  MetalModularInputEncoding preparation;
  MTL::Buffer *diagnostics = nullptr;
};
void Encode(MetalBackend &, MTL::ComputeCommandEncoder *encoder, const void *opaque) {
  const auto &c = *static_cast<const EncodeContext *>(opaque);
  if (c.prepare) {
    EncodeMetalModularInput(encoder, c.preparation);
    const MTL::Resource *dependencies[] = {c.preparation.output};
    encoder->memoryBarrier(dependencies, 1);
  }
  encoder->setComputePipelineState(c.pipeline);
  RecordMetalComputePipelineState(c.pipeline);
  for (size_t i = 0; i < 4; ++i)
    encoder->setBuffer(c.input[i], 0, i);
  encoder->setBuffer(c.output, 0, 4);
  encoder->setBuffer(c.dispatch, 0, 5);
  encoder->setBytes(&c.params, sizeof(Params), 6);
  if (c.diagnostics)
    encoder->setBuffer(c.diagnostics, kDiagnosticGuard, 7);
  DispatchMetalThreads(encoder, MTL::Size(c.maximum_slice_tokens, c.params.slice_count, 1),
                       MTL::Size(64, 1, 1));
}
} // namespace

ModularTokenDeviceLimits MetalBackend::modular_token_limits() const noexcept {
  return {size_t(PTRDIFF_MAX) / sizeof(uint32_t), UINT32_MAX,
          static_cast<size_t>(device_->maxBufferLength())};
}

Status MetalBackend::GenerateModularTokens(const PreparedGpuModularInput &input,
                                           const ModularTokenizationOptions &options,
                                           std::span<const ModularTokenSlice> slices,
                                           std::unique_ptr<PreparedGpuModularTokens> *out) {
  return GenerateModularTokensImpl(&input, nullptr, options, slices, out);
}
Status MetalBackend::PrepareModularTokens(PackedModularImageView input,
                                          const ModularTokenizationOptions &options,
                                          std::span<const ModularTokenSlice> slices,
                                          std::unique_ptr<PreparedGpuModularTokens> *out) {
  return GenerateModularTokensImpl(nullptr, &input, options, slices, out);
}
Status MetalBackend::GenerateModularTokensImpl(const PreparedGpuModularInput *resident,
                                               const PackedModularImageView *packed,
                                               const ModularTokenizationOptions &options,
                                               std::span<const ModularTokenSlice> slices,
                                               std::unique_ptr<PreparedGpuModularTokens> *out,
                                               DeviceBuffer *diagnostics) try {
  if (!out || (resident && resident->backend_id() != id()))
    return Status::InvalidArgument("Invalid Metal Modular token owner/output");
  if (packed) {
    if (auto s = packed->Validate(); !s.ok())
      return s;
    if (packed->format != options.format)
      return Status::InvalidArgument("Metal Modular token packed format mismatch");
  }
  ModularTokenEligibility eligibility;
  if (auto s = ResolveModularTokenEligibility(options, &eligibility); !s.ok())
    return s;
  ModularInputProfile profile;
  if (auto s = ResolveModularInput(packed ? packed->extent : resident->metadata().extent,
                                   options.format, &profile);
      !s.ok())
    return s;
  if (options.coding.rct)
    profile.metadata.modular_16_bit_buffer_sufficient = false;
  if (resident &&
      (resident->rct() != options.coding.rct || resident->metadata() != profile.metadata ||
       resident->channels().size() != profile.channel_count))
    return Status::InvalidArgument("Metal Modular token input format/RCT mismatch");
  const auto extent = profile.metadata.extent;
  ModularInputStorageOptions packed_layout;
  ModularInputStoragePlan packed_plan;
  if (packed) {
    packed_layout = {
        extent, packed->row_stride, packed->format, packed->byte_order, options.coding.rct, false};
    const auto input_limits = ModularInputLimits();
    if (auto s = ComputeModularInputStoragePlan(packed_layout, &packed_plan, &input_limits);
        !s.ok())
      return s;
  }
  ModularFrameGeometry geometry;
  if (auto s = ModularFrameGeometry::Create(extent, &geometry); !s.ok())
    return s;
  if (18 + 3 * geometry.dc_group_count() + geometry.group_count() - 1 > INT32_MAX)
    return Status::Unsupported("Metal Modular token stream IDs exceed property range");
  const bool global = extent.width <= kGroupDimension && extent.height <= kGroupDimension;
  const size_t groups = global ? 1 : geometry.group_count();
  if (slices.size() != groups * profile.channel_count)
    return Status::InvalidArgument("Metal Modular token slice count mismatch");
  size_t area;
  if (!extent.try_area(&area) || uint64_t{area} > (uint64_t{1} << 40) ||
      area > SIZE_MAX / profile.channel_count)
    return Status::InvalidArgument("Metal Modular token count overflow");
  ModularTokenStoragePlan plan;
  const auto limits = modular_token_limits();
  if (auto s = ComputeModularTokenStoragePlan(area * profile.channel_count, slices.size(), &plan,
                                              &limits);
      !s.ok())
    return s;
  EncodeContext context{};
  context.params.source_width = static_cast<uint32_t>(extent.width);
  context.params.slice_count = static_cast<uint32_t>(slices.size());
  context.params.node_count = static_cast<uint32_t>(eligibility.node_count);
  context.params.contexts_offset_bytes = plan.contexts_offset;
  context.params.nodes_offset_bytes = plan.nodes_offset;
  context.params.status_offset_bytes = plan.status_offset;
  if (resident) {
    std::array<DeviceMemoryRange, 4> ranges{};
    for (size_t c = 0; c < profile.channel_count; ++c) {
      const auto &channel = resident->channels()[c];
      const auto &p = channel.plane;
      if (channel.descriptor != profile.descriptors[c] || p.extent != extent ||
          p.row_stride != extent.width || p.element_type != DeviceElementType::kI32)
        return Status::InvalidArgument("Metal Modular token input plane mismatch");
      if (auto s = ComputeDevicePlaneRange(p, id(), &ranges[c]); !s.ok())
        return s;
      const MetalBuffer *native = nullptr;
      if (auto s = RequireMetalBuffer(p.buffer, ranges[c].size_bytes, p.offset_bytes,
                                      "Modular token input", &native);
          !s.ok())
        return s;
      for (size_t prior = 0; prior < c; ++prior)
        if (DeviceRangesOverlap(ranges[c], ranges[prior]))
          return Status::InvalidArgument("Metal Modular token input planes overlap");
      context.input[c] = native->handle();
      context.params.plane_offset_elements[c] = p.offset_bytes / sizeof(int32_t);
    }
    for (size_t c = profile.channel_count; c < 4; ++c)
      context.input[c] = context.input[0];
  }
  // Reconstruct the canonical RCT-only rectangles without allocating a second
  // stream plan. Reject duplicate/reordered/overlapping slices and static IDs.
  size_t next = 0, index = 0;
  for (size_t group = 0; group < groups; ++group) {
    ModularRect rect{0, 0, extent};
    if (!global) {
      if (auto s = geometry.GroupRect(group, false, &rect); !s.ok())
        return s;
    }
    rect.extent.width = std::min(rect.extent.width, extent.width - rect.x);
    rect.extent.height = std::min(rect.extent.height, extent.height - rect.y);
    const size_t count = rect.extent.width * rect.extent.height;
    const size_t stream = global ? 0 : 18 + 3 * geometry.dc_group_count() + group;
    for (size_t channel = 0; channel < profile.channel_count; ++channel) {
      const ModularTokenSlice expected{next,
                                       static_cast<uint32_t>(stream),
                                       static_cast<uint32_t>(channel),
                                       static_cast<uint32_t>(rect.x),
                                       static_cast<uint32_t>(rect.y),
                                       static_cast<uint32_t>(rect.extent.width),
                                       static_cast<uint32_t>(rect.extent.height)};
      if (slices[index++] != expected)
        return Status::InvalidArgument("Noncanonical Metal Modular token slice");
      next += count;
    }
    context.maximum_slice_tokens = std::max(context.maximum_slice_tokens, count);
  }
  if (next != plan.token_count)
    return Status::Internal("Metal Modular token coverage mismatch");
  if (diagnostics) {
    if (plan.token_count >
            (size_t(PTRDIFF_MAX) - 2 * kDiagnosticGuard) / sizeof(ModularTokenDiagnostic) ||
        plan.output_bytes >
            limits.maximum_buffer_bytes - std::min(kDiagnosticGuard, limits.maximum_buffer_bytes))
      return Status::Unsupported("Modular diagnostic guards exceed buffer limits");
    MetalBuffer *native = nullptr;
    const size_t bytes = plan.token_count * sizeof(ModularTokenDiagnostic) + 2 * kDiagnosticGuard;
    if (auto s = RequireMetalBuffer(diagnostics, bytes, 0, "Modular token diagnostics", &native);
        !s.ok())
      return s;
    context.diagnostics = native->handle();
  }
  const ResourceClassScope resources(ResourceClass::kPreparation);
  auto candidate = std::make_unique<CompletedTokens>(id(), plan, eligibility.context_count);
  std::unique_ptr<DeviceBuffer> dispatch, upload, planes;
  Pending pending; // Destruction drains before either backing or ticket is freed.
  if (packed) {
    if (auto s = Allocate(packed_plan.upload_bytes, &upload); !s.ok())
      return s;
    if (auto s = Allocate(packed_plan.output_bytes, &planes); !s.ok())
      return s;
    if (auto s = CopyHostToDevice(*upload, packed->bytes.data(), packed_plan.upload_bytes, 0);
        !s.ok())
      return s;
    if (auto s = ResolveModularInputDispatch({upload.get(), 0, planes.get(), 0, packed_layout},
                                             &context.preparation);
        !s.ok())
      return s;
    context.prepare = true;
    for (size_t c = 0; c < 4; ++c) {
      context.input[c] = AsMetalBuffer(*planes)->handle();
      if (c < packed_plan.channel_count)
        context.params.plane_offset_elements[c] =
            packed_plan.planes[c].offset_bytes / sizeof(int32_t);
    }
  }
  {
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    if (auto s =
            Allocate(plan.output_bytes + (diagnostics ? kDiagnosticGuard : 0), &candidate->buffer);
        !s.ok())
      return s;
  }
  if (diagnostics)
    std::memset(AsMetalBuffer(*candidate->buffer)->contents(), 0xa5,
                candidate->buffer->size_bytes());
  if (auto s = Allocate(plan.dispatch_bytes, &dispatch); !s.ok())
    return s;
  auto *native_dispatch = AsMetalBuffer(*dispatch);
  auto *bytes = static_cast<std::byte *>(native_dispatch->contents());
  std::memset(bytes, 0, plan.dispatch_bytes);
  std::memcpy(bytes, slices.data(), plan.slices_bytes);
  std::memcpy(bytes + plan.nodes_offset, eligibility.nodes.data(), sizeof(eligibility.nodes));
  context.output = AsMetalBuffer(*candidate->buffer)->handle();
  context.dispatch = native_dispatch->handle();
  context.pipeline = modular_token_pipelines_[diagnostics                 ? 2
                                              : eligibility.gradient_leaf ? 0
                                                                          : 1]
                         .get();
  if (auto s = SubmitCompute(packed ? "modular.unpack_rct_tokens" : "modular.tokens", Encode,
                             &context, &pending.submission);
      !s.ok())
    return s;
  if (!pending.submission)
    return Status::Internal("Metal Modular tokens returned no submission");
  if (auto s = pending.Wait(); !s.ok())
    return s;
  uint32_t status;
  std::memcpy(&status, bytes + plan.status_offset, sizeof(status));
  if (auto s = CheckModularTokenDeviceStatus(status); !s.ok())
    return s;
  if (diagnostics) {
    const auto *tokens =
        static_cast<const std::byte *>(AsMetalBuffer(*candidate->buffer)->contents());
    if (!GuardIntact(tokens + plan.values_bytes, plan.contexts_offset - plan.values_bytes) ||
        !GuardIntact(tokens + plan.output_bytes, kDiagnosticGuard))
      return Status::DeviceError("Metal Modular token padding/guard overwritten");
  }
  std::span<const std::byte> output;
  if (auto s = BorrowCompletedReadOnly(*candidate->buffer, plan.output_bytes, 0, &output); !s.ok())
    return s;
  candidate->host = {
      std::span(reinterpret_cast<const uint32_t *>(output.data()), plan.token_count),
      std::span(reinterpret_cast<const uint16_t *>(output.data() + plan.contexts_offset),
                plan.token_count)};
  pending.submission.reset();
  dispatch.reset();
  planes.reset();
  upload.reset();
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Metal Modular token allocation failed");
}

Status MetalBackend::DiagnoseModularTokens(const PreparedGpuModularInput &input,
                                           const ModularTokenizationOptions &options,
                                           std::span<const ModularTokenSlice> slices,
                                           ManagedVector<ModularTokenDiagnostic> *out) try {
  if (!out || input.backend_id() != id())
    return Status::InvalidArgument("Invalid Metal Modular diagnostic output/input");
  ModularTokenEligibility eligibility;
  if (auto s = ResolveModularTokenEligibility(options, &eligibility); !s.ok())
    return s;
  size_t count;
  if (!input.metadata().extent.try_area(&count) || !count ||
      count > (size_t(PTRDIFF_MAX) - 2 * kDiagnosticGuard) /
                  (eligibility.channel_count * sizeof(ModularTokenDiagnostic)))
    return Status::InvalidArgument("Modular diagnostics storage overflow");
  count *= eligibility.channel_count;
  const size_t bytes = count * sizeof(ModularTokenDiagnostic);
  const ManagedHostScope scope(ResourceClass::kDiagnostics);
  std::unique_ptr<DeviceBuffer> diagnostic;
  if (auto s = Allocate(bytes + 2 * kDiagnosticGuard, &diagnostic); !s.ok())
    return s;
  auto *buffer = static_cast<std::byte *>(AsMetalBuffer(*diagnostic)->contents());
  std::memset(buffer, 0xa5, bytes + 2 * kDiagnosticGuard);
  std::unique_ptr<PreparedGpuModularTokens> tokens;
  if (auto s =
          GenerateModularTokensImpl(&input, nullptr, options, slices, &tokens, diagnostic.get());
      !s.ok())
    return s;
  if (!GuardIntact(buffer, kDiagnosticGuard) ||
      !GuardIntact(buffer + kDiagnosticGuard + bytes, kDiagnosticGuard))
    return Status::DeviceError("Metal Modular diagnostic guard overwritten");
  ManagedVector<ModularTokenDiagnostic> candidate(count);
  std::memcpy(candidate.data(), buffer + kDiagnosticGuard, bytes);
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Metal Modular diagnostic allocation failed");
}
Status DiagnoseMetalModularTokens(GpuBackend &backend, const PreparedGpuModularInput &input,
                                  const ModularTokenizationOptions &options,
                                  std::span<const ModularTokenSlice> slices,
                                  ManagedVector<ModularTokenDiagnostic> *out) {
  auto *metal = dynamic_cast<MetalBackend *>(&backend);
  if (!metal)
    return Status::Unsupported("Modular token diagnostics require Metal");
  return metal->DiagnoseModularTokens(input, options, slices, out);
}
} // namespace gjxl::metal_internal
