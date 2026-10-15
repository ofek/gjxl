// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/public.h"
#include "codestream/modular/reference_workflow.h"
#include "codestream/workflow_admission_scope.h"
#include "codestream/workflow_internal.h"
#include "core/thread_budget.h"
#include <chrono>
namespace gjxl::modular_internal {
Status ResolvePublicModularPolicy(const gjxl::ModularEncodingOptions &options,
                                  ModularPolicy *policy) {
  switch (options.backend) {
  case ModularBackend::kAutomatic:
  case ModularBackend::kCpu:
  case ModularBackend::kMetal:
    break;
  case ModularBackend::kCuda:
    return Status::Unsupported("Modular CUDA encoding is not implemented");
  default:
    return Status::InvalidArgument("Invalid Modular backend");
  }
  if (options.entropy != ModularEntropy::kPrefix && options.entropy != ModularEntropy::kAns)
    return Status::InvalidArgument("Invalid Modular entropy mode");
  if (options.cpu_thread_count > 256)
    return Status::InvalidArgument("Modular CPU thread count must be at most 256");
  if (options.policy != ModularPolicy::kLearned && options.policy != ModularPolicy::kBaseline &&
      options.policy != ModularPolicy::kSearch)
    return Status::InvalidArgument("Invalid Modular policy");
  if (options.search && options.policy == ModularPolicy::kBaseline)
    return Status::InvalidArgument("Modular search conflicts with baseline policy");
  *policy = options.search ? ModularPolicy::kSearch : options.policy;
  if (options.backend == ModularBackend::kMetal) {
#if !defined(GJXL_ENABLE_METAL)
    return Status::Unsupported("Modular Metal encoding is not built");
#else
    if (*policy != ModularPolicy::kLearned)
      return Status::Unsupported("Modular Metal encoding requires the learned policy");
#endif
  }
  return Status::Ok();
}

Status EncodePublicModularImageOwned(PackedModularImageView input,
                                     const gjxl::ModularEncodingOptions &options,
                                     codestream_internal::CodestreamBuffer *output) try {
  if (!output)
    return Status::InvalidArgument("Null Modular output");
  ModularPolicy policy;
  if (auto s = ResolvePublicModularPolicy(options, &policy); !s.ok())
    return s;
  if (auto s = input.Validate(); !s.ok())
    return s;
  if (policy != ModularPolicy::kLearned) {
    ModularEncodingOptions legacy;
    legacy.execution_domain = options.execution_domain;
    legacy.cpu_thread_count = options.cpu_thread_count;
    legacy.entropy = options.entropy == ModularEntropy::kPrefix ? EntropyCodingMode::kPrefix
                                                               : EntropyCodingMode::kAns;
    legacy.search = policy == ModularPolicy::kSearch;
    return EncodeModularImageOwned(input, legacy, output);
  }
  auto context = resource_budget_internal::CurrentResourceContext();
  auto owned_domain = options.execution_domain;
  if (!owned_domain && !context.domain)
    owned_domain = ExecutionDomain::Default();
  const auto *domain = owned_domain ? owned_domain.get() : context.domain;
  const auto &budget = codestream_internal::WorkflowAdmission::Budget(*domain);
  if (context.reservation && !budget.SharesDomain(*context.reservation))
    return Status::InvalidArgument("Nested Modular workflow uses a different execution domain");
  context.domain = domain;
  resource_budget_internal::ResourceContextScope resources(context);
  thread_budget_internal::CpuExecutionScope cpu;
  if (auto s = cpu.Start(owned_domain, options.cpu_thread_count); !s.ok())
    return s;
  thread_budget_internal::EncodeScope threads(1);
  ReferenceWorkflowOptions learned;
  if (options.backend == ModularBackend::kMetal)
    if (auto s = codestream_internal::BorrowProductionMetalBackend(&learned.gpu); !s.ok())
      return s;
  if (options.entropy == ModularEntropy::kPrefix)
    learned.entropy = {PreparedModularEntropyPolicy::kPrefix};
  return EncodeReferenceModularImage(input, learned, budget, output);
} catch (const resource_budget_internal::ManagedAllocationFailure &error) {
  return error.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Public Modular workflow allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Public Modular workflow storage overflow");
}
} // namespace gjxl::modular_internal

namespace gjxl {
Status EncodeModularImage(ModularImageView input, ModularEncodingOptions options,
                          std::vector<uint8_t> *output, ModularEncodingSummary *summary) {
  if (!output)
    return Status::InvalidArgument("Null Modular output");
  const auto start = std::chrono::steady_clock::now();
  modular_internal::PackedModularImageView packed{
      input.bytes, input.extent, input.row_stride_bytes,
      static_cast<modular_internal::PackedModularFormat>(input.format),
      static_cast<modular_internal::SampleByteOrder>(input.byte_order)};
  codestream_internal::CodestreamBuffer candidate;
  if (auto s = modular_internal::EncodePublicModularImageOwned(packed, options, &candidate); !s.ok())
    return s;
  candidate.PublishTo(output);
  if (summary)
    *summary = {input.extent, output->size(),
                options.backend == ModularBackend::kMetal ? ModularBackend::kMetal
                                                          : ModularBackend::kCpu,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()};
  return Status::Ok();
}
} // namespace gjxl
