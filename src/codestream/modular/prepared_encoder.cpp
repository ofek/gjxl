// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/prepared_encoder.h"

namespace gjxl::modular_internal {
Status EncodePreparedModularFrame(ModularEncoderFrame frame, const ModularCodingPolicy &policy,
                                  EntropyCodingMode entropy,
                                  const ModularWorkflowStoragePlan &storage,
                                  codestream_internal::CodestreamBuffer *out) {
  if (!out)
    return Status::InvalidArgument("Null prepared Modular codestream output");
  auto metadata = frame.metadata();
  if (policy != ModularCodingPolicy{})
    metadata.modular_16_bit_buffer_sufficient = false;
  ChannelShape shape;
  if (auto s = DescribeImage(frame.image(), &shape); !s.ok())
    return s;
  ModularStreamPlan layout;
  if (auto s = BuildModularStreamPlan(storage.geometry, shape.channels(), shape.metadata, &layout);
      !s.ok())
    return s;
  PreparedModularTokens tokens;
  if (auto s = TokenizeModular(frame, layout, policy, &tokens, storage.participants); !s.ok())
    return s;
  frame = {}; // The CPU workflow plan excludes samples from subsequent phases.
  return EncodeModularFrame(metadata, layout, std::move(tokens), entropy, storage, out);
}
} // namespace gjxl::modular_internal
