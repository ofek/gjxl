// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "modular_token_params.h"
namespace gjxl_modular_adaptive {
using gjxl_modular_tokens::Address;
using gjxl_modular_tokens::Node;
using gjxl_modular_tokens::Signed;
using gjxl_modular_tokens::Word;
#ifdef __METAL_VERSION__
using Wide = long;
#else
using Wide = int64_t;
#endif
struct Slice {
  Address token_offset, plane_offset_elements, plane_stride_elements;
  Word stream_id, property_channel, width, height, reserved[2];
};
struct WeightedFeature {
  Wide prediction, property15;
};
struct Params {
  Address contexts_offset_bytes, nodes_offset_bytes, status_offset_bytes;
  Word slice_count, node_count, has_weighted, reserved;
};
struct Diagnostic {
  Wide properties[16], predictions[14], selected_prediction;
  Word leaf, context;
  Signed sample, residual;
  Word value, reserved;
};
} // namespace gjxl_modular_adaptive
