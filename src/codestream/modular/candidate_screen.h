// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/candidate_cost.h"

namespace gjxl::modular_internal {
inline constexpr size_t kMaximumScreenSamples = 32768;
inline constexpr size_t kMaximumSearchShortlist = 4;
struct ModularCandidateEstimate {
  double bits = 0;
  size_t sampled_tokens = 0;
};
// Private screening only. A canonical stream gets ceil(N/stride) stratified
// samples; total samples <=32768+streams. Each stream's estimated payload is
// weighted by its own population, then real tree/model/header/TOC rules are
// added. No ordered ANS simulation and no complete-payload allocation. This
// score is never a storage reservation or an exact-rate claim.
[[nodiscard]] Status ScreenModularCandidate(
    const codec_internal::ImageMetadata &metadata, const ModularStreamPlan &layout,
    const PreparedModularTokens &tokens, EntropyCodingMode mode, ModularCandidateEstimate *out);
} // namespace gjxl::modular_internal
