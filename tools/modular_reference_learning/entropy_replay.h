// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/prepared_streams.h"
#include "lib/jxl/enc_ans_params.h"
#include "modular_reference.h"
#include <filesystem>
#include <iosfwd>
#include <string>

namespace reference_learning {
// All branches retain the same native frame/stream syntax and checked logical
// tokens. Reference modeling owns a fresh copy because LZ77 rewrites tokens.
void ReplayEntropyAblations(const gjxl::modular_internal::PreparedModularStreamsView &input,
                            const gjxl::modular_internal::ModularStreamPlan &plan,
                            const gjxl::modular_internal::PreparedModularStreamTokens &tokens,
                            std::span<const jxl::HistogramParams> parameters,
                            const gjxl::test::modular_reference::IntegerImage &expected,
                            const std::filesystem::path &output, const std::string &variants,
                            std::ostream &out);
} // namespace reference_learning
