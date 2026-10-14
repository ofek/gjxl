// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <filesystem>
#include <iosfwd>
#include <string>

namespace reference_learning {
// Encode frozen, already transformed e9 streams with native tree/model scopes.
// Verify tree/sample tokens independently and decode against the source PNM.
// No learning or transform search is performed. Never a timing control.
void ReplayCodingInput(const std::filesystem::path &source,
                       const std::filesystem::path &description,
                       const std::filesystem::path &output, std::ostream &out);
// v2 projection additionally carries captured entropy settings. All variants
// use the same checked tree/sample tokens and native frame/stream headers.
void ReplayEntropyInput(const std::filesystem::path &source,
                        const std::filesystem::path &description,
                        const std::filesystem::path &output, const std::string &variants,
                        std::ostream &out, bool measure = false);
// Compare the fixed-gradient selection proxy on captured planes and all 19
// eligible RCTs. No tokenization, fitting, output encoding or timing claim.
void ReplaySelectionCostInput(const std::filesystem::path &source,
                              const std::filesystem::path &description, std::ostream &out);
void ReplayWeightedCostInput(const std::filesystem::path &source,
                             const std::filesystem::path &description, std::ostream &out);
} // namespace reference_learning
