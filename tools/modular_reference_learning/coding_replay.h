// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <filesystem>
#include <iosfwd>

namespace reference_learning {
// Encode frozen, already transformed e9 streams with native tree/model scopes.
// Verify tree/sample tokens independently and decode against the source PNM.
// No learning or transform search is performed. Never a timing control.
void ReplayCodingInput(const std::filesystem::path &source,
                       const std::filesystem::path &description,
                       const std::filesystem::path &output, std::ostream &out);
} // namespace reference_learning
