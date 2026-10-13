// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <filesystem>
#include <iosfwd>

namespace reference_learning {
// Reconstruct a captured learning call, then compare reference/native samples
// and raw trees. This is a correctness/resource check, never a timing control.
void ReplayLearningInput(const std::filesystem::path &description,
                         const std::filesystem::path &pixels,
                         std::ostream &out);
} // namespace reference_learning
