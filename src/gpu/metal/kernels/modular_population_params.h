// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
namespace gjxl_modular_population { using Word = uint; using Wide = ulong;
#else
#include <cstdint>
namespace gjxl_modular_population { using Word = uint32_t; using Wide = uint64_t;
#endif
struct Params {
  Wide status_offset;
  Word tokens, contexts, bins, shards;
};
#ifdef __METAL_VERSION__
constant Word kValueLimit = 4096, kGroupTokens = 2048, kThreads = 128;
#else
inline constexpr Word kValueLimit = 4096, kGroupTokens = 2048, kThreads = 128;
#endif
} // namespace gjxl_modular_population
