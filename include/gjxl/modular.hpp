// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include "core/geometry.h"
#include "core/execution_domain.h"
#include "core/status.h"
namespace gjxl {
// Samples are source-space unsigned integers. RGBA alpha is unassociated;
// invisible RGB is preserved. RGB uses sRGB; gray uses the sRGB transfer curve.
enum class ModularPixelFormat : uint8_t { kGray8, kRgb8, kRgba8, kGray16, kRgb16, kRgba16 };
enum class ModularByteOrder : uint8_t { kLittleEndian, kBigEndian };
enum class ModularBackend : uint8_t { kAutomatic, kCpu, kMetal, kCuda };
enum class ModularEntropy : uint8_t { kPrefix, kAns };
enum class ModularPolicy : uint8_t { kLearned, kBaseline, kSearch };
struct ModularImageView {
  std::span<const uint8_t> bytes;
  Extent2D extent;
  size_t row_stride_bytes = 0;
  ModularPixelFormat format = ModularPixelFormat::kRgb8;
  ModularByteOrder byte_order = ModularByteOrder::kLittleEndian;
};
struct ModularEncodingOptions {
  ModularBackend backend = ModularBackend::kAutomatic; // Automatic resolves to CPU.
  size_t cpu_thread_count = 0; // 0 automatic, 1 serial oracle, 2..256 upper bound.
  std::shared_ptr<const ExecutionDomain> execution_domain;
  ModularEntropy entropy = ModularEntropy::kAns;
  bool search = false; // Compatibility alias for kSearch; conflicts with kBaseline.
  // Learned selects transforms once and fits one histogram tree per scope.
  // Baseline is the legacy identity/gradient policy; Search is its legacy
  // encoded-size portfolio. All policies preserve the input samples exactly.
  ModularPolicy policy = ModularPolicy::kLearned;
};
struct ModularEncodingSummary {
  Extent2D extent;
  size_t encoded_bytes = 0;
  ModularBackend backend = ModularBackend::kCpu;
  double total_seconds = 0;
};
// Atomic output and summary replacement. Input layout and format are validated
// before admission; no float conversion occurs. Explicit Metal supports the
// learned policy, sharing weighted features between CPU learning and GPU token
// generation. Automatic remains CPU; CUDA and legacy-policy GPU requests are
// unsupported. Transform/WP selection, fitting and entropy coding stay on CPU.
// Learned uses staged, nonblocking memory admission: a busy domain can return
// Unavailable. Independent local selection and sample preparation/fits use the
// requested CPU limit and shared domain; global fitting stays serial. Local entropy models
// and payloads run across streams; shared ANS configuration search can use
// cluster workers. Memory admission may reduce concurrency without changing
// encoding decisions.
[[nodiscard]] Status EncodeModularImage(ModularImageView input, ModularEncodingOptions options,
                                        std::vector<uint8_t> *output,
                                        ModularEncodingSummary *summary = nullptr);
} // namespace gjxl
