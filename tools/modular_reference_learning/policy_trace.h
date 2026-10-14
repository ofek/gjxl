// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "reference_capture.h"
#include "lib/jxl/modular/modular_image.h"
#include <jxl/encode.h>

namespace reference_learning {
extern thread_local int reference_buffering;
void ConfigureReferencePolicy(JxlEncoderFrameSettings *settings);
void TraceCodingStream(const jxl::Image &image, const jxl::ModularOptions &options,
                       size_t stream);

// Marks the actual local-tree call site, including a one-group image where
// input range [0,1) alone cannot distinguish global from local learning.
class ScopedLocalLearning {
public:
  explicit ScopedLocalLearning(size_t stream);
  ~ScopedLocalLearning();
  ScopedLocalLearning(const ScopedLocalLearning &) = delete;
  ScopedLocalLearning &operator=(const ScopedLocalLearning &) = delete;
private:
  int64_t previous_;
};

class LearningInputTrace {
public:
  LearningInputTrace(const jxl::Image *images,
                     const jxl::ModularOptions *options, uint32_t start,
                     uint32_t stop,
                     const std::vector<jxl::ModularMultiplierInfo> &multipliers);
  void Finish(const jxl::Tree &tree) const;
private:
  Session *session_;
  const jxl::Image *images_;
  const jxl::ModularOptions *options_;
  uint32_t start_, stop_;
  const std::vector<jxl::ModularMultiplierInfo> &multipliers_;
  size_t fit_begin_;
  int64_t local_stream_;
};
} // namespace reference_learning
