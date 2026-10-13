// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/histogram_sample_preparation.h"
#include "reference_capture.h"

namespace reference_learning {
struct BorrowedTrainingImages {
  std::vector<std::vector<gjxl::modular_internal::ModularChannelView>> channels;
  std::vector<gjxl::modular_internal::HistogramTrainingImage> images;
};
BorrowedTrainingImages
BorrowReferenceImages(std::span<const jxl::Image> images,
                      std::span<const jxl::ModularOptions> options);
gjxl::modular_internal::HistogramSamplingOptions
NativeSamplingOptions(const jxl::ModularOptions &options);
void CheckPreparedSamples(
    const gjxl::modular_internal::PreparedHistogramSamples &samples,
    const Fixture &reference);
void CheckNativeTree(const gjxl::modular_internal::DecisionTree &tree,
                     const jxl::Tree &expected);
gjxl::modular_internal::DecisionTree
CheckPreparedTree(gjxl::modular_internal::PreparedHistogramSamples &samples,
                  const jxl::Tree &expected);
size_t RunPreparationDifferentialTests();
} // namespace reference_learning
