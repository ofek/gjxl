// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/coding.h"
#include "codestream/profile_internal.h"
#include <array>
#include <chrono>
#include <cstddef>

namespace gjxl::modular_internal {
// Private, opt-in caller-wall-time instrumentation. Workers are deliberately
// not given the sink: tokenization/emission measure the caller's joined interval.
enum class ProfileStage {
  kInput,
  kTransforms,
  kTraining,
  kTokens,
  kModel,
  kEmission,
  kAssembly,
  kGpuInputTokens,
  kCount
};
// Disjoint nested attribution within kTraining. These are caller-wall intervals,
// not additive top-level phases. Only the original learner has split detail;
// the alternative quantized learner is reported as one separate interval.
enum class TrainingProfileStage {
  kRctRanking,
  kPaletteProbe,
  kSampleGathering,
  kSimpleLearning,
  kEntropyPreparation,
  kSplitHistograms,
  kPropertySorting,
  kSplitScanning,
  kTreeControl,
  kQuantizedLearning,
  kCount
};
struct TrainingWorkProfile {
  std::array<double, static_cast<size_t>(TrainingProfileStage::kCount)> seconds{};
  size_t learning_calls = 0, population = 0, samples = 0;
  size_t split_evaluations = 0, sorted_properties = 0, sorted_samples = 0;
  bool operator==(const TrainingWorkProfile &) const = default;
};
struct ModularProfile {
  std::array<double, static_cast<size_t>(ProfileStage::kCount)> seconds{};
  // Nested diagnostics within kModel, not additional caller-wall stages.
  // Accumulates across candidates. Modular currently models serially; the
  // shared counter type retains its work-time semantics if that changes.
  codestream_internal::EntropyWorkProfile entropy_work;
  TrainingWorkProfile training_work;
  // A large winning tree retains its managed backing after the encoder call.
  // Release this diagnostic copy before asserting that the domain is empty.
  ModularCodingPolicy resolved;
  size_t encoded_candidates = 0;
};
inline thread_local ModularProfile *active_modular_profile = nullptr;
class ProfileSession {
public:
  explicit ProfileSession(ModularProfile *profile) : previous_(active_modular_profile) {
    active_modular_profile = profile;
  }
  ~ProfileSession() { active_modular_profile = previous_; }
  ProfileSession(const ProfileSession &) = delete;
  ProfileSession &operator=(const ProfileSession &) = delete;

private:
  ModularProfile *previous_;
};
class ProfileScope {
  using Clock = std::chrono::steady_clock;

public:
  explicit ProfileScope(ProfileStage stage) : sink_(active_modular_profile), stage_(stage) {
    if (sink_) {
      start_ = Clock::now();
      parent_ = current_;
      current_ = this;
    }
  }
  ~ProfileScope() {
    if (sink_) {
      const double elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
      sink_->seconds[static_cast<size_t>(stage_)] += elapsed - children_;
      if (parent_ && parent_->sink_ == sink_)
        parent_->children_ += elapsed;
      current_ = parent_;
    }
  }
  ProfileScope(const ProfileScope &) = delete;
  ProfileScope &operator=(const ProfileScope &) = delete;

private:
  inline static thread_local ProfileScope *current_ = nullptr;
  ModularProfile *sink_;
  ProfileStage stage_;
  Clock::time_point start_{};
  ProfileScope *parent_ = nullptr;
  double children_ = 0;
};
class TrainingProfileScope {
  using Clock = std::chrono::steady_clock;

public:
  explicit TrainingProfileScope(TrainingProfileStage stage)
      : sink_(active_modular_profile), stage_(stage) {
    if (sink_) {
      start_ = Clock::now();
      parent_ = current_;
      current_ = this;
    }
  }
  ~TrainingProfileScope() { Stop(); }
  void Stop() {
    if (sink_) {
      const double elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
      sink_->training_work.seconds[static_cast<size_t>(stage_)] += elapsed - children_;
      if (parent_ && parent_->sink_ == sink_)
        parent_->children_ += elapsed;
      current_ = parent_;
      sink_ = nullptr;
    }
  }
  TrainingProfileScope(const TrainingProfileScope &) = delete;
  TrainingProfileScope &operator=(const TrainingProfileScope &) = delete;

private:
  inline static thread_local TrainingProfileScope *current_ = nullptr;
  ModularProfile *sink_;
  TrainingProfileStage stage_;
  Clock::time_point start_{};
  TrainingProfileScope *parent_ = nullptr;
  double children_ = 0;
};
} // namespace gjxl::modular_internal
