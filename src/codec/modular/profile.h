// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/coding.h"
#include <array>
#include <chrono>
#include <cstddef>
#include <ctime>

namespace gjxl::modular_internal {
// Private, opt-in caller-wall and process-CPU instrumentation. Workers are
// deliberately not given the sink: tokenization/emission measure the caller's
// joined interval, with process CPU including joined workers' work.
enum class ProfileStage {
  kInput,
  kTransforms,
  kTraining,
  kTokens,
  kModel,
  kEmission,
  kAssembly,
  kCount
};
struct ModularProfile {
  std::array<double, static_cast<size_t>(ProfileStage::kCount)> seconds{};
  std::array<double, static_cast<size_t>(ProfileStage::kCount)> cpu_seconds{};
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
      cpu_start_ = std::clock();
      parent_ = current_;
      current_ = this;
    }
  }
  ~ProfileScope() {
    if (sink_) {
      const double elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
      const double cpu = double(std::clock() - cpu_start_) / CLOCKS_PER_SEC;
      sink_->seconds[static_cast<size_t>(stage_)] += elapsed - children_;
      sink_->cpu_seconds[static_cast<size_t>(stage_)] += cpu - cpu_children_;
      if (parent_ && parent_->sink_ == sink_) {
        parent_->children_ += elapsed;
        parent_->cpu_children_ += cpu;
      }
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
  std::clock_t cpu_start_ = 0;
  ProfileScope *parent_ = nullptr;
  double children_ = 0, cpu_children_ = 0;
};
} // namespace gjxl::modular_internal
