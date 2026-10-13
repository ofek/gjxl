// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iosfwd>
#include <vector>

#include "lib/jxl/modular/encoding/enc_ma.h"

namespace reference_learning {
using Clock = std::chrono::steady_clock;
struct Timing {
  double wall = 0, cpu = 0;
};
struct Timer {
  Clock::time_point start = Clock::now();
  std::clock_t cpu_start = std::clock();
  Timing Elapsed() const;
};
struct FitEvent {
  size_t samples = 0, distinct = 0, properties = 0, predictors = 0, nodes = 0;
  float threshold = 0, fast_decode_multiplier = 0;
  Timing fit;
  bool replayed = false;
};
struct Fixture;
struct Session {
  // Empty directory disables copying/serialization/replay for timing runs.
  std::filesystem::path directory;
  std::vector<FitEvent> fits;
  std::vector<Timing> learning;
  // Test-only in-memory capture. Null in ordinary capture/timing runs.
  std::vector<Fixture> *captured = nullptr;
};
extern thread_local Session *active;
class ScopedSession {
public:
  explicit ScopedSession(Session *value);
  ~ScopedSession();
  ScopedSession(const ScopedSession &) = delete;
  ScopedSession &operator=(const ScopedSession &) = delete;

private:
  Session *previous_;
};
struct LearningTimer {
  Timer timer;
  ~LearningTimer();
};
struct Fixture {
  jxl::TreeSamples samples;
  float threshold = 0, fast_decode_multiplier = 0;
  std::vector<jxl::ModularMultiplierInfo> multipliers;
  jxl::StaticPropRange range{};
  jxl::Tree expected;
};
void WriteFixture(const std::filesystem::path &path, const Fixture &fixture);
Fixture ReadFixture(const std::filesystem::path &path);
bool SameTree(const jxl::Tree &a, const jxl::Tree &b);
Timing Replay(const Fixture &fixture);
void PrintSession(std::ostream &out, const Session &session);
} // namespace reference_learning
