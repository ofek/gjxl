// SPDX-License-Identifier: Apache-2.0
// Research-only libjxl timing. Keep trace emission outside the measured fit.
#pragma once
#include <chrono>
#include <cstdio>
#include <ctime>
namespace learning_reference_study {
using Clock = std::chrono::steady_clock;
struct Timer {
  const char* event;
  Clock::time_point start = Clock::now();
  std::clock_t cpu_start = std::clock();
  explicit Timer(const char* name) : event(name) {}
  ~Timer() {
    const double seconds = std::chrono::duration<double>(Clock::now()-start).count();
    const double cpu = double(std::clock()-cpu_start)/CLOCKS_PER_SEC;
    std::fprintf(stderr,"STUDY {\"event\":\"%s\",\"seconds\":%.17g,\"cpu\":%.17g}\n",event,seconds,cpu);
  }
};
}
