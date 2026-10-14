// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <chrono>
#include <cstdint>
#include <ctime>
#include <vector>
#include "codec/modular/coding.h"
#include "codec/modular/tree_learning.h"
namespace gjxl::modular_internal::learning_study {
using Clock = std::chrono::steady_clock;
struct Pass {
  ModularCodingPolicy base;
  size_t sample_cap = 0, leaf_cap = 0, samples = 0, population = 0, leaves = 0;
  size_t sample_bytes = sizeof(TrainingSample);
  double total = 0, small_tree = 0, fit = 0, setup = 0, parent = 0, sort = 0, sweep = 0, partition = 0;
  double cpu = 0, fit_cpu = 0;
  uint64_t nodes = 0, node_samples = 0, sorts = 0, sort_items = 0;
  uint64_t constant_properties = 0, constant_items = 0, thresholds = 0, sweep_items = 0;
};
inline thread_local std::vector<Pass>* sink = nullptr;
inline thread_local Pass* current = nullptr;
struct Timer {
  double* counter;
  double* cpu_counter;
  Clock::time_point start = Clock::now();
  std::clock_t cpu_start;
  explicit Timer(double* value, double* cpu_value=nullptr) : counter(value), cpu_counter(cpu_value), cpu_start(cpu_value ? std::clock() : 0) {}
  ~Timer() {
    if (counter) *counter += std::chrono::duration<double>(Clock::now()-start).count();
    if (cpu_counter) *cpu_counter += double(std::clock()-cpu_start)/CLOCKS_PER_SEC;
  }
};
struct PassScope {
  Pass pass;
  Pass* previous = current;
  Clock::time_point start = Clock::now();
  std::clock_t cpu_start = std::clock();
  PassScope(const ModularCodingPolicy& base, TreeLearningOptions options) {
    pass.base.rct=base.rct; pass.base.transforms=base.transforms; pass.base.weighted=base.weighted;
    pass.sample_cap=options.maximum_samples; pass.leaf_cap=options.maximum_leaves;
    current = sink ? &pass : nullptr;
  }
  ~PassScope() {
    pass.total=std::chrono::duration<double>(Clock::now()-start).count();
    pass.cpu=double(std::clock()-cpu_start)/CLOCKS_PER_SEC;
    current=previous;
    if (sink) sink->push_back(std::move(pass));
  }
};
} // namespace gjxl::modular_internal::learning_study
