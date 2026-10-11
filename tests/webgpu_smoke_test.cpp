// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho

#include <cstdlib>
#include <iostream>

#include <webgpu/webgpu_cpp.h>

int main() {
  // Exercise the generated C++ headers, native entry points, and RAII lifetime
  // without requiring a GPU, window system, or codec shader on the test host.
  const wgpu::Instance instance = wgpu::CreateInstance();
  if (!instance) {
    std::cerr << "Dawn WebGPU instance creation failed\n";
    return EXIT_FAILURE;
  }
  instance.ProcessEvents();
  std::cout << "Dawn WebGPU initialization passed\n";
  return EXIT_SUCCESS;
}
