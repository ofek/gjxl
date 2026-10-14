// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <cstdlib>
#include <stdexcept>

namespace gjxl_test {
// Call only before starting test workers or after joining them. A null value
// removes the variable; Windows' CRT uses an empty value for that operation.
inline void SetEnvironment(const char* name, const char* value = nullptr) {
#ifdef _WIN32
  const int result = _putenv_s(name, value ? value : "");
#else
  const int result = value ? setenv(name, value, 1) : unsetenv(name);
#endif
  if (result != 0) throw std::runtime_error("Could not set test environment");
}
}  // namespace gjxl_test
