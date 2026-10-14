// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "lib/jxl/modular/modular_image.h"
#include <vector>
namespace jxl::gjxl_selection_reference {
struct Trace {
  std::vector<float> gradient, weighted;
  size_t wp_mode = 0;
};
Status Select(Image &image, bool palettes, float channel_percent,
              int palette_colors, bool do_color, bool weighted, Trace *trace);
} // namespace jxl::gjxl_selection_reference
