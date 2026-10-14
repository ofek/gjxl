// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "policy_trace.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/transform/transform.h"
#include <algorithm>
#include <bit>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace reference_learning {
thread_local int reference_buffering = -1;
namespace {
thread_local int64_t local_stream = -1;
void Require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template <class T> void Array(std::ostream &out, const T &values) {
  out << '[';
  bool first = true;
  for (auto value : values) {
    if (!first) out << ',';
    first = false;
    out << value;
  }
  out << ']';
}
void Options(std::ostream &out, const jxl::ModularOptions &o) {
  out << "{\"max_chan_size\":" << o.max_chan_size
      << ",\"group_dim\":" << o.group_dim
      << ",\"sampling_fraction\":" << o.nb_repeats
      << ",\"sampling_fraction_bits\":" << std::bit_cast<uint32_t>(o.nb_repeats)
      << ",\"split_threshold\":" << o.splitting_heuristics_node_threshold
      << ",\"split_threshold_bits\":" << std::bit_cast<uint32_t>(o.splitting_heuristics_node_threshold)
      << ",\"fast_decode_multiplier\":" << o.fast_decode_multiplier
      << ",\"fast_decode_bits\":" << std::bit_cast<uint32_t>(o.fast_decode_multiplier)
      << ",\"max_properties\":" << o.max_properties
      << ",\"max_property_values\":" << o.max_property_values
      << ",\"predictor\":" << static_cast<uint32_t>(o.predictor)
      << ",\"wp_mode\":" << o.wp_mode
      << ",\"wp_tree_mode\":" << static_cast<int>(o.wp_tree_mode)
      << ",\"tree_kind\":" << static_cast<int>(o.tree_kind)
      << ",\"skip_encoder_fast_path\":" << o.skip_encoder_fast_path
      << ",\"zero_tokens\":" << o.zero_tokens << ",\"properties\":";
  Array(out, o.splitting_heuristics_properties);
  jxl::weighted::Header wp;
  if (jxl::PredictorHasWeighted(o.predictor))
    jxl::weighted::PredictorMode(o.wp_mode, &wp);
  out << ",\"wp_coefficients\":";
  Array(out, std::array{wp.p1C, wp.p2C, wp.p3Ca, wp.p3Cb, wp.p3Cc, wp.p3Cd, wp.p3Ce});
  out << ",\"wp_weights\":";
  Array(out, wp.w);
  const auto &h = o.histogram_params;
  out << ",\"histogram_params\":{\"clustering\":" << static_cast<int>(h.clustering)
      << ",\"uint_method\":" << static_cast<int>(h.uint_method)
      << ",\"lz77_method\":" << static_cast<int>(h.lz77_method)
      << ",\"ans_histogram_strategy\":" << static_cast<int>(h.ans_histogram_strategy)
      << ",\"max_histograms\":" << h.max_histograms
      << ",\"force_huffman\":" << h.force_huffman
      << ",\"initialize_global_state\":" << h.initialize_global_state
      << ",\"streaming_mode\":" << h.streaming_mode
      << ",\"add_missing_symbols\":" << h.add_missing_symbols
      << ",\"add_fixed_histograms\":" << h.add_fixed_histograms
      << ",\"image_widths\":";
  Array(out, h.image_widths);
  out << "}}";
}
void Transforms(std::ostream &out, const std::vector<jxl::Transform> &transforms) {
  out << '[';
  for (size_t i = 0; i < transforms.size(); ++i) {
    const auto &t = transforms[i];
    if (i) out << ',';
    out << "{\"id\":" << static_cast<unsigned>(t.id);
    if (t.id == jxl::TransformId::kRCT) {
      out << ",\"begin_c\":" << t.begin_c << ",\"rct_type\":" << t.rct_type;
    } else if (t.id == jxl::TransformId::kPalette) {
      out << ",\"begin_c\":" << t.begin_c << ",\"num_c\":" << t.num_c
          << ",\"nb_colors\":" << t.nb_colors << ",\"nb_deltas\":" << t.nb_deltas
          << ",\"predictor\":" << static_cast<unsigned>(t.predictor)
          << ",\"ordered_palette\":" << t.ordered_palette
          << ",\"lossy_palette\":" << t.lossy_palette;
    } else {
      Require(t.id == jxl::TransformId::kSqueeze, "Invalid traced transform");
      out << ",\"squeezes\":[";
      for (size_t j = 0; j < t.squeezes.size(); ++j) {
        const auto &s = t.squeezes[j];
        if (j) out << ',';
        out << "{\"horizontal\":" << s.horizontal << ",\"in_place\":" << s.in_place
            << ",\"begin_c\":" << s.begin_c << ",\"num_c\":" << s.num_c << '}';
      }
      out << ']';
    }
    out << '}';
  }
  out << ']';
}
} // namespace

void ConfigureReferencePolicy(JxlEncoderFrameSettings *settings) {
  Require(reference_buffering == -1 || reference_buffering == 0,
          "Unsupported reference buffering control");
  // Preserve the stock API sequence when no ablation was requested.
  if (reference_buffering == 0)
    Require(JxlEncoderFrameSettingsSetOption(
                settings, JXL_ENC_FRAME_SETTING_BUFFERING, 0) == JXL_ENC_SUCCESS,
            "Reference buffering control rejected");
}
void TraceCodingStream(const jxl::Image &image, const jxl::ModularOptions &options,
                       size_t stream) {
  if (!active || active->stream_directory.empty()) return;
  const auto path = active->stream_directory /
      ("coding-" + std::to_string(active->traced_coding_streams) + ".json");
  Require(!std::filesystem::exists(path), "Refusing to overwrite coding trace");
  std::ofstream out(path);
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out << std::setprecision(17) << std::boolalpha;
  out << "{\"schema\":1,\"event\":" << active->traced_coding_streams
      << ",\"stream_id\":" << stream << ",\"local_tree\":" << (local_stream >= 0)
      << ",\"width\":" << image.w << ",\"height\":" << image.h
      << ",\"channels\":" << image.channel.size()
      << ",\"metadata_channels\":" << image.nb_meta_channels << ",\"options\":";
  Options(out, options);
  out << ",\"transforms\":"; Transforms(out, image.transform); out << "}\n";
  out.close();
  ++active->traced_coding_streams;
}
ScopedLocalLearning::ScopedLocalLearning(size_t stream) : previous_(local_stream) {
  Require(stream <= INT64_MAX, "Unrepresentable local stream ID");
  local_stream = static_cast<int64_t>(stream);
}
ScopedLocalLearning::~ScopedLocalLearning() { local_stream = previous_; }
LearningInputTrace::LearningInputTrace(
    const jxl::Image *images, const jxl::ModularOptions *options,
    uint32_t start, uint32_t stop,
    const std::vector<jxl::ModularMultiplierInfo> &multipliers)
    : session_(active), images_(images), options_(options), start_(start), stop_(stop),
      multipliers_(multipliers), fit_begin_(active ? active->fits.size() : 0),
      local_stream_(local_stream) {}

void LearningInputTrace::Finish(const jxl::Tree &tree) const {
  if (!session_ || session_->stream_directory.empty()) return;
  Require(start_ < stop_, "Invalid traced learning range");
  static_assert(std::endian::native == std::endian::little,
                "Research pixel capture is qualified on little-endian CPUs");
  static_assert(sizeof(jxl::pixel_type) == 4);
  const size_t index = session_->traced_learning_calls;
  const auto base = session_->stream_directory / ("learning-" + std::to_string(index));
  const auto metadata = base.string() + ".json", pixels = base.string() + ".i32le";
  Require(!std::filesystem::exists(metadata) && !std::filesystem::exists(pixels),
          "Refusing to overwrite stream capture");
  std::ofstream out(metadata), raw(pixels, std::ios::binary);
  out.exceptions(std::ios::badbit | std::ios::failbit);
  raw.exceptions(std::ios::badbit | std::ios::failbit);
  out << std::setprecision(17) << std::boolalpha;
  out << "{\"schema\":1,\"event\":" << index << ",\"local_stream_id\":" << local_stream_
      << ",\"start\":" << start_ << ",\"stop\":" << stop_
      << ",\"fit_begin\":" << fit_begin_ << ",\"fit_end\":" << session_->fits.size()
      << ",\"multipliers\":[";
  for (size_t i = 0; i < multipliers_.size(); ++i) {
    if (i) out << ',';
    const auto &m = multipliers_[i];
    out << "{\"multiplier\":" << m.multiplier << ",\"range\":[";
    Array(out, m.range[0]); out << ','; Array(out, m.range[1]); out << "]}";
  }
  out << "],\"images\":[";
  size_t offset = 0;
  constexpr size_t maximum_bytes = size_t{512} << 20;
  // The callback is also used to read back every serialized sample. No hash is
  // substituted for equality. The Python ledger later binds files by SHA-256.
  auto channels = [&](auto callback) {
    for (size_t i = start_; i < stop_; ++i) {
      const auto &im = images_[i];
      bool included = im.w && im.h;
      for (size_t c = 0; c < im.channel.size(); ++c) {
        const auto &ch = im.channel[c];
        if (c >= im.nb_meta_channels &&
            (ch.w > options_[i].max_chan_size || ch.h > options_[i].max_chan_size))
          included = false;
        callback(i, c, included);
      }
    }
  };
  for (size_t i = start_; i < stop_; ++i) {
    if (i != start_) out << ',';
    const auto &im = images_[i];
    out << "{\"index\":" << i << ",\"width\":" << im.w << ",\"height\":" << im.h
        << ",\"bitdepth\":" << im.bitdepth << ",\"metadata_channels\":" << im.nb_meta_channels
        << ",\"options\":";
    Options(out, options_[i]); out << ",\"transforms\":"; Transforms(out, im.transform);
    out << ",\"channels\":[";
    bool included = im.w && im.h;
    for (size_t c = 0; c < im.channel.size(); ++c) {
      const auto &ch = im.channel[c];
      if (c) out << ',';
      if (c >= im.nb_meta_channels &&
          (ch.w > options_[i].max_chan_size || ch.h > options_[i].max_chan_size))
        included = false;
      Require(!included || !ch.w || ch.h <= (maximum_bytes - offset) / 4 / ch.w,
              "Learning pixel trace exceeds 512 MiB");
      const size_t bytes = included ? ch.w * ch.h * 4 : 0;
      out << "{\"width\":" << ch.w << ",\"height\":" << ch.h
          << ",\"hshift\":" << ch.hshift << ",\"vshift\":" << ch.vshift
          << ",\"component\":" << ch.component << ",\"included\":" << included
          << ",\"offset\":" << offset << ",\"bytes\":" << bytes << '}';
      if (bytes)
        for (size_t y = 0; y < ch.h; ++y)
          raw.write(reinterpret_cast<const char *>(ch.Row(y)), ch.w * 4);
      offset += bytes;
    }
    out << "]}";
  }
  raw.close();
  std::ifstream check(pixels, std::ios::binary);
  check.exceptions(std::ios::badbit | std::ios::failbit);
  channels([&](size_t i, size_t c, bool included) {
    const auto &ch = images_[i].channel[c];
    if (!included || !ch.w || !ch.h) return;
    std::vector<jxl::pixel_type> row(ch.w);
    for (size_t y = 0; y < ch.h; ++y) {
      check.read(reinterpret_cast<char *>(row.data()), ch.w * 4);
      Require(std::equal(row.begin(), row.end(), ch.Row(y)), "Traced pixels changed on disk");
    }
  });
  Require(check.peek() == std::char_traits<char>::eof(), "Trailing pixel trace data");
  out << "],\"pixel_bytes\":" << offset << ",\"pixels_verified\":true,\"tree\":[";
  for (size_t i = 0; i < tree.size(); ++i) {
    if (i) out << ',';
    const auto &n = tree[i];
    out << '[' << n.property << ',' << n.splitval << ',' << n.lchild << ',' << n.rchild
        << ',' << static_cast<unsigned>(n.predictor) << ',' << n.predictor_offset << ',' << n.multiplier << ']';
  }
  out << "]}\n";
  out.close();
  ++session_->traced_learning_calls;
}
} // namespace reference_learning
