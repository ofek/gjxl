// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "stream_replay.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/encoding/enc_encoding.h"
#include "lib/jxl/modular/transform/transform.h"
#include "native_preparation.h"
#include <bit>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace reference_learning {
namespace native = gjxl::modular_internal;
namespace {
void Require(bool good, const char *message) {
  if (!good)
    throw std::runtime_error(message);
}
void Ok(const gjxl::Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
// A deliberately small versioned text projection avoids adding a JSON parser
// to the codec/harness. The Python producer also binds the complete JSON trace,
// including transforms and entropy options that LearnTree does not inspect.
class Reader {
public:
  explicit Reader(const std::filesystem::path &path) : input_(path) {
    Require(std::filesystem::file_size(path) <= (16u << 20),
            "Replay description too large");
    Require(input_.good(), "Cannot open replay description");
  }
  int64_t Integer(int64_t low, int64_t high) {
    int64_t value;
    Require(bool(input_ >> value) && value >= low && value <= high,
            "Invalid or truncated replay integer");
    return value;
  }
  void Tag(const char *expected) {
    std::string tag;
    Require(bool(input_ >> tag) && tag == expected, "Invalid replay field tag");
  }
  void Finish() {
    Tag("end");
    input_ >> std::ws;
    Require(input_.peek() == std::char_traits<char>::eof(),
            "Trailing replay description data");
  }

private:
  std::ifstream input_;
};
bool SameSampling(const native::HistogramSamplingOptions &a,
                  const native::HistogramSamplingOptions &b) {
  return std::bit_cast<uint32_t>(a.sampling_fraction) ==
             std::bit_cast<uint32_t>(b.sampling_fraction) &&
         std::bit_cast<uint32_t>(a.split_threshold) ==
             std::bit_cast<uint32_t>(b.split_threshold) &&
         std::bit_cast<uint32_t>(a.fast_decode_multiplier) ==
             std::bit_cast<uint32_t>(b.fast_decode_multiplier) &&
         a.maximum_property_values == b.maximum_property_values &&
         a.maximum_channel_size == b.maximum_channel_size &&
         a.predictor_count == b.predictor_count &&
         a.property_count == b.property_count && a.predictors == b.predictors &&
         a.properties == b.properties;
}
} // namespace

void ReplayLearningInput(const std::filesystem::path &description,
                         const std::filesystem::path &pixels,
                         std::ostream &out) {
  static_assert(std::endian::native == std::endian::little &&
                sizeof(jxl::pixel_type) == 4);
  Reader in(description);
  in.Tag("gjxl-learning-input-v1");
  in.Tag("range");
  const uint32_t start = in.Integer(0, 65534),
                 stop = in.Integer(start + 1, 65535);
  const auto local_stream = in.Integer(-1, UINT32_MAX);
  const size_t expected_fits = in.Integer(0, 1);
  const size_t pixel_bytes = in.Integer(0, 512u << 20);
  Require(local_stream == -1 || (start == 0 && stop == 1),
          "Invalid local learning range");
  Require(std::filesystem::file_size(pixels) == pixel_bytes,
          "Replay pixel length mismatch");
  std::ifstream raw(pixels, std::ios::binary);
  raw.exceptions(std::ios::badbit | std::ios::failbit);
  JxlMemoryManager manager;
  Require(bool(jxl::MemoryManagerInit(&manager, nullptr)),
          "Reference memory initialization failed");
  std::vector<jxl::Image> images;
  images.reserve(stop);
  for (size_t i = 0; i < stop; ++i)
    images.emplace_back(&manager);
  std::vector<jxl::ModularOptions> options(stop);
  std::vector<native::WeightedPredictorParameters> headers(stop);
  size_t offset = 0, channel_count = 0, plane_bound = 0;
  for (size_t i = start; i < stop; ++i) {
    in.Tag("image");
    Require(in.Integer(start, stop - 1) == i,
            "Missing replay image or group-ID hole");
    auto &im = images[i];
    im.w = in.Integer(0, 1u << 20);
    im.h = in.Integer(0, 1u << 20);
    im.bitdepth = in.Integer(1, 32);
    im.nb_meta_channels = in.Integer(0, 65535);
    const size_t channels =
        in.Integer(im.nb_meta_channels, 65535 - channel_count);
    channel_count += channels;
    Require((im.w && im.h) || (!channels && !im.w && !im.h),
            "Invalid empty replay image");
    im.error = false;
    in.Tag("options");
    auto &o = options[i];
    o.max_chan_size = in.Integer(1, UINT32_MAX);
    o.group_dim = in.Integer(1, UINT32_MAX);
    o.nb_repeats =
        std::bit_cast<float>(static_cast<uint32_t>(in.Integer(0, UINT32_MAX)));
    o.splitting_heuristics_node_threshold =
        std::bit_cast<float>(static_cast<uint32_t>(in.Integer(0, UINT32_MAX)));
    o.fast_decode_multiplier =
        std::bit_cast<float>(static_cast<uint32_t>(in.Integer(0, UINT32_MAX)));
    o.max_properties =
        in.Integer(0, 0); // Previous-channel properties are unsupported.
    o.max_property_values = in.Integer(2, 256);
    o.predictor = static_cast<jxl::Predictor>(in.Integer(0, 15));
    o.wp_mode = in.Integer(0, 4);
    o.wp_tree_mode =
        static_cast<jxl::ModularOptions::TreeMode>(in.Integer(0, 3));
    o.tree_kind = static_cast<jxl::ModularOptions::TreeKind>(in.Integer(1, 1));
    o.skip_encoder_fast_path = in.Integer(0, 1);
    o.zero_tokens = in.Integer(0, 0);
    o.splitting_heuristics_properties.resize(in.Integer(1, 16));
    for (auto &p : o.splitting_heuristics_properties)
      p = in.Integer(0, 15);
    Require(
        SameSampling(NativeSamplingOptions(o),
                     NativeSamplingOptions(options[start])),
        "Per-image sampling/growth options exceed the shared native profile");
    in.Tag("wp");
    for (auto &c : headers[i].coefficients)
      c = in.Integer(0, 31);
    for (auto &w : headers[i].weights)
      w = in.Integer(0, 15);
    bool included = im.w && im.h;
    for (size_t c = 0; c < channels; ++c) {
      in.Tag("channel");
      const size_t w = in.Integer(0, 1u << 20), h = in.Integer(0, 1u << 20);
      const int hs = in.Integer(-31, 31), vs = in.Integer(-31, 31);
      const int component = in.Integer(-1, 65535);
      if (c >= im.nb_meta_channels &&
          (w > o.max_chan_size || h > o.max_chan_size))
        included = false;
      Require(in.Integer(0, 1) == included,
              "Replay channel inclusion mismatch");
      Require(in.Integer(0, pixel_bytes) == offset,
              "Replay channel offset mismatch");
      const uint64_t bytes = included ? uint64_t{w} * h * 4 : 0;
      Require(bytes <= pixel_bytes - offset &&
                  in.Integer(0, pixel_bytes) == bytes,
              "Replay channel size mismatch");
      // Bound alignment/padding conservatively before the reference allocation.
      const uint64_t allocation = bytes ? (uint64_t{w} + 256) * h * 4 + 256 : 0;
      Require(allocation <= (1u << 30) - plane_bound,
              "Replay aligned planes exceed 1 GiB");
      plane_bound += allocation;
      auto created =
          jxl::Channel::Create(&manager, bytes ? w : 0, bytes ? h : 0, hs, vs);
      Require(created.ok(), "Replay plane allocation failed");
      auto channel = std::move(created).value_();
      // Excluded channels retain shape but deliberately have no pixel backing.
      channel.w = w;
      channel.h = h;
      channel.component = component;
      if (bytes)
        for (size_t y = 0; y < h; ++y)
          raw.read(reinterpret_cast<char *>(channel.Row(y)), w * 4);
      offset += bytes;
      im.channel.push_back(std::move(channel));
    }
  }
  Require(offset == pixel_bytes && raw.peek() == std::char_traits<char>::eof(),
          "Unclaimed replay pixels");
  in.Tag("multipliers");
  std::vector<jxl::ModularMultiplierInfo> multipliers(in.Integer(0, 65536));
  std::vector<native::HistogramMultiplierRange> native_multipliers;
  for (auto &m : multipliers) {
    m.multiplier = in.Integer(1, INT32_MAX);
    for (auto &range : m.range)
      for (auto &v : range)
        v = in.Integer(0, 65535);
    native_multipliers.push_back({m.range, m.multiplier});
  }
  in.Tag("tree");
  jxl::Tree expected(in.Integer(1, native::kMaximumTreeNodes));
  for (auto &n : expected) {
    n.property = in.Integer(-1, 15);
    n.splitval = in.Integer(INT32_MIN, INT32_MAX);
    n.lchild = in.Integer(0, expected.size() - 1);
    n.rchild = in.Integer(0, expected.size() - 1);
    n.predictor = static_cast<jxl::Predictor>(in.Integer(0, 13));
    n.predictor_offset = in.Integer(INT32_MIN, INT32_MAX);
    n.multiplier = in.Integer(1, INT32_MAX);
  }
  in.Finish();
  const auto borrowed = BorrowReferenceImages(
      std::span(images).subspan(start), std::span(options).subspan(start));
  for (size_t i = start; i < stop; ++i)
    Require(borrowed.images[i - start].weighted == headers[i],
            "Replay WP preset/header mismatch");
  auto native_options = NativeSamplingOptions(options[start]);
  native_options.first_group = start;
  native_options.multipliers = native_multipliers;
  using namespace gjxl::resource_budget_internal;
  HostStorageBound prep_bound, fit_bound;
  Ok(native::ComputeHistogramSamplePreparationStorageBound(
      borrowed.images, native_options, &prep_bound));
  ResourceBudget prep_budget(prep_bound.peak_bytes);
  ResourceReservation prep_reservation;
  Ok(prep_budget.Reserve(prep_bound.peak_bytes, &prep_reservation));
  native::PreparedHistogramSamples prepared;
  {
    const ResourceContextScope scope(
        {&prep_reservation, ResourceClass::kPreparation});
    Ok(native::PrepareHistogramSamples(borrowed.images, &prepared,
                                       native_options));
  }
  Require(prep_budget.snapshot().peak_backing_bytes <= prep_bound.peak_bytes,
          "Stream preparation exceeded storage bound");
  std::vector<Fixture> fixtures;
  Session session;
  session.captured = &fixtures;
  jxl::Tree reference_tree;
  {
    const ScopedSession scope(&session);
    auto learned =
        jxl::LearnTree(images.data(), options.data(), start, stop, multipliers);
    Require(learned.ok(), "Reconstructed reference learning failed");
    reference_tree = std::move(learned).value_();
  }
  Require(SameTree(reference_tree, expected),
          "Reconstructed reference tree differs from capture");
  Require(fixtures.size() == expected_fits &&
              session.fits.size() == expected_fits &&
              bool(prepared.samples().total_samples) == bool(expected_fits),
          "Reconstructed learning fit count differs from capture");
  if (expected_fits)
    CheckPreparedSamples(prepared, fixtures.front());
  Ok(native::ComputeHistogramTreeLearningStorageBound(
      prepared.samples(), prepared.tree_options(), &fit_bound));
  ResourceBudget fit_budget(fit_bound.peak_bytes);
  ResourceReservation fit_reservation;
  Ok(fit_budget.Reserve(fit_bound.peak_bytes, &fit_reservation));
  native::DecisionTree tree;
  {
    const ResourceContextScope scope(
        {&fit_reservation, ResourceClass::kPreparation});
    Ok(native::LearnPreparedHistogramTree(prepared, &tree));
  }
  Require(fit_budget.snapshot().peak_backing_bytes <= fit_bound.peak_bytes,
          "Stream fitting exceeded storage bound");
  CheckNativeTree(tree, expected);
  const auto &stats = prepared.stats();
  out << "{\"schema\":1,\"reference_tree_exact\":true,\"samples_exact\":true,"
         "\"tree_exact\":true,\"timing_qualified\":false"
      << ",\"start\":" << start << ",\"stop\":" << stop
      << ",\"local_stream_id\":" << local_stream
      << ",\"fits\":" << expected_fits << ",\"nodes\":" << tree.size
      << ",\"samples\":" << stats.selected_samples
      << ",\"distinct\":" << prepared.samples().weights.size()
      << ",\"scanned_pixels\":" << stats.scanned_pixels
      << ",\"population\":" << stats.population
      << ",\"quantization_samples\":" << stats.quantization_samples
      << ",\"channels\":" << stats.channels
      << ",\"preparation_storage_bound_bytes\":" << prep_bound.peak_bytes
      << ",\"preparation_storage_peak_bytes\":"
      << prep_budget.snapshot().peak_backing_bytes
      << ",\"fitting_storage_bound_bytes\":" << fit_bound.peak_bytes
      << ",\"fitting_storage_peak_bytes\":"
      << fit_budget.snapshot().peak_backing_bytes << "}\n";
}
} // namespace reference_learning
