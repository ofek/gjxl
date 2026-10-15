// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Original-input qualification driver. No reference decisions enter the native
// workflow; libjxl is used only to independently decode the completed file.
#include "codec/modular/profile.h"
#include "codestream/modular/reference_workflow.h"
#include "io/pnm.h"
#include "modular_reference.h"
#include <bit>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
namespace reference = gjxl::test::modular_reference;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
struct Timer {
  std::chrono::steady_clock::time_point wall = std::chrono::steady_clock::now();
  std::clock_t cpu = std::clock();
  void Write(std::ostream &out) const {
    out << "{\"cpu\":" << double(std::clock() - cpu) / CLOCKS_PER_SEC
        << ",\"wall\":"
        << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         wall)
               .count()
        << '}';
  }
};
void Transforms(std::ostream &out, const TransformSequence &transforms) {
  out << '[';
  for (size_t i = 0; i < transforms.size; ++i) {
    if (i)
      out << ',';
    const auto &t = transforms.entries[i];
    Check(t.kind != TransformKind::kSqueeze, "Unexpected responsive transform");
    out << "{\"id\":" << (t.kind == TransformKind::kRct ? 0 : 1)
        << ",\"begin_c\":" << unsigned(t.begin);
    if (t.kind == TransformKind::kRct)
      out << ",\"rct_type\":" << unsigned(t.rct_type);
    else
      out << ",\"num_c\":" << unsigned(t.count) << ",\"nb_colors\":" << t.colors
          << ",\"nb_deltas\":0,\"predictor\":0,\"ordered_palette\":"
          << t.palette_ordered << ",\"lossy_palette\":false";
    out << '}';
  }
  out << ']';
}
template <class T> void Array(std::ostream &out, const T &array) {
  out << '[';
  for (size_t i = 0; i < array.size(); ++i) {
    if (i)
      out << ',';
    out << +array[i];
  }
  out << ']';
}
void Stats(std::ostream &out, const ReferenceWorkflowStats &s) {
  out << "{\"local_trees\":" << s.local_trees
      << ",\"selection_calls\":" << s.selection_calls
      << ",\"learning_calls\":" << s.learning_calls << ",\"fits\":" << s.fits
      << ",\"empty_learning_calls\":" << s.empty_learning_calls
      << ",\"scanned_pixels\":" << s.scanned_pixels
      << ",\"selected_samples\":" << s.selected_samples
      << ",\"distinct_samples\":" << s.distinct_samples
      << ",\"tree_nodes\":" << s.tree_nodes
      << ",\"maximum_stage_bytes\":" << s.maximum_stage_bytes << '}';
}
void Dump(const std::filesystem::path &directory,
          const ReferenceModularFrame &frame) {
  static_assert(std::endian::native == std::endian::little);
  std::ofstream out(directory / "workflow.json"),
      raw(directory / "pixels.i32le", std::ios::binary);
  out.exceptions(std::ios::failbit | std::ios::badbit);
  raw.exceptions(std::ios::failbit | std::ios::badbit);
  out << std::setprecision(17) << std::boolalpha << "{\"schema\":1,\"stats\":";
  Stats(out, frame.stats());
  out << ",\"streams\":[";
  size_t offset = 0;
  const auto views = frame.view().streams;
  for (size_t i = 0; i < frame.streams().size(); ++i) {
    if (i)
      out << ',';
    const auto &s = frame.streams()[i];
    const auto &v = views[i];
    out << "{\"stream_id\":" << s.id << ",\"width\":" << s.learning_extent.width
        << ",\"height\":" << s.learning_extent.height
        << ",\"metadata_channels\":" << s.metadata_channels
        << ",\"coded_channels\":" << v.channels.size()
        << ",\"local_tree\":" << (v.local_tree != nullptr)
        << ",\"wp_mode\":" << unsigned(s.selection.wp_mode)
        << ",\"wp_coefficients\":";
    Array(out, v.weighted.coefficients);
    out << ",\"wp_weights\":";
    Array(out, v.weighted.weights);
    out << ",\"transforms\":";
    Transforms(out, s.selection.transforms);
    out << ",\"channels\":[";
    bool included = s.learning_extent.width && s.learning_extent.height;
    for (size_t c = 0; c < s.channel_count; ++c) {
      const auto ch = s.channels[c];
      const auto &d = ch.descriptor;
      if (c)
        out << ',';
      if (c >= s.metadata_channels &&
          (d.extent.width > 256 || d.extent.height > 256))
        included = false;
      const size_t bytes =
          included ? d.extent.width * d.extent.height * sizeof(int32_t) : 0;
      // Native metadata represents the reference's -1 shifts by its role.
      out << "{\"width\":" << d.extent.width
          << ",\"height\":" << d.extent.height
          << ",\"hshift\":" << (c < s.metadata_channels ? -1 : int(d.hshift))
          << ",\"vshift\":" << (c < s.metadata_channels ? -1 : int(d.vshift))
          << ",\"included\":" << included << ",\"offset\":" << offset
          << ",\"bytes\":" << bytes << '}';
      if (bytes)
        for (size_t y = 0; y < d.extent.height; ++y)
          raw.write(reinterpret_cast<const char *>(ch.Row(y)),
                    d.extent.width * sizeof(int32_t));
      offset += bytes;
    }
    const auto &l = s.learning;
    const auto &p = l.preparation;
    out << "],\"learning\":{\"start\":" << l.first_group
        << ",\"stop\":" << l.stop_group
        << ",\"scanned_pixels\":" << p.scanned_pixels
        << ",\"population\":" << p.population
        << ",\"selected_samples\":" << p.selected_samples
        << ",\"quantization_samples\":" << p.quantization_samples
        << ",\"channels\":" << p.channels
        << ",\"distinct_samples\":" << l.distinct_samples << "},\"tree\":[";
    if (l.stop_group)
      for (size_t n = 0; n < s.tree.size; ++n) {
        if (n)
          out << ',';
        const auto &node = s.tree.nodes[n];
        out << '[' << node.property << ',' << node.split << ',' << node.left
            << ',' << node.right << ',' << unsigned(node.predictor) << ','
            << node.offset << ',' << node.multiplier << ']';
      }
    out << "]}";
  }
  out << "],\"pixel_bytes\":" << offset << "}\n";
  out.close();
  raw.close();
}
reference::IntegerImage Unpack(const io::IntegerImage &input,
                               const ModularInputProfile &profile) {
  reference::IntegerImage image{input.extent.width,
                                input.extent.height,
                                uint32_t(profile.channel_count),
                                profile.metadata.bits,
                                {}};
  image.samples.reserve(image.width * image.height * image.channels);
  for (size_t y = 0; y < image.height; ++y)
    for (size_t x = 0; x < image.width * image.channels; ++x) {
      const size_t pos = y * input.row_stride + x * profile.bytes_per_sample;
      image.samples.push_back(profile.bytes_per_sample == 1
                                  ? input.bytes[pos]
                                  : (uint16_t(input.bytes[pos]) << 8) |
                                        input.bytes[pos + 1]);
    }
  return image;
}
} // namespace
int main(int argc, char **argv) try {
  Check(
      argc == 6,
      "reference_workflow_check INPUT OUT auto|buffered best|all trace|timing");
  const std::string buffering = argv[3], predictors = argv[4], mode = argv[5];
  Check(buffering == "auto" || buffering == "buffered",
        "Invalid buffering mode");
  Check(predictors == "best" || predictors == "all", "Invalid predictor mode");
  Check(mode == "trace" || mode == "timing", "Invalid recording mode");
  const std::filesystem::path directory = argv[2];
  Check(!std::filesystem::exists(directory), "Output directory already exists");
  std::filesystem::create_directories(directory);
  io::IntegerImage input;
  Ok(io::ReadPnm(argv[1], &input));
  const PackedModularImageView view{
      input.bytes, input.extent, input.row_stride,
      static_cast<PackedModularFormat>(input.format),
      SampleByteOrder::kBigEndian};
  ModularInputProfile input_profile;
  Ok(ResolveModularInput(view.extent, view.format, &input_profile));
  const auto original = Unpack(input, input_profile);
  ReferenceWorkflowOptions options;
  options.buffering = buffering == "auto" ? ReferenceBuffering::kAutomatic
                                          : ReferenceBuffering::kBuffered;
  options.predictors = predictors == "best" ? ReferencePredictors::kBest
                                            : ReferencePredictors::kAll;
  // The qualification budget covers native managed backing. Source bytes,
  // trace/decode buffers, stack, and allocator bookkeeping are reported as
  // excluded; this is not a process RSS limit.
  ResourceBudget budget(size_t{8} << 30);
  ModularProfile profile;
  ProfileSession session(&profile);
  ReferenceModularFrame frame;
  std::ostringstream preparation, coding, total;
  preparation << std::setprecision(17);
  coding << std::setprecision(17);
  total << std::setprecision(17);
  const Timer whole;
  {
    Timer timer;
    Ok(PrepareReferenceModularFrame(view, options, budget, &frame));
    timer.Write(preparation);
  }
  const auto prepared_stats = frame.stats();
  codestream_internal::CodestreamBuffer bytes;
  size_t coding_stage_bytes = 0;
  {
    Timer timer;
    ModularWorkflowStoragePlan plan;
    Ok(ComputePreparedModularStreamsStoragePlan(frame.view(), options.entropy,
                                                &plan));
    coding_stage_bytes = plan.working.peak_bytes;
    ResourceReservation reservation;
    Ok(budget.TryReserve(coding_stage_bytes, &reservation));
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    Ok(EncodePreparedModularStreams(frame.view(), options.entropy, &bytes));
    timer.Write(coding);
  }
  // Trace output requires retained prepared data. Timing runs release it
  // within the total interval, as the complete native wrapper does. Source
  // IO/unpacking for the independent decoder and all verification stay out.
  if (mode == "timing") {
    frame = {};
    whole.Write(total);
  }
  Check(reference::DecodeLossless(bytes.view()) == original,
        "Decoded source integers differ");
  if (mode == "trace")
    Dump(directory, frame);
  std::ofstream file(directory / "output.jxl", std::ios::binary);
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.write(reinterpret_cast<const char *>(bytes.view().data()), bytes.size());
  file.close();
  const auto snapshot = budget.snapshot();
  const size_t encoded_size = bytes.size();
  frame = {};
  bytes = {};
  const auto released = budget.snapshot();
  Check(!released.committed_bytes() && !released.total.backing_count &&
            !released.total.pending_count && !released.open_reservations,
        "Native workflow leaked managed capacity");
  std::ofstream result(directory / "result.json");
  result.exceptions(std::ios::failbit | std::ios::badbit);
  result << std::setprecision(17) << std::boolalpha
         << "{\"schema\":2,\"buffering\":\"" << buffering
         << "\",\"predictors\":\"" << predictors
         << "\",\"exact_decode\":true,\"bytes\":" << encoded_size
         << ",\"stats\":";
  Stats(result, prepared_stats);
  result << ",\"coding_stage_bytes\":" << coding_stage_bytes
         << ",\"peak_backing_bytes\":" << snapshot.peak_backing_bytes
         << ",\"peak_committed_bytes\":" << snapshot.peak_committed_bytes
         << ",\"prepare\":" << preparation.str()
         << ",\"encode_prepared\":" << coding.str() << ",\"profile_wall\":";
  // This unadmitted private control stays serial; preserve schema 2's eight
  // stage entries when new parallel diagnostics are appended to the profile.
  Array(result, std::span(profile.seconds).first(8));
  result << ",\"profile_cpu\":";
  Array(result, std::span(profile.cpu_seconds).first(8));
  result << ",\"total\":" << (mode == "timing" ? total.str() : "null")
         << ",\"total_includes_prepared_teardown\":" << (mode == "timing");
  result << "}\n";
  result.close();
  std::cout << directory.string() << ": " << encoded_size << " bytes; "
            << prepared_stats.fits << " fits; exact decode\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
