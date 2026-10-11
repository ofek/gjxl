// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/search.h"
#include "codestream/modular/workflow.h"
#include "io/pnm.h"
#include "modular_reference.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace gjxl;
using namespace gjxl::modular_internal;
namespace ref = gjxl::test::modular_reference;
void Ok(const Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
void Policy(const ModularCodingPolicy &policy) {
  std::cout << "{\"rct\":" << unsigned(policy.rct) << ",\"transforms\":[";
  for (size_t i = 0; i < policy.transforms.size; ++i) {
    const auto &t = policy.transforms.entries[i];
    std::cout << (i ? "," : "") << '[' << int(t.kind) << ',' << unsigned(t.begin) << ','
              << unsigned(t.count) << ',' << t.colors << ',' << t.horizontal << ',' << t.in_place
              << ']';
  }
  std::cout << "],\"nodes\":[";
  for (size_t i = 0; i < policy.tree.size; ++i) {
    const auto &n = policy.tree.nodes[i];
    std::cout << (i ? "," : "") << '[' << n.property << ',' << n.split << ',' << n.left << ','
              << n.right << ',' << int(n.predictor) << ',' << n.offset << ',' << n.multiplier
              << ']';
  }
  std::cout << "]}";
}
int main(int argc, char **argv) try {
  if (argc != 3)
    throw std::runtime_error("training_runner INPUT OUTPUT_DIRECTORY");
  io::IntegerImage image;
  Ok(io::ReadPnm(argv[1], &image));
  PackedModularImageView view{image.bytes, image.extent, image.row_stride,
                              static_cast<PackedModularFormat>(image.format),
                              SampleByteOrder::kBigEndian};
  ModularInputProfile input;
  Ok(ResolveModularInput(image.extent, view.format, &input));
  const unsigned bits = input.metadata.bits, channels = input.channel_count;
  ref::IntegerImage original{image.extent.width, image.extent.height, channels, bits, {}};
  for (size_t y = 0; y < image.extent.height; ++y)
    for (size_t x = 0; x < image.extent.width * channels; ++x) {
      const size_t i = y * image.row_stride + x * (bits / 8);
      original.samples.push_back(bits == 8 ? image.bytes[i]
                                           : (uint16_t{image.bytes[i]} << 8) | image.bytes[i + 1]);
    }
  std::filesystem::create_directories(argv[2]);
  auto report = [&](const std::string &config, const std::vector<uint8_t> &encoded,
                    const ModularCodingPolicy &policy, double seconds, TrainingCoverage coverage,
                    size_t candidates, size_t baseline_bytes) {
    if (ref::DecodeLossless(encoded) != original)
      throw std::runtime_error("Decode mismatch");
    const auto path = std::filesystem::path(argv[2]) / (config + ".jxl");
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write(reinterpret_cast<const char *>(encoded.data()), encoded.size());
    out.close();
    std::cout << std::setprecision(17) << "{\"config\":\"" << config
              << "\",\"encoded_bytes\":" << encoded.size()
              << ",\"retained_bytes\":" << std::min(encoded.size(), baseline_bytes)
              << ",\"seconds\":" << seconds
              << ",\"decoded_exact\":true,\"candidates\":" << candidates
              << ",\"coverage\":{\"population\":" << coverage.population
              << ",\"samples\":" << coverage.samples << ",\"slices\":" << coverage.slices
              << ",\"sampled_slices\":" << coverage.sampled_slices
              << ",\"minimum\":" << coverage.minimum_slice_samples
              << ",\"maximum\":" << coverage.maximum_slice_samples
              << ",\"floor\":" << coverage.slice_floor << ",\"quota_hash\":" << coverage.quota_hash
              << "},\"policy\":";
    Policy(policy);
    std::cout << "}" << std::endl;
  };
  using Clock = std::chrono::steady_clock;
  auto start = Clock::now();
  gjxl::modular_internal::ModularEncodingOptions options;
  options.search = true;
  options.entropy = EntropyCodingMode::kAns;
  ModularProfile profile;
  std::vector<uint8_t> baseline;
  {
    ProfileSession session(&profile);
    Ok(EncodeModularImage(view, options, &baseline));
  }
  report("baseline", baseline, profile.resolved,
         std::chrono::duration<double>(Clock::now() - start).count(), {},
         profile.encoded_candidates, baseline.size());
  const auto fixed = profile.resolved;
  for (auto mode : {TreeLearningMode::kLegacy, TreeLearningMode::kQuantized})
    for (size_t samples : {size_t{4096}, size_t{65536}})
      for (size_t leaves : {size_t{16}, size_t{64}}) {
        const std::string config =
            std::string(mode == TreeLearningMode::kLegacy ? "legacy" : "quantized") + "-" +
            std::to_string(samples) + "-" + std::to_string(leaves);
        start = Clock::now();
        ModularCodingPolicy single, split, learned;
        TrainingCoverage coverage;
        {
          ModularEncoderFrame frame;
          Ok(ModularEncoderFrame::Prepare(view, fixed, &frame));
          ChannelShape shape;
          Ok(DescribeImage(frame.image(), &shape));
          ModularFrameGeometry geometry;
          Ok(ModularFrameGeometry::Create(image.extent, &geometry));
          ModularStreamPlan layout;
          Ok(BuildModularStreamPlan(geometry, shape.channels(), shape.metadata, &layout));
          Ok(LearnModularPolicies(frame, layout, fixed, &single, &split, &learned,
                                  {samples, leaves, mode}, &coverage));
        }
        if (learned.rct != fixed.rct || learned.transforms != fixed.transforms ||
            learned.weighted != fixed.weighted)
          throw std::runtime_error("Frozen transform or predictor settings changed");
        options.search = false;
        options.coding = learned;
        std::vector<uint8_t> encoded;
        Ok(EncodeModularImage(view, options, &encoded));
        report(config, encoded, learned,
               std::chrono::duration<double>(Clock::now() - start).count(), coverage, 1,
               baseline.size());
      }
  for (auto mode : {TreeLearningMode::kLegacy, TreeLearningMode::kQuantized}) {
    start = Clock::now();
    options.search = true;
    options.coding = {};
    options.learning = {65536, 64, mode};
    ModularProfile searched;
    std::vector<uint8_t> encoded;
    {
      ProfileSession session(&searched);
      Ok(EncodeModularImage(view, options, &encoded));
    }
    if (encoded.size() > baseline.size())
      throw std::runtime_error("Retained search regressed");
    report(mode == TreeLearningMode::kLegacy ? "search-legacy" : "search-quantized", encoded,
           searched.resolved, std::chrono::duration<double>(Clock::now() - start).count(), {},
           searched.encoded_candidates, baseline.size());
  }
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
