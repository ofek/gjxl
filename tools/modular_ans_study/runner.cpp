// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/ans_internal.h"
#include "codestream/modular/search.h"
#include "codestream/modular/workflow.h"
#include "codestream/profile_internal.h"
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
using namespace gjxl::codestream_internal;
using Clock = std::chrono::steady_clock;
namespace ref = gjxl::test::modular_reference;
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
void Require(bool v, const char *message) {
  if (!v)
    throw std::runtime_error(message);
}
const char *Name(size_t mode) { return mode ? "estimated" : "exact"; }
AnsSelectionMode Selection(size_t mode) {
  return mode ? AnsSelectionMode::kEstimated : AnsSelectionMode::kExact;
}
void PolicyJson(const ModularCodingPolicy &policy) {
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
ModularCodingPolicy ReadPolicy(const std::string &path, size_t channels) {
  ModularCodingPolicy policy;
  if (path == "gradient") {
    policy.rct = channels >= 3 ? 6 : 0;
    return policy;
  }
  std::ifstream in(path);
  in.exceptions(std::ios::badbit | std::ios::failbit);
  unsigned rct;
  in >> rct >> policy.transforms.size >> policy.tree.size;
  Require(rct < 42 && policy.transforms.size <= kMaximumTransforms && policy.tree.size &&
              policy.tree.size <= kMaximumTreeNodes,
          "Invalid saved policy dimensions");
  policy.rct = rct;
  for (size_t i = 0; i < policy.transforms.size; ++i) {
    auto &t = policy.transforms.entries[i];
    unsigned kind, begin, count;
    in >> kind >> begin >> count >> t.colors >> t.horizontal >> t.in_place;
    Require(kind <= 1 && begin <= 255 && count <= 255, "Invalid transform in saved policy");
    t.kind = static_cast<TransformKind>(kind);
    t.begin = begin;
    t.count = count;
  }
  policy.tree.nodes.resize(policy.tree.size);
  for (size_t i = 0; i < policy.tree.size; ++i) {
    auto &n = policy.tree.nodes[i];
    unsigned predictor;
    in >> n.property >> n.split >> n.left >> n.right >> predictor >> n.offset >> n.multiplier;
    Require(predictor < 14, "Invalid saved predictor");
    n.predictor = static_cast<Predictor>(predictor);
  }
  Ok(ValidateCodingPolicy(policy));
  return policy;
}
int main(int argc, char **argv) try {
  if (argc != 9)
    throw std::runtime_error(
        "ans_runner INPUT OUT fixed|search POLICY REPEATS WARMUPS THREADS ORDER_SEED");
  const std::filesystem::path out = argv[2];
  const std::string phase = argv[3];
  const size_t repeats = std::stoul(argv[5]), warmups = std::stoul(argv[6]);
  const size_t threads = std::stoul(argv[7]), seed = std::stoul(argv[8]);
  Require((phase == "fixed" || phase == "search") && repeats && threads && threads <= 4,
          "Invalid study options");
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
  std::filesystem::create_directories(out);
  gjxl::modular_internal::ModularEncodingOptions options;
  options.entropy = EntropyCodingMode::kAns;
  options.cpu_thread_count = threads;
  options.search = phase == "search";
  options.learning = {262144, 256};
  options.tree_checkpoints = options.search;
  if (!options.search)
    options.coding = ReadPolicy(argv[4], channels);
  std::array<std::vector<uint8_t>, 2> expected;
  std::array<ModularCodingPolicy, 2> resolved;
  std::cout << std::setprecision(17);
  // Search needs its diagnostic profile to retain the resolved policy. Fixed
  // encode timing leaves profiling disabled, with separate diagnostic calls.
  for (size_t pair = 0; pair < repeats + warmups + (options.search ? 0 : 1); ++pair) {
    const bool diagnostic = !options.search && pair == repeats + warmups;
    for (size_t position = 0; position < 2; ++position) {
      const size_t mode = (pair + position + seed) % 2;
      options.ans_selection = Selection(mode);
      ModularProfile profile;
      std::vector<uint8_t> encoded;
      const auto start = Clock::now();
      {
        ProfileSession session(options.search || diagnostic ? &profile : nullptr);
        Ok(EncodeModularImage(view, options, &encoded));
      }
      const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
      if (expected[mode].empty()) {
        Require(ref::DecodeLossless(encoded) == original, "Complete file decoded incorrectly");
        expected[mode] = encoded;
        std::ofstream file(out / (std::string(Name(mode)) + ".jxl"), std::ios::binary);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.write(reinterpret_cast<const char *>(encoded.data()), encoded.size());
      }
      Require(expected[mode] == encoded, "Repeated or profiled encode changed bytes");
      if (options.search || diagnostic)
        resolved[mode] = profile.resolved;
      std::cout << "{\"kind\":\"encode\",\"mode\":\"" << Name(mode) << "\",\"pair\":" << pair
                << ",\"position\":" << position
                << ",\"warmup\":" << (pair < warmups ? "true" : "false")
                << ",\"diagnostic\":" << (diagnostic ? "true" : "false")
                << ",\"profiled\":" << (options.search || diagnostic ? "true" : "false")
                << ",\"seconds\":" << seconds << ",\"bytes\":" << encoded.size()
                << ",\"decoded_exact\":true,\"candidates\":" << profile.encoded_candidates
                << ",\"stages\":[";
      for (size_t i = 0; i < profile.seconds.size(); ++i)
        std::cout << (i ? "," : "") << profile.seconds[i];
      std::cout << "],\"policy\":";
      PolicyJson(options.search || diagnostic ? profile.resolved : options.coding);
      std::cout << "}" << std::endl;
    }
  }
  if (options.search)
    return 0;
  Require(resolved[0] == options.coding && resolved[1] == options.coding,
          "Prescribed policy changed");
  ModularEncoderFrame frame;
  Ok(ModularEncoderFrame::Prepare(view, options.coding, &frame));
  ChannelShape shape;
  Ok(DescribeImage(frame.image(), &shape));
  ModularFrameGeometry geometry;
  Ok(ModularFrameGeometry::Create(image.extent, &geometry));
  ModularStreamPlan layout;
  Ok(BuildModularStreamPlan(geometry, shape.channels(), shape.metadata, &layout));
  PreparedModularTokens tokens;
  Ok(TokenizeModular(frame, layout, options.coding, &tokens, threads));
  frame = {};
  EntropyCode prefix;
  Ok(OptimizeEntropyCode(tokens.streams, {.context_count = uint32_t(tokens.context_count)},
                         &prefix));
  std::array<EntropyCode, 2> models;
  uint64_t token_hash = 1469598103934665603ull;
  size_t token_count = 0;
  for (auto stream : tokens.streams) {
    token_hash = (token_hash ^ stream.size()) * 1099511628211ull;
    for (size_t i = 0; i < stream.size(); ++i) {
      token_hash = (token_hash ^ stream[i].value) * 1099511628211ull;
      token_hash = (token_hash ^ stream[i].context) * 1099511628211ull;
    }
    token_count += stream.size();
  }
  for (size_t pair = 0; pair < repeats + warmups + 1; ++pair)
    for (size_t position = 0; position < 2; ++position) {
      const size_t mode = (pair + position + seed) % 2;
      const bool diagnostic = pair == repeats + warmups;
      EntropyCode model;
      EntropyWorkProfile profile;
      const auto start = Clock::now();
      Ok(OptimizeAnsEntropyCode(tokens.streams, prefix, &model, nullptr,
                                diagnostic ? &profile : nullptr, Selection(mode)));
      const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
      if (pair == 0)
        models[mode] = model;
      Require(models[mode] == model, "Repeated ANS model differs");
      if (mode)
        Require(profile.ans_token_cost_nanoseconds == 0, "Estimated replay simulated ANS states");
      std::cout << "{\"kind\":\"replay\",\"mode\":\"" << Name(mode) << "\",\"pair\":" << pair
                << ",\"position\":" << position
                << ",\"warmup\":" << (pair < warmups ? "true" : "false")
                << ",\"diagnostic\":" << (diagnostic ? "true" : "false")
                << ",\"seconds\":" << seconds << ",\"tokens\":" << token_count
                << ",\"streams\":" << tokens.streams.size() << ",\"token_hash\":" << token_hash
                << ",\"width\":" << unsigned(model.ans_log_alpha_size)
                << ",\"clusters\":" << model.ans_histograms.size()
                << ",\"exact_simulation_ns\":" << profile.ans_token_cost_nanoseconds
                << ",\"config_candidates\":" << profile.ans_uint_config_candidate_count
                << ",\"histogram_candidates\":" << profile.ans_histogram_candidate_count
                << ",\"width_candidates\":" << profile.ans_alphabet_width_candidate_count << "}"
                << std::endl;
    }
  Require(models[0].context_map == models[1].context_map, "Replay changed the context partition");
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
