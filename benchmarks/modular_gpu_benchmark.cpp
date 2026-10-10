// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
// Paired prescribed-policy CPU/Metal measurement. All I/O and comparisons are
// outside timing. Fresh output for each call; device setup is recorded separately.
#include "codec/modular/profile.h"
#include "codestream/modular/gpu_workflow.h"
#include "gpu/metal/metal_backend.h"
#include "io/pnm.h"
#if GJXL_MODULAR_BENCHMARK_REFERENCE
#include "modular_reference.h"
#endif
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
void Check(bool v, const char *m) {
  if (!v)
    throw std::runtime_error(m);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
size_t Number(std::string_view s) {
  size_t n = 0;
  auto result = std::from_chars(s.data(), s.data() + s.size(), n);
  Check(result.ec == std::errc{} && result.ptr == s.data() + s.size(), "Invalid unsigned integer");
  return n;
}
uint64_t Nanoseconds(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
void Write(const fs::path &path, std::span<const uint8_t> bytes) {
  std::ofstream file(path, std::ios::binary);
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.write(reinterpret_cast<const char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
}
std::vector<uint8_t> Read(const fs::path &path) {
  std::ifstream file(path, std::ios::binary);
  file.exceptions(std::ios::badbit);
  Check(file.good(), "Missing qualified expected file");
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
uint64_t Fingerprint(std::span<const uint8_t> bytes) {
  uint64_t hash = 14695981039346656037ull;
  for (auto byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}
struct Config {
  std::string name;
  fs::path path;
  uint8_t rct = 0;
  EntropyCodingMode entropy = EntropyCodingMode::kPrefix;
  size_t threads = 1, jobs = 1;
  SampleByteOrder order = SampleByteOrder::kLittleEndian;
  size_t padding = 0;
};
std::vector<Config> Configs(const fs::path &path) {
  std::ifstream file(path);
  Check(file.good(), "Cannot open benchmark manifest");
  std::vector<Config> configs;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream row(line);
    std::vector<std::string> fields;
    std::string field;
    while (std::getline(row, field, '\t'))
      fields.push_back(field);
    Check(fields.size() == 8, "Manifest requires name/path/rct/coder/threads/jobs/order/padding");
    Check(!fields[0].empty() && fields[0].find_first_not_of(
                                    "abcdefghijklmnopqrstuvwxyz0123456789-_") == std::string::npos,
          "Invalid case name");
    Check(Number(fields[2]) < 42 && (fields[3] == "prefix" || fields[3] == "ans") &&
              (fields[6] == "little" || fields[6] == "big"),
          "Invalid policy/order");
    Config c{fields[0],
             fields[1],
             static_cast<uint8_t>(Number(fields[2])),
             fields[3] == "prefix" ? EntropyCodingMode::kPrefix : EntropyCodingMode::kAns,
             Number(fields[4]),
             Number(fields[5]),
             fields[6] == "little" ? SampleByteOrder::kLittleEndian : SampleByteOrder::kBigEndian,
             Number(fields[7])};
    Check((c.threads == 1 || c.threads == 4) && (c.jobs == 1 || c.jobs == 4) &&
              (c.jobs == 1 || c.threads == 1) && c.padding <= (1u << 20),
          "Invalid workload bounds");
    for (const auto &old : configs)
      Check(old.name != c.name, "Duplicate case name");
    configs.push_back(std::move(c));
  }
  Check(!configs.empty(), "Empty benchmark manifest");
  return configs;
}
struct Fixture {
  io::IntegerImage source;
  ModularInputProfile profile;
  std::vector<uint8_t> bytes;
  size_t stride;
  SampleByteOrder order;
  PackedModularFormat format;
  explicit Fixture(const Config &config) : order(config.order) {
    Ok(io::ReadPnm(config.path, &source));
    format = static_cast<PackedModularFormat>(source.format);
    Ok(ResolveModularInput(source.extent, format, &profile));
    const size_t row = source.extent.width * profile.channel_count * profile.bytes_per_sample;
    stride = row + config.padding;
    size_t size = 0;
    Ok(ComputePackedModularInputSize(source.extent, format, order, stride, &size));
    bytes.assign(3 + size, 0xad);
    for (size_t y = 0; y < source.extent.height; ++y) {
      auto *dst = bytes.data() + 3 + y * stride;
      const auto *src = source.bytes.data() + y * source.row_stride;
      std::copy_n(src, row, dst);
      if (profile.bytes_per_sample == 2 && order == SampleByteOrder::kLittleEndian)
        for (size_t x = 0; x < row; x += 2)
          std::swap(dst[x], dst[x + 1]);
    }
  }
  PackedModularImageView view() const {
    return {std::span(bytes).subspan(3), source.extent, stride, format, order};
  }
  void Decode(std::span<const uint8_t> encoded) const {
#if GJXL_MODULAR_BENCHMARK_REFERENCE
    const auto decoded = test::modular_reference::DecodeLossless(encoded);
    Check(decoded.width == source.extent.width && decoded.height == source.extent.height &&
              decoded.channels == profile.channel_count && decoded.bits == profile.metadata.bits,
          "Pinned decoded profile differs");
    for (size_t y = 0; y < source.extent.height; ++y)
      for (size_t x = 0; x < source.extent.width * profile.channel_count; ++x) {
        const auto *p = source.bytes.data() + y * source.row_stride + x * profile.bytes_per_sample;
        const uint16_t expected =
            profile.bytes_per_sample == 1 ? p[0] : (uint16_t{p[0]} << 8) | p[1];
        Check(decoded.samples[y * source.extent.width * profile.channel_count + x] == expected,
              "Pinned decoded sample differs");
      }
#else
    (void)encoded;
#endif
  }
};
struct Sample {
  uint64_t ns = 0;
  size_t bytes = 0, capacity = 0, peak = 0, participants = 0;
  uint64_t submissions = 0, allocations = 0, fingerprint = 0;
  ModularProfile profile;
};
struct Workload {
  const Config &config;
  const Fixture &fixture;
  GpuBackend &backend;
  modular_internal::ModularEncodingOptions options;
  GpuModularWorkflowStoragePlan plan;
  size_t capacity;
  Workload(const Config &c, const Fixture &f, GpuBackend &gpu)
      : config(c), fixture(f), backend(gpu) {
    options.entropy = c.entropy;
    options.cpu_thread_count = c.threads;
    options.coding.rct = c.rct;
    Check(QueryGpuModularInputPreparation(gpu) != nullptr, "Backend has no Modular capability");
    Ok(ComputeGpuModularWorkflowStoragePlan(
        f.source.extent, f.stride, f.format, f.order, options,
        QueryGpuModularInputPreparation(gpu)->modular_input_limits(), &plan));
    Check(plan.peak_bytes <= SIZE_MAX / c.jobs, "Batch capacity overflow");
    capacity = plan.peak_bytes * c.jobs;
  }
  Sample Run(bool gpu, bool profile, std::vector<uint8_t> *result) const {
    Check(!profile || config.jobs == 1, "Stage profiles require one calling thread");
    auto opts = options;
    Ok(ExecutionDomain::Create({capacity, 4}, &opts.execution_domain));
    std::array<std::vector<uint8_t>, 4> output;
    std::array<Status, 4> statuses;
    Sample sample;
    const auto stats = backend.stats();
    // Session installation/domain/output construction precede the timing boundary.
    ProfileSession session(profile ? &sample.profile : nullptr);
    const auto start = Clock::now();
    auto encode = [&](size_t i) {
      statuses[i] = gpu ? EncodeGpuModularImage(backend, fixture.view(), opts, &output[i])
                        : EncodeModularImage(fixture.view(), opts, &output[i]);
    };
    if (config.jobs == 1)
      encode(0);
    else {
      std::array<std::jthread, 4> workers;
      for (size_t i = 0; i < workers.size(); ++i)
        workers[i] = std::jthread([&, i] { encode(i); });
      for (auto &worker : workers)
        worker.join();
    }
    sample.ns = Nanoseconds(start);
    // Status checks, output equality, counters, hashing and destruction are untimed.
    for (size_t i = 0; i < config.jobs; ++i) {
      Ok(statuses[i]);
      Check(output[i] == output[0], "Concurrent output differs");
    }
    const auto resources = opts.execution_domain->snapshot();
    Check(resources.live_capacity_bytes == 0 && resources.reserved_unbacked_bytes == 0 &&
              resources.active_reservations == 0 && resources.active_cpu_participants == 0 &&
              resources.peak_backing_bytes <= capacity && resources.peak_cpu_participants <= 4,
          "Benchmark call leaked or exceeded its resource envelope");
    sample.submissions = backend.stats().committed_submissions - stats.committed_submissions;
    sample.allocations = backend.stats().successful_allocations - stats.successful_allocations;
    Check(sample.submissions == (gpu ? config.jobs : 0) &&
              sample.allocations == (gpu ? 2 * config.jobs : 0),
          "Benchmark took an unexpected GPU path");
    sample.bytes = output[0].size();
    sample.capacity = capacity;
    sample.peak = resources.peak_backing_bytes;
    sample.participants = resources.peak_cpu_participants;
    sample.fingerprint = Fingerprint(output[0]);
    *result = std::move(output[0]);
    if (profile) {
      Check(sample.profile.encoded_candidates == 1 && sample.profile.resolved == options.coding,
            "Profile changed the prescribed policy");
      double sum = 0;
      for (double s : sample.profile.seconds)
        sum += s;
      Check(sum * 1e9 <= static_cast<double>(sample.ns), "Profile scopes exceed full-call time");
    }
    return sample;
  }
};
void Row(std::ostream &out, const Config &c, const Fixture &f, std::string_view kind, size_t round,
         bool cpu_first, bool gpu, const Sample &s) {
  out << c.name << ',' << kind << ',' << round << ',' << cpu_first << ',' << (gpu ? "gpu" : "cpu")
      << ',' << f.source.extent.width << ',' << f.source.extent.height << ',' << int(f.format)
      << ',' << int(c.rct) << ',' << int(c.entropy) << ',' << c.threads << ',' << c.jobs << ','
      << f.stride << ',' << int(c.order) << ',' << s.ns << ',' << s.bytes << ',' << s.capacity
      << ',' << s.peak << ',' << s.participants << ',' << s.submissions << ',' << s.allocations
      << ',' << s.fingerprint;
  for (double seconds : s.profile.seconds)
    out << ',' << static_cast<uint64_t>(seconds * 1e9);
  out << '\n';
}
} // namespace
int main(int argc, char **argv) try {
  Check(argc == 7, "Usage: benchmark qualify|measure|profile MANIFEST OUTDIR ROUNDS WARMUPS SEED");
  const std::string mode = argv[1];
  Check(mode == "qualify" || mode == "measure" || mode == "profile", "Invalid mode");
  const size_t rounds = Number(argv[4]), warmups = Number(argv[5]), seed = Number(argv[6]);
  Check(rounds > 0 && rounds <= 100 && warmups <= 10, "Invalid measurement counts");
  auto configs = Configs(argv[2]);
  const fs::path outdir = argv[3];
  fs::create_directories(outdir / "expected");
  std::ofstream csv(outdir / (mode + "-" + std::to_string(seed) + ".csv"));
  csv.exceptions(std::ios::failbit | std::ios::badbit);
  csv << "case,kind,round,cpu_first,backend,width,height,format,rct,entropy,threads,jobs,stride,"
         "byte_order,ns,bytes,domain_capacity,peak_backing,cpu_peak,submissions,allocations,"
         "fingerprint,input_ns,transforms_ns,training_ns,tokens_ns,model_ns,emission_ns,assembly_"
         "ns\n";
  std::unique_ptr<GpuBackend> gpu;
  const auto start = Clock::now();
  Ok(CreateEmbeddedMetalBackend({}, &gpu));
  const uint64_t setup = Nanoseconds(start);
  std::ofstream setup_file(outdir / (mode + "-" + std::to_string(seed) + "-setup.txt"));
  setup_file << "backend=" << gpu->name() << "\nbackend_setup_ns=" << setup << '\n'
             << "reference=" << bool(GJXL_MODULAR_BENCHMARK_REFERENCE) << '\n';
  if (mode != "qualify")
    std::shuffle(configs.begin(), configs.end(), std::mt19937(seed));
  for (size_t index = 0; index < configs.size(); ++index) {
    const auto &config = configs[index];
    if (mode == "profile" && config.jobs != 1)
      continue;
    const Fixture fixture(config);
    const Workload work(config, fixture, *gpu);
    const auto expected_path = outdir / "expected" / (config.name + ".jxl");
    const auto expected = mode == "qualify" ? std::vector<uint8_t>{} : Read(expected_path);
    for (size_t round = 0; round < (mode == "qualify" ? 1 : warmups + rounds); ++round) {
      const bool cpu_first = (round + seed + index) % 2 == 0;
      std::array<std::vector<uint8_t>, 2> output;
      std::array<Sample, 2> samples;
      for (size_t position = 0; position < 2; ++position) {
        const bool is_gpu = cpu_first ? position == 1 : position == 0;
        samples[is_gpu] = work.Run(is_gpu, mode == "profile", &output[is_gpu]);
      }
      Check(output[0] == output[1] && (mode == "qualify" || output[0] == expected),
            "CPU/GPU/profiled/timed bytes differ from qualified output");
      if (mode == "qualify") {
        fixture.Decode(output[0]);
        Write(expected_path, output[0]);
      }
      const auto kind = mode == "qualify" ? "qualify" : round < warmups ? "warmup" : mode.c_str();
      for (bool is_gpu : {false, true})
        Row(csv, config, fixture, kind, round < warmups ? round : round - warmups, cpu_first,
            is_gpu, samples[is_gpu]);
    }
    csv.flush();
    std::cout << mode << ' ' << config.name << " passed\n" << std::flush;
  }
  const auto budget = resource_budget_internal::DefaultResourceBudget().snapshot();
  Check(budget.committed_bytes() == 0, "Benchmark leaked default resource charges");
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
