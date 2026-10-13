// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "reference_capture.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <type_traits>

namespace reference_learning {
namespace {
constexpr uint64_t kMagic = 0x31594c50524c5847ULL; // GXL RPLY1, little endian.
constexpr uint64_t kMaximumFixtureBytes = uint64_t{512} << 20;
constexpr size_t kMaximumSamples =
    size_t{16} << 20; // Research ceiling, not native admission.

void Require(bool value, const char *what) {
  if (!value)
    throw std::runtime_error(what);
}

template <class T> T ReverseBytes(T value) {
  using U = std::make_unsigned_t<T>;
  U result = 0, bits = std::bit_cast<U>(value);
  for (size_t i = 0; i < sizeof(T); ++i) {
    result = (result << 8) | (bits & 255);
    bits >>= 8;
  }
  return std::bit_cast<T>(result);
}

class Writer {
public:
  explicit Writer(const std::filesystem::path &path)
      : stream_(path, std::ios::binary) {
    stream_.exceptions(std::ios::badbit | std::ios::failbit);
  }
  template <class T> void Scalar(T value) {
    static_assert(std::is_integral_v<T>);
    if constexpr (std::endian::native != std::endian::little)
      value = ReverseBytes(value);
    stream_.write(reinterpret_cast<const char *>(&value), sizeof(value));
  }
  template <class T> void Vector(const std::vector<T> &values) {
    Scalar<uint64_t>(values.size());
    if constexpr (std::endian::native == std::endian::little ||
                  sizeof(T) == 1) {
      if (!values.empty())
        stream_.write(reinterpret_cast<const char *>(values.data()),
                      values.size() * sizeof(T));
    } else {
      for (auto value : values)
        Scalar(value);
    }
  }
  void Tokens(const std::vector<jxl::ResidualToken> &values) {
    static_assert(sizeof(jxl::ResidualToken) == 2);
    Scalar<uint64_t>(values.size());
    if (!values.empty())
      stream_.write(reinterpret_cast<const char *>(values.data()),
                    values.size() * 2);
  }
  void Close() { stream_.close(); }

private:
  std::ofstream stream_;
};

class Reader {
public:
  explicit Reader(const std::filesystem::path &path)
      : stream_(path, std::ios::binary),
        remaining_(std::filesystem::file_size(path)) {
    stream_.exceptions(std::ios::badbit | std::ios::failbit);
    Require(remaining_ <= kMaximumFixtureBytes, "Oversized replay fixture");
  }
  template <class T> T Scalar() {
    static_assert(std::is_integral_v<T>);
    T value;
    Read(&value, sizeof(value));
    if constexpr (std::endian::native != std::endian::little)
      value = ReverseBytes(value);
    return value;
  }
  size_t Count(size_t limit) {
    const uint64_t value = Scalar<uint64_t>();
    Require(value <= limit, "Replay fixture count exceeds limit");
    return static_cast<size_t>(value);
  }
  template <class T> std::vector<T> Vector(size_t limit) {
    const size_t size =
        Count(std::min<uint64_t>(limit, remaining_ / sizeof(T)));
    std::vector<T> result(size);
    if (size)
      Read(result.data(), size * sizeof(T));
    if constexpr (std::endian::native != std::endian::little && sizeof(T) > 1) {
      for (T &value : result)
        value = ReverseBytes(value);
    }
    return result;
  }
  std::vector<jxl::ResidualToken> Tokens() {
    const size_t size =
        Count(std::min<uint64_t>(kMaximumSamples, remaining_ / 2));
    std::vector<jxl::ResidualToken> result(size);
    if (size)
      Read(result.data(), size * 2);
    return result;
  }
  void End() { Require(remaining_ == 0, "Trailing bytes in replay fixture"); }

private:
  void Read(void *ptr, size_t size) {
    Require(size <= remaining_, "Truncated replay fixture");
    stream_.read(reinterpret_cast<char *>(ptr), size);
    remaining_ -= size;
  }
  std::ifstream stream_;
  uint64_t remaining_;
};

void WriteRange(Writer &w, const jxl::StaticPropRange &range) {
  for (auto pair : range)
    for (auto value : pair)
      w.Scalar(value);
}
jxl::StaticPropRange ReadRange(Reader &r) {
  jxl::StaticPropRange range;
  for (auto &pair : range)
    for (auto &value : pair)
      value = r.Scalar<uint32_t>();
  for (auto pair : range)
    Require(pair[0] < pair[1], "Empty replay static range");
  return range;
}
void WriteTree(Writer &w, const jxl::Tree &tree) {
  w.Scalar<uint64_t>(tree.size());
  for (const auto &node : tree) {
    w.Scalar(node.property);
    w.Scalar(node.splitval);
    w.Scalar(node.lchild);
    w.Scalar(node.rchild);
    w.Scalar(static_cast<uint32_t>(node.predictor));
    w.Scalar(node.predictor_offset);
    w.Scalar(node.multiplier);
  }
}
jxl::Tree ReadTree(Reader &r) {
  // Native tree limits are checked by the importer, not imposed on reference
  // captures.
  jxl::Tree tree(r.Count(size_t{1} << 22));
  Require(!tree.empty(), "Empty expected replay tree");
  for (auto &node : tree) {
    node.property = r.Scalar<int16_t>();
    node.splitval = r.Scalar<int32_t>();
    node.lchild = r.Scalar<uint32_t>();
    node.rchild = r.Scalar<uint32_t>();
    node.predictor = static_cast<jxl::Predictor>(r.Scalar<uint32_t>());
    node.predictor_offset = r.Scalar<int64_t>();
    node.multiplier = r.Scalar<uint32_t>();
    Require(node.property >= -1 && node.property < 256 &&
                static_cast<uint32_t>(node.predictor) < 14 &&
                node.multiplier != 0,
            "Invalid expected replay node");
    Require(node.property == -1 ||
                (node.lchild < tree.size() && node.rchild < tree.size()),
            "Invalid expected replay children");
  }
  return tree;
}
} // namespace
} // namespace reference_learning

namespace jxl {
// Research-only friend declared in the generated header. No layout changes.
struct ReferenceLearningSampleAccess {
  static void Write(reference_learning::Writer &w, const TreeSamples &s) {
    reference_learning::Require(s.dedup_table_.empty(),
                                "Capture requires finalized samples");
    w.Scalar<uint64_t>(s.num_samples);
    w.Scalar<uint64_t>(s.num_static_props);
    w.Vector(s.props_to_use);
    std::vector<uint32_t> predictors;
    for (auto p : s.predictors)
      predictors.push_back(static_cast<uint32_t>(p));
    w.Vector(predictors);
    w.Vector(s.sample_counts);
    for (const auto &col : s.residuals)
      w.Tokens(col);
    for (const auto &col : s.static_props)
      w.Vector(col);
    for (const auto &col : s.props)
      w.Vector(col);
    for (const auto &col : s.compact_properties)
      w.Vector(col);
    for (const auto &col : s.static_property_mapping)
      w.Vector(col);
    for (const auto &col : s.property_mapping)
      w.Vector(col);
  }
  static TreeSamples Read(reference_learning::Reader &r) {
    using namespace reference_learning;
    TreeSamples s;
    s.num_samples = r.Count(kMaximumSamples);
    s.num_static_props = r.Count(kNumStaticProperties);
    s.props_to_use = r.Vector<uint32_t>(63);
    auto predictors = r.Vector<uint32_t>(14);
    Require(!predictors.empty() && !s.props_to_use.empty() &&
                s.num_static_props <= s.props_to_use.size(),
            "Invalid replay feature dimensions");
    for (uint32_t p : predictors) {
      Require(p < 14 &&
                  std::find(s.predictors.begin(), s.predictors.end(),
                            static_cast<Predictor>(p)) == s.predictors.end(),
              "Invalid replay predictor");
      s.predictors.push_back(static_cast<Predictor>(p));
    }
    for (size_t i = 0; i < s.props_to_use.size(); ++i) {
      const auto p = s.props_to_use[i];
      Require(p < 256 &&
                  (i < s.num_static_props ? p == i : p >= kNumStaticProperties),
              "Invalid replay property");
      Require(std::find(s.props_to_use.begin(), s.props_to_use.begin() + i,
                        p) == s.props_to_use.begin() + i,
              "Duplicate replay property");
    }
    s.sample_counts = r.Vector<uint16_t>(kMaximumSamples);
    const size_t n = s.sample_counts.size();
    Require(n && n <= s.num_samples &&
                std::all_of(s.sample_counts.begin(), s.sample_counts.end(),
                            [](auto c) { return c > 0; }) &&
                std::accumulate(s.sample_counts.begin(), s.sample_counts.end(),
                                uint64_t{0}) == s.num_samples,
            "Invalid replay sample weights");
    s.residuals.resize(predictors.size());
    for (auto &col : s.residuals) {
      col = r.Tokens();
      Require(col.size() == n, "Replay token column size mismatch");
      for (auto token : col)
        Require(token.nbits <= 63, "Invalid replay extra-bit count");
    }
    for (size_t i = 0; i < kNumStaticProperties; ++i) {
      s.static_props[i] = r.Vector<uint32_t>(kMaximumSamples);
      Require(s.static_props[i].size() == (i < s.num_static_props ? n : 0),
              "Replay static column size mismatch");
    }
    s.props.resize(s.props_to_use.size() - s.num_static_props);
    for (auto &col : s.props) {
      col = r.Vector<uint8_t>(kMaximumSamples);
      Require(col.size() == n, "Replay property column size mismatch");
    }
    s.compact_properties.resize(s.props_to_use.size());
    for (size_t i = 0; i < s.compact_properties.size(); ++i) {
      auto &cuts = s.compact_properties[i];
      cuts = r.Vector<int32_t>(i < s.num_static_props ? 65535 : 255);
      Require(std::is_sorted(cuts.begin(), cuts.end()) &&
                  std::adjacent_find(cuts.begin(), cuts.end()) == cuts.end(),
              "Invalid replay dequantization cuts");
      if (i < s.num_static_props) {
        for (auto value : s.static_props[i])
          Require(value <= cuts.size(), "Static replay bin out of range");
      } else {
        for (auto value : s.props[i - s.num_static_props])
          Require(value <= cuts.size(), "Replay bin out of range");
      }
    }
    for (size_t i = 0; i < kNumStaticProperties; ++i) {
      auto &map = s.static_property_mapping[i];
      map = r.Vector<uint16_t>(1023);
      Require(map.size() == (i < s.num_static_props ? 1023 : 0),
              "Invalid static replay mapping size");
      for (auto value : map)
        Require(value <= s.compact_properties[i].size(),
                "Invalid static replay mapping");
    }
    s.property_mapping.resize(s.props.size());
    for (size_t i = 0; i < s.property_mapping.size(); ++i) {
      auto &map = s.property_mapping[i];
      map = r.Vector<uint8_t>(1023);
      Require(map.size() == 1023, "Invalid replay mapping size");
      for (auto value : map)
        Require(value <= s.compact_properties[i + s.num_static_props].size(),
                "Invalid replay mapping");
    }
    return s;
  }
};

Status ComputeBestTreeUncaptured(TreeSamples &, float,
                                 const std::vector<ModularMultiplierInfo> &,
                                 StaticPropRange, float, Tree *);

Status ComputeBestTree(TreeSamples &samples, float threshold,
                       const std::vector<ModularMultiplierInfo> &multipliers,
                       StaticPropRange range, float fast_decode_multiplier,
                       Tree *tree) {
  using namespace reference_learning;
  if (!active)
    return ComputeBestTreeUncaptured(samples, threshold, multipliers, range,
                                     fast_decode_multiplier, tree);
  Require(tree->empty(), "Capture expects an empty output tree");
  FitEvent event{samples.NumSamples(),
                 samples.NumDistinctSamples(),
                 samples.NumProperties(),
                 samples.NumPredictors(),
                 0,
                 threshold,
                 fast_decode_multiplier,
                 {},
                 false};
  const bool capture = !active->directory.empty() || active->captured;
  Fixture fixture;
  if (capture)
    fixture = {samples,     threshold, fast_decode_multiplier,
               multipliers, range,     {}};
  const Timer timer;
  auto status = ComputeBestTreeUncaptured(samples, threshold, multipliers,
                                          range, fast_decode_multiplier, tree);
  event.fit = timer.Elapsed();
  if (!status)
    return status;
  event.nodes = tree->size();
  if (capture) {
    fixture.expected = *tree;
  }
  if (!active->directory.empty()) {
    const auto path = active->directory /
                      ("fit-" + std::to_string(active->fits.size()) + ".bin");
    WriteFixture(path, fixture);
    // Verify a disk roundtrip, not just an in-memory copy of the inputs.
    auto loaded = ReadFixture(path);
    (void)Replay(loaded);
    event.replayed = true;
  }
  if (active->captured)
    active->captured->push_back(std::move(fixture));
  active->fits.push_back(event);
  return true;
}
} // namespace jxl

namespace reference_learning {
thread_local Session *active = nullptr;
Timing Timer::Elapsed() const {
  return {std::chrono::duration<double>(Clock::now() - start).count(),
          double(std::clock() - cpu_start) / CLOCKS_PER_SEC};
}
ScopedSession::ScopedSession(Session *value) : previous_(active) {
  active = value;
}
ScopedSession::~ScopedSession() { active = previous_; }
LearningTimer::~LearningTimer() {
  if (active)
    active->learning.push_back(timer.Elapsed());
}

void WriteFixture(const std::filesystem::path &path, const Fixture &fixture) {
  Require(!std::filesystem::exists(path),
          "Refusing to overwrite a replay fixture");
  Writer w(path);
  w.Scalar(kMagic);
  w.Scalar<uint32_t>(1);
  w.Scalar(std::bit_cast<uint32_t>(fixture.threshold));
  w.Scalar(std::bit_cast<uint32_t>(fixture.fast_decode_multiplier));
  WriteRange(w, fixture.range);
  w.Scalar<uint64_t>(fixture.multipliers.size());
  for (const auto &multiplier : fixture.multipliers) {
    WriteRange(w, multiplier.range);
    w.Scalar(multiplier.multiplier);
  }
  jxl::ReferenceLearningSampleAccess::Write(w, fixture.samples);
  WriteTree(w, fixture.expected);
  w.Close();
}

Fixture ReadFixture(const std::filesystem::path &path) {
  Reader r(path);
  Require(r.Scalar<uint64_t>() == kMagic && r.Scalar<uint32_t>() == 1,
          "Unsupported replay fixture format");
  Fixture fixture;
  fixture.threshold = std::bit_cast<float>(r.Scalar<uint32_t>());
  fixture.fast_decode_multiplier = std::bit_cast<float>(r.Scalar<uint32_t>());
  Require(std::isfinite(fixture.threshold) && fixture.threshold >= 0 &&
              std::isfinite(fixture.fast_decode_multiplier) &&
              fixture.fast_decode_multiplier >= 1,
          "Invalid replay fitting parameters");
  fixture.range = ReadRange(r);
  fixture.multipliers.resize(r.Count(size_t{1} << 20));
  for (auto &multiplier : fixture.multipliers) {
    multiplier.range = ReadRange(r);
    multiplier.multiplier = r.Scalar<uint32_t>();
    Require(multiplier.multiplier != 0, "Invalid replay multiplier");
  }
  fixture.samples = jxl::ReferenceLearningSampleAccess::Read(r);
  fixture.expected = ReadTree(r);
  r.End();
  return fixture;
}

bool SameTree(const jxl::Tree &a, const jxl::Tree &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i) {
    const auto &x = a[i];
    const auto &y = b[i];
    if (x.property != y.property || x.splitval != y.splitval ||
        x.lchild != y.lchild || x.rchild != y.rchild ||
        x.predictor != y.predictor ||
        x.predictor_offset != y.predictor_offset ||
        x.multiplier != y.multiplier)
      return false;
  }
  return true;
}

Timing Replay(const Fixture &fixture) {
  auto samples =
      fixture.samples; // Fitting partitions in place; copy outside timer.
  jxl::Tree tree;
  const Timer timer;
  Require(bool(jxl::ComputeBestTreeUncaptured(
              samples, fixture.threshold, fixture.multipliers, fixture.range,
              fixture.fast_decode_multiplier, &tree)),
          "Reference replay failed");
  const Timing elapsed = timer.Elapsed();
  Require(SameTree(tree, fixture.expected), "Reference replay tree mismatch");
  return elapsed;
}

void PrintSession(std::ostream &out, const Session &session) {
  const bool capturing = !session.directory.empty() || session.captured;
  out << "{\"capture\":" << (capturing ? "true" : "false")
      << ",\"fits\":[";
  for (size_t i = 0; i < session.fits.size(); ++i) {
    const auto &f = session.fits[i];
    out << (i ? "," : "") << "{\"samples\":" << f.samples
        << ",\"distinct\":" << f.distinct << ",\"properties\":" << f.properties
        << ",\"predictors\":" << f.predictors << ",\"nodes\":" << f.nodes
        << ",\"threshold\":" << f.threshold
        << ",\"fast_decode_multiplier\":" << f.fast_decode_multiplier
        << ",\"cpu\":" << f.fit.cpu << ",\"wall\":" << f.fit.wall
        << ",\"replayed\":" << (f.replayed ? "true" : "false")
        << ",\"exceeds_native_sample_admission\":"
        << (f.samples > (size_t{1} << 20) ? "true" : "false") << '}';
  }
  out << "],\"learning\":[";
  for (size_t i = 0; i < session.learning.size(); ++i) {
    const auto &t = session.learning[i];
    out << (i ? "," : "") << "{\"cpu\":" << t.cpu << ",\"wall\":" << t.wall
        << '}';
  }
  out << "],\"learning_includes_capture_overhead\":"
      << (capturing ? "true" : "false") << '}';
}
} // namespace reference_learning
