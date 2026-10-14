// Copyright (c) the JPEG XL Project Authors.
// Copyright (c) 2026 Yunho Cho
// SPDX-License-Identifier: BSD-3-Clause
// Lossless palette policy/implicit colors adapted from pinned libjxl.
#include "codec/modular/transform/reference_palette.h"
#include "codec/modular/transform/operations.h"
#include <algorithm>
#include <bit>
#include <stdexcept>

namespace gjxl::modular_internal {
namespace {
using Color = std::array<int32_t, 4>;
constexpr uint16_t kMissing = UINT16_MAX;
constexpr uint8_t kNoImplicit = UINT8_MAX;
Status Invalid() { return Status::InvalidArgument("Invalid reference palette"); }
int32_t ImplicitValue(size_t index, size_t channel, uint8_t bits) {
  if (channel >= 3)
    return 0;
  const uint64_t maximum = (uint64_t{1} << bits) - 1;
  if (index < 64)
    return static_cast<int32_t>((((index >> (2 * channel)) % 4) * maximum) >> 2) +
           (1 << std::max(0, int{bits} - 3));
  index -= 64;
  for (size_t c = 0; c < channel; ++c)
    index /= 5;
  return static_cast<int32_t>(((index % 5) * maximum) >> 2);
}
Color ImplicitColor(size_t index, const ReferencePaletteOptions &o) {
  Color color{};
  for (size_t c = 0; c < o.count; ++c)
    color[c] = ImplicitValue(index, c, o.bit_depth);
  return color;
}
Color ReadColor(std::span<const ModularChannelView> input, size_t x, size_t y) {
  Color color{};
  for (size_t c = 0; c < input.size(); ++c)
    color[c] = input[c].Row(y)[x];
  return color;
}
struct Record {
  Color color;
  uint16_t explicit_index = kMissing;
  uint8_t implicit_index = kNoImplicit, frequency = 0;
};
// Non-arithmetic order records retain the reference's ordinary std::sort
// path, including equivalent-key ordering on the qualified standard library.
struct Order {
  uint16_t record;
};
struct Dictionary {
  std::array<Record, kMaximumReferencePaletteColors> records;
  std::array<uint16_t, 4096> table;
  std::array<Order, kMaximumReferencePaletteColors> order;
  size_t used = 0, colors = 0;
  Dictionary() { table.fill(kMissing); }
  size_t Slot(const Color &color) const {
    uint32_t hash = 2166136261u;
    for (auto value : color) {
      hash ^= static_cast<uint32_t>(value);
      hash *= 16777619u;
      hash ^= hash >> 16;
    }
    size_t slot = hash & (table.size() - 1);
    while (table[slot] != kMissing && records[table[slot]].color != color)
      slot = (slot + 1) & (table.size() - 1);
    return slot;
  }
  uint16_t Insert(size_t slot, const Color &color) {
    const auto id = static_cast<uint16_t>(used++);
    records[id] = Record{color};
    table[slot] = id;
    return id;
  }
};
static_assert(sizeof(Dictionary) < 35 * 1024);
// No pixel-sized scratch or output allocation until a proposal succeeds.
bool Build(std::span<const ModularChannelView> input, const ReferencePaletteOptions &o,
           Dictionary *d) {
  if (o.maximum_colors < 2)
    return false;
  if (o.count > 1) {
    for (size_t k = 0; k < kImplicitPaletteColors; ++k) {
      const auto color = ImplicitColor(k, o);
      const auto slot = d->Slot(color);
      const auto id = d->table[slot] == kMissing ? d->Insert(slot, color) : d->table[slot];
      d->records[id].implicit_index = static_cast<uint8_t>(k); // Last duplicate wins.
    }
  }
  size_t implicit_used = 0;
  const auto [w, h] = input[0].descriptor.extent;
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x) {
      const auto color = ReadColor(input, x, y);
      const auto slot = d->Slot(color);
      auto id = d->table[slot];
      if (id == kMissing) {
        if (d->colors == o.maximum_colors)
          return false;
        id = d->Insert(slot, color);
        d->order[d->colors++] = {id};
      }
      auto &record = d->records[id];
      if (!record.frequency && record.implicit_index != kNoImplicit)
        ++implicit_used;
      // Only >4 and >10 matter. Saturation avoids population overflow.
      if (record.frequency < 11)
        ++record.frequency;
    }
  if (o.count > 1 && d->colors + implicit_used == 1)
    return false;
  if (o.count > 1)
    for (size_t k = 0; k < kImplicitPaletteColors; ++k) {
      const auto id = d->table[d->Slot(ImplicitColor(k, o))];
      if (d->records[id].frequency > 10)
        d->order[d->colors++] = {id}; // Include repeated implicit colors, like libjxl.
    }
  auto first = d->order.begin(), last = first + d->colors;
  if (o.count == 1) {
    std::sort(first, last, [&](Order a, Order b) {
      return d->records[a.record].color[0] < d->records[b.record].color[0];
    });
  } else if (o.ordered && o.count >= 3) {
    auto key = [&](Order id) {
      const auto &r = d->records[id.record];
      const auto &v = r.color;
      float y = 0.299f * v[0] + 0.587f * v[1] + 0.114f * v[2] + 0.1f;
      if (o.count > 3)
        y *= 1.f + v[3];
      return r.frequency > 4 ? -y : y;
    };
    std::sort(first, last, [&](Order a, Order b) { return key(a) < key(b); });
  }
  for (size_t i = 0; i < d->colors; ++i)
    d->records[d->order[i].record].explicit_index = static_cast<uint16_t>(i);
  return true;
}
TransformDescriptor Descriptor(const ReferencePaletteOptions &o, size_t colors) {
  TransformDescriptor t{TransformKind::kPalette, o.begin, o.count, static_cast<uint32_t>(colors)};
  t.palette_bit_depth = o.bit_depth;
  t.palette_ordered = o.ordered;
  return t;
}
Status Shape(std::span<const ModularChannelView> input, size_t metadata,
             const ReferencePaletteOptions &o, size_t colors, ChannelShape *out) {
  if (input.empty() || input.size() > kMaximumTransformedChannels || !o.count || o.count > 4 ||
      !o.bit_depth || o.bit_depth > 24 ||
      o.maximum_colors >
          (o.count == 1 ? kMaximumReferenceChannelPaletteColors : kMaximumPaletteColors))
    return Invalid();
  ChannelShape source;
  source.count = input.size();
  source.metadata = metadata;
  for (size_t c = 0; c < input.size(); ++c) {
    if (auto s = input[c].Validate(); !s.ok())
      return s;
    source.descriptors[c] = input[c].descriptor;
  }
  TransformSequence sequence;
  sequence.size = 1;
  sequence.entries[0] = Descriptor(o, colors);
  std::array<ChannelShape, kMaximumTransforms + 1> shapes;
  if (auto s = PlanTransforms(source.channels(), metadata, sequence, &shapes); !s.ok())
    return s;
  *out = shapes[1];
  return Status::Ok();
}
void Copy(ModularChannelView source, std::span<int32_t> destination) {
  const auto [w, h] = source.descriptor.extent;
  if (!w)
    return;
  for (size_t y = 0; y < h; ++y)
    std::copy_n(source.Row(y), w, destination.data() + y * w);
}
// Large single-channel dictionaries have no implicit entries or frequency
// ordering. A <=65536 value span uses a membership bitmap plus per-word ranks,
// avoiding a heap LUT. Sparse signed ranges use two bounded managed arrays.
struct ChannelDictionary {
  std::array<uint64_t, 1024> present{};
  std::array<uint32_t, 1024> preceding;
  resource_budget_internal::ManagedVector<int32_t> values;
  resource_budget_internal::ManagedVector<uint32_t> lookup;
  int32_t minimum = 0;
  uint32_t colors = 0;
  bool dense = true, accepted = false;
  size_t Slot(int32_t value) const {
    uint32_t hash = static_cast<uint32_t>(value);
    hash ^= hash >> 16;
    hash *= 0x7feb352du;
    hash ^= hash >> 15;
    hash *= 0x846ca68bu;
    hash ^= hash >> 16;
    size_t slot = hash & (lookup.size() - 1);
    while (lookup[slot] && values[lookup[slot] - 1] != value)
      slot = (slot + 1) & (lookup.size() - 1);
    return slot;
  }
  Status Discover(ModularChannelView input, uint32_t limit) {
    const auto [w, h] = input.descriptor.extent;
    minimum = input.Row(0)[0];
    int32_t maximum = minimum;
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x) {
        minimum = std::min(minimum, input.Row(y)[x]);
        maximum = std::max(maximum, input.Row(y)[x]);
      }
    dense = int64_t{maximum} - minimum < 65536;
    try {
      if (!dense) {
        const size_t capacity = std::min(size_t{limit}, w * h);
        values.reserve(capacity);
        lookup.resize(std::bit_ceil(2 * capacity));
      }
      for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) {
          const int32_t value = input.Row(y)[x];
          if (dense) {
            const size_t offset = int64_t{value} - minimum;
            const uint64_t bit = uint64_t{1} << (offset % 64);
            auto &word = present[offset / 64];
            if (word & bit)
              continue;
            if (colors == limit)
              return Status::Ok();
            word |= bit;
          } else {
            const size_t slot = Slot(value);
            if (lookup[slot])
              continue;
            if (colors == limit)
              return Status::Ok();
            values.push_back(value);
            lookup[slot] = colors + 1;
          }
          ++colors;
        }
      if (dense) {
        uint32_t before = 0;
        for (size_t i = 0; i < present.size(); ++i) {
          preceding[i] = before;
          before += std::popcount(present[i]);
        }
      } else {
        std::sort(values.begin(), values.end());
        std::fill(lookup.begin(), lookup.end(), 0);
        for (size_t i = 0; i < values.size(); ++i)
          lookup[Slot(values[i])] = static_cast<uint32_t>(i + 1);
      }
      accepted = true;
      return Status::Ok();
    } catch (const resource_budget_internal::ManagedAllocationFailure &e) {
      return e.status();
    } catch (const std::bad_alloc &) {
      return Status::OutOfMemory("Channel palette allocation failed");
    } catch (const std::length_error &) {
      return Invalid();
    }
  }
  uint32_t Index(int32_t value) const {
    if (!dense)
      return lookup[Slot(value)] - 1;
    const size_t offset = int64_t{value} - minimum;
    return preceding[offset / 64] +
           std::popcount(present[offset / 64] & ((uint64_t{1} << (offset % 64)) - 1));
  }
  void Write(std::span<const ModularChannelView> input, size_t begin, ModularImage *out) const {
    auto palette = out->samples(0);
    if (dense) {
      size_t index = 0;
      for (size_t i = 0; i < present.size(); ++i) {
        uint64_t word = present[i];
        while (word) {
          palette[index++] = static_cast<int32_t>(
              int64_t{minimum} + static_cast<int64_t>(64 * i + std::countr_zero(word)));
          word &= word - 1;
        }
      }
    } else {
      std::copy(values.begin(), values.end(), palette.begin());
    }
    const auto [w, h] = input[begin].descriptor.extent;
    auto indices = out->samples(begin + 1);
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x)
        indices[y * w + x] = Index(input[begin].Row(y)[x]);
    for (size_t c = 0; c < input.size(); ++c)
      if (c != begin)
        Copy(input[c], out->samples(c + 1));
  }
};
static_assert(sizeof(ChannelDictionary) < 13 * 1024);
void Write(std::span<const ModularChannelView> input, const ReferencePaletteOptions &o,
           const Dictionary &d, ModularImage *out) {
  for (size_t c = 0; c < o.count; ++c)
    for (size_t i = 0; i < d.colors; ++i)
      out->samples(0)[c * d.colors + i] = d.records[d.order[i].record].color[c];
  const auto selected = input.subspan(o.begin, o.count);
  const auto [w, h] = selected[0].descriptor.extent;
  auto indices = out->samples(o.begin + 1);
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x) {
      const auto &r = d.records[d.table[d.Slot(ReadColor(selected, x, y))]];
      indices[y * w + x] =
          r.explicit_index != kMissing ? r.explicit_index : d.colors + r.implicit_index;
    }
  for (size_t c = 0; c < input.size(); ++c)
    if (c < o.begin || c >= size_t{o.begin} + o.count)
      Copy(input[c], out->samples(c < o.begin ? c + 1 : c + 2 - o.count));
}
} // namespace
Status ComputeReferenceChannelPaletteScratchBound(Extent2D extent, uint32_t maximum_colors,
                                                  resource_budget_internal::HostStorageBound *out) {
  size_t area;
  if (!out || extent.empty() || !extent.try_area(&area) ||
      maximum_colors > kMaximumReferenceChannelPaletteColors)
    return Invalid();
  resource_budget_internal::HostStorageBound bound;
  if (maximum_colors > kMaximumPaletteColors) {
    const size_t capacity = std::min(size_t{maximum_colors}, area);
    using resource_budget_internal::VectorCapacityPolicy;
    if (!bound.AddVector<int32_t>(capacity, VectorCapacityPolicy::kFreshExact) ||
        !bound.AddVector<uint32_t>(std::bit_ceil(2 * capacity), VectorCapacityPolicy::kFreshExact))
      return Invalid();
    bound.retained_bytes = 0;
  }
  *out = bound;
  return Status::Ok();
}
Status ComputeReferencePaletteStorageBound(std::span<const ModularChannelView> input,
                                           size_t metadata, const ReferencePaletteOptions &o,
                                           resource_budget_internal::HostStorageBound *out) {
  if (!out)
    return Invalid();
  ChannelShape shape;
  const size_t colors =
      std::max(size_t{1}, size_t{o.maximum_colors}) + (o.count > 1 ? kImplicitPaletteColors : 0);
  if (auto s = Shape(input, metadata, o, colors, &shape); !s.ok())
    return s;
  resource_budget_internal::HostStorageBound bound;
  if (o.maximum_colors >= 2)
    if (auto s = ComputeModularImageStorageBound(shape.channels(), shape.metadata, &bound); !s.ok())
      return s;
  if (o.count == 1) {
    resource_budget_internal::HostStorageBound scratch;
    if (auto s = ComputeReferenceChannelPaletteScratchBound(input[o.begin].descriptor.extent,
                                                            o.maximum_colors, &scratch);
        !s.ok())
      return s;
    if (!bound.Add(scratch))
      return Invalid();
  }
  *out = bound;
  return Status::Ok();
}
Status TryBuildReferencePalette(std::span<const ModularChannelView> input, size_t metadata,
                                const ReferencePaletteOptions &o, ReferencePaletteCandidate *out) {
  if (!out)
    return Invalid();
  resource_budget_internal::HostStorageBound bound;
  if (auto s = ComputeReferencePaletteStorageBound(input, metadata, o, &bound); !s.ok())
    return s;
  ReferencePaletteCandidate candidate;
  if (o.count == 1 && o.maximum_colors > kMaximumPaletteColors) {
    ChannelDictionary dictionary;
    if (auto s = dictionary.Discover(input[o.begin], o.maximum_colors); !s.ok())
      return s;
    if (dictionary.accepted) {
      ChannelShape shape;
      if (auto s = Shape(input, metadata, o, dictionary.colors, &shape); !s.ok())
        return s;
      if (auto s = ModularImage::Create(shape.channels(), shape.metadata, &candidate.image);
          !s.ok())
        return s;
      dictionary.Write(input, o.begin, &candidate.image);
      candidate.transform = Descriptor(o, dictionary.colors);
      candidate.applied = true;
    }
  } else {
    Dictionary dictionary;
    if (Build(input.subspan(o.begin, o.count), o, &dictionary)) {
      ChannelShape shape;
      if (auto s = Shape(input, metadata, o, dictionary.colors, &shape); !s.ok())
        return s;
      if (auto s = ModularImage::Create(shape.channels(), shape.metadata, &candidate.image);
          !s.ok())
        return s;
      Write(input, o, dictionary, &candidate.image);
      candidate.transform = Descriptor(o, dictionary.colors);
      candidate.applied = true;
    }
  }
  *out = std::move(candidate);
  return Status::Ok();
}
Status ApplyReferencePalette(const ModularImage &in, const TransformDescriptor &t, bool inverse,
                             ModularImage *out) {
  if (!inverse) {
    std::array<ModularChannelView, kMaximumTransformedChannels> views;
    for (size_t c = 0; c < in.channel_count(); ++c)
      views[c] = in.view(c);
    const auto input = std::span(views).first(in.channel_count());
    if (t.count == 1 && t.colors > kMaximumPaletteColors) {
      ChannelDictionary dictionary;
      if (auto s = dictionary.Discover(input[t.begin], t.colors); !s.ok())
        return s;
      if (!dictionary.accepted || dictionary.colors != t.colors)
        return Invalid();
      dictionary.Write(input, t.begin, out);
      return Status::Ok();
    }
    const ReferencePaletteOptions o{
        t.begin, t.count, t.palette_bit_depth,
        static_cast<uint16_t>(
            std::max(size_t{2}, std::min(size_t{t.colors}, kMaximumPaletteColors))),
        t.palette_ordered};
    Dictionary dictionary;
    if (!Build(input.subspan(t.begin, t.count), o, &dictionary) || dictionary.colors != t.colors)
      return Invalid();
    Write(input, o, dictionary, out);
  } else {
    const auto indices = in.view(t.begin + 1);
    const auto [w, h] = indices.descriptor.extent;
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x) {
        const int32_t index = indices.Row(y)[x];
        if (index < 0 || index >= t.colors + (t.count > 1 ? kImplicitPaletteColors : 0))
          return Invalid();
        for (size_t c = 0; c < t.count; ++c)
          out->samples(t.begin + c)[y * w + x] =
              index < t.colors ? in.view(0).Row(c)[index]
                               : ImplicitValue(index - t.colors, c, t.palette_bit_depth);
      }
    for (size_t c = 0; c < out->channel_count(); ++c)
      if (c < t.begin || c >= size_t{t.begin} + t.count)
        Copy(in.view(c < t.begin ? c + 1 : c + 2 - t.count), out->samples(c));
  }
  return Status::Ok();
}
} // namespace gjxl::modular_internal
