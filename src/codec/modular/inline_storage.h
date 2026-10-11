// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "core/managed_allocator.h"
#include <algorithm>
#include <array>

namespace gjxl::modular_internal {
// Small fixed-capacity records remain allocation-free. Larger records use one
// fresh exact managed backing, never implicit vector growth. Resize preserves
// the prefix; allocation failure preserves the old record. Inline/control
// storage is excluded from HostStorageBound, like the original fixed arrays.
template <typename T, size_t InlineCount> class InlineStorage {
public:
  InlineStorage() = default;
  InlineStorage(const InlineStorage &) = default;
  InlineStorage(InlineStorage &&) noexcept = default;
  InlineStorage &operator=(InlineStorage &&) noexcept = default;
  InlineStorage &operator=(const InlineStorage &other) {
    if (this != &other) {
      InlineStorage copy(other);
      *this = std::move(copy);
    }
    return *this;
  }
  void resize(size_t count) {
    if (count <= InlineCount) {
      if (!large_.empty()) {
        small_ = {};
        std::copy_n(large_.begin(), count, small_.begin());
        large_ = resource_budget_internal::ManagedVector<T>{};
      }
    } else if (count != large_.size()) {
      resource_budget_internal::ManagedVector<T> fresh(count);
      std::copy_n(data(), std::min(count, size()), fresh.data());
      large_ = std::move(fresh);
    }
  }
  size_t size() const noexcept { return large_.empty() ? InlineCount : large_.size(); }
  T *data() noexcept { return large_.empty() ? small_.data() : large_.data(); }
  const T *data() const noexcept { return large_.empty() ? small_.data() : large_.data(); }
  T *begin() noexcept { return data(); }
  const T *begin() const noexcept { return data(); }
  T *end() noexcept { return data() + size(); }
  const T *end() const noexcept { return data() + size(); }
  T &operator[](size_t i) noexcept { return data()[i]; }
  const T &operator[](size_t i) const noexcept { return data()[i]; }
  bool operator==(const InlineStorage &other) const {
    return size() == other.size() && std::equal(begin(), end(), other.begin());
  }

private:
  std::array<T, InlineCount> small_{};
  resource_budget_internal::ManagedVector<T> large_;
};
} // namespace gjxl::modular_internal
