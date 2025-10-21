#pragma once
#include "MetaVec.h"

namespace Optiz {

template <int I, typename T> struct Entry {
  enum CompileTimeStuff { Index = I };
  T val;
  Entry(const T &val) : val(val) {}
};

template <typename Vec = MetaVec<>> struct MetaIntMap {
  Vec entries;
  MetaIntMap(const Vec &vec = MetaVec()) : entries(vec) {}

  template <int I = 0, int N = 0> static constexpr int min_int() {
    if constexpr (I == 0) {
      return min_int<I + 1, TYPE(entries.template get<I>())::Index>();
    } else if constexpr (I == TYPE(entries)::Size) {
      return N;
    } else {
      constexpr int index = TYPE(entries.template get<I>())::Index;
      return min_int<I + 1, std::min(N, index)>();
    }
  }
  template <int I = 0, int N = 0> static constexpr int max_int() {
    if constexpr (I == 0) {
      return max_int<I + 1, TYPE(entries.template get<I>())::Index>();
    } else if constexpr (I == TYPE(entries)::Size) {
      return N;
    } else {
      constexpr int index = TYPE(entries.template get<I>())::Index;
      return max_int<I + 1, std::max(N, index)>();
    }
  }

  template <int I, typename Default, int J = 0> decltype(auto) get() const {
    if constexpr (J == TYPE(entries)::Size) {
      return Default();
    } else {
      constexpr int index = TYPE(entries.template get<J>())::Index;
      if constexpr (index == I) {
        return entries.template get<J>().val;
      } else {
        return get<I, Default, J + 1>();
      }
    }
  }

  template <int I, typename T, int J = 0> auto set(const T &value) const {
    if constexpr (J == TYPE(entries)::Size) {
      auto res = entries.push(Entry<I, T>(value));
      return MetaIntMap<TYPE(res)>(res);
    } else {
      constexpr int index = TYPE(entries.template get<J>())::Index;
      if constexpr (index == I) {
        auto new_entry = Entry<I, T>(value);
        auto res = entries.template set<J>(new_entry);
        return MetaIntMap<TYPE(res)>(res);
      } else {
        return set<I, T, J + 1>(value);
      }
    }
  }

  template <typename OtherMap, int I = 0>
  auto operator+(const OtherMap &other) const {
    if constexpr (I == TYPE(other.entries)::Size) {
      return *this;
    } else {
      constexpr int index = TYPE(other.entries.template get<I>())::Index;
      auto entry = other.entries.template get<I>();
      auto new_map = this->template set<index>(
          this->template get<index, double>() + entry.val);
      return new_map.template operator+ <OtherMap, I + 1>(other);
    }
  }
  template <typename OtherMap, int I = 0>
  auto operator-(const OtherMap &other) const {
    if constexpr (I == TYPE(other.entries)::Size) {
      return *this;
    } else {
      constexpr int index = TYPE(other.entries.template get<I>())::Index;
      auto entry = other.entries.template get<I>();
      auto new_map = this->template set<index>(
          this->template get<index, double>() - entry.val);
      return new_map.template operator- <OtherMap, I + 1>(other);
    }
  }

  template <typename Func, int I = 0> auto map(const Func &f) const {
    if constexpr (I == TYPE(entries)::Size) {
      return *this;
    } else {
      constexpr int index = TYPE(entries.template get<I>())::Index;
      return this->template set<index>(f(entries.template get<I>()))
          .template map<Func, I + 1>(f);
    }
  }
  template <int I = 0>
  friend std::ostream &operator<<(std::ostream &s, MetaIntMap const &expr) {
    if constexpr (I == TYPE(expr.entries)::Size) {
      return s;
    } else {
      constexpr int index = TYPE(expr.entries.template get<I>())::Index;
      s << "{" << index << ": " << expr.entries.template get<I>().val << "}"
        << " ";
      return operator<< <I + 1>(s, expr);
    }
  }
};

} // namespace Optiz