#pragma once

#include <iostream>

namespace Optiz {

#define CONST(x) Optiz::MetaConst<x>()
#define CONST_VAL(x) std::decay_t<decltype(x)>::value

template <auto V> struct MetaConst;

template <typename T, T V> struct MetaConstBase {
  static constexpr T value = V;
  using Type = T;

  template <typename G, G V2>
  decltype(auto) operator*(const MetaConstBase<G, V2> &other) const {
    return MetaConst<V * V2>();
  }
  template <typename G, G V2>
  decltype(auto) operator/(const MetaConstBase<G, V2> &other) const {
    return MetaConst<V / V2>();
  }
  template <typename G, G V2>
  decltype(auto) operator+(const MetaConstBase<G, V2> &other) const {
    return MetaConst<V + V2>();
  }
  template <typename G, G V2>
  decltype(auto) operator-(const MetaConstBase<G, V2> &other) const {
    return MetaConst<V - V2>();
  }

  operator T() const {
    return V;
  }

  friend std::ostream &operator<<(std::ostream &s, const MetaConstBase &var) {
    s << var.value;
    return s;
  }
};

template<typename T, T V> decltype(auto) operator-(const MetaConstBase<T, V> &other) {
  return MetaConst<-V>();
}

template <auto V> struct MetaConst : public MetaConstBase<decltype(V), V> {};
} // namespace Optiz
