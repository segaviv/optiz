#pragma once

#include <cmath>
#include <type_traits>

namespace Optiz {
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
inline T sqr(const T &x) {
  return x * x;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
inline T exp(const T &x) {
  return std::exp(x);
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
inline T sin(const T &x) {
  return std::sin(x);
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
inline T cos(const T &x) {
  return std::cos(x);
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
inline T log(const T &x) {
  return std::log(x);
}
} // namespace Optiz