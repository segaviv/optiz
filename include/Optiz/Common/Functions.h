#pragma once

#include <type_traits>

namespace Optiz {
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
inline T sqr(const T &x) {
  return x * x;
}
} // namespace Optiz