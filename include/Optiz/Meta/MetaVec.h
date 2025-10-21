#pragma once
#include "../Common/Functions.h"
#include "MetaUtils.h"
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace Optiz {

template <typename... Args> struct MetaVec {

  std::tuple<Args...> exprs;
  MetaVec(Args... args) : exprs(args...) {}

  enum CompileTimeStuff { Size = sizeof...(Args) };

  template <int I> decltype(auto) get() const { return std::get<I>(exprs); }

  decltype(auto) x() const { return get<0>(); }
  decltype(auto) y() const { return get<1>(); }
  decltype(auto) z() const { return get<2>(); }

  template <typename T> decltype(auto) push(const T &t) const {
    if constexpr (sizeof...(Args) == 0) {
      return MetaVec<T>(t);
    } else {
      return std::apply(
          [&](auto &...args) { return MetaVec<Args..., T>(args..., t); },
          exprs);
    }
  }

  template <typename... OtherArgs>
  decltype(auto) concat(const MetaVec<OtherArgs...> &other) const {
    return std::apply(
        [&](auto &...args) {
          return std::apply(
              [&](auto &...other_args) {
                return MetaVec<Args..., OtherArgs...>(args..., other_args...);
              },
              other.exprs);
        },
        exprs);
  }

  template <int Start, int End> decltype(auto) slice() const {
    if constexpr (Start >= End) {
      return MetaVec<>();
    } else {
      auto elem_start = get<Start>();
      return MetaVec<TYPE(elem_start)>(elem_start)
          .concat(slice<Start + 1, End>());
    }
  }

  // Set an element.
  template <int I, typename T> decltype(auto) set(const T &t) const {
    return slice<0, I>().push(t).concat(slice<I + 1, sizeof...(Args)>());
  }
  template <int I> decltype(auto) pop() const {
    return slice<0, I>().concat(slice<I + 1, sizeof...(Args)>());
  }

  // Map elements.
  template <int I = 0, typename Func, typename... OtherArgs>
  decltype(auto) map(Func &&f,
                     const MetaVec<OtherArgs...> &vec = MetaVec<>()) const {
    if constexpr (I == sizeof...(Args)) {
      return vec;
    } else {
      // Only the element.
      if constexpr (std::is_invocable_v<Func, decltype(get<I>())>) {
        auto res = map<I + 1>(std::forward<Func>(f), vec.push(f(get<I>())));
        return res;
      } else {
        // Also get the index.
        auto res = map<I + 1>(std::forward<Func>(f),
                              vec.push(f(get<I>(), ForIndex<I>())));
        return res;
      }
    }
  }

  template <int N, typename Func, int I = 0, typename... OtherArgs>
  static decltype(auto)
  from_func(Func &&f, const MetaVec<OtherArgs...> &vec = MetaVec<>()) {
    if constexpr (I == N) {
      return vec;
    } else {
      auto res = from_func<N, Func, I + 1>(
          std::forward<Func>(f), vec.push(f.template operator()<I>()));
      return res;
    }
  }

  // Reduce stuff.
  template <int I, typename Func, typename Aggregate>
  decltype(auto) reduce_aux(Func &&f, const Aggregate &aggregate) const {
    if constexpr (I == sizeof...(Args)) {
      return aggregate;
    } else {
      if constexpr (std::is_invocable_v<Func, TYPE(get<I>()),
                                        TYPE(aggregate)>) {
        auto res =
            reduce_aux<I + 1>(std::forward<Func>(f), f(get<I>(), aggregate));
        return res;
      } else {
        auto res = reduce_aux<I + 1>(std::forward<Func>(f),
                                     f(get<I>(), ForIndex<I>(), aggregate));
        return res;
      }
    }
  }

  template <typename Func> decltype(auto) reduce(Func &&f) const {
    auto res = reduce_aux<1>(std::forward<Func>(f), get<0>());
    return res;
  }
  template <typename Func, typename Aggregate>
  decltype(auto) reduce(Func &&f, const Aggregate &aggregate) {
    auto res = reduce_aux<0>(std::forward<Func>(f), aggregate);
    return res;
  }

  // Some other operators.
  decltype(auto) sum() const {
    return reduce([](const auto &a, const auto &b) { return a + b; });
  }
  decltype(auto) mul() const {
    return reduce([](const auto &a, const auto &b) { return a * b; });
  }

  decltype(auto) squaredNorm() const {
    return reduce_aux<1>(
        [](const auto &a, const auto &b) { return Optiz::sqr(a) + b; },
        Optiz::sqr(get<0>()));
  }

  template <typename... OtherArgs>
  decltype(auto) operator+(const MetaVec<OtherArgs...> &other) const {
    return map([&](const auto &a, const auto &map_index) {
      return a + other.template get<INDEX(map_index)>();
    });
  }
  template <typename... OtherArgs>
  decltype(auto) operator-(const MetaVec<OtherArgs...> &other) const {
    return map([&](const auto &a, const auto &map_index) {
      return a - other.template get<INDEX(map_index)>();
    });
  }

  template <typename... OtherArgs>
  decltype(auto) operator*(const MetaVec<OtherArgs...> &other) const {
    return map([&](const auto &a, const auto &map_index) {
      return a * other.template get<INDEX(map_index)>();
    });
  }

  template <typename G, typename = typename std::enable_if<
                            std::is_arithmetic<G>::value, G>::type>
  decltype(auto) operator*(G other) const {
    return map([&](const auto &a, const auto &map_index) { return a * other; });
  }

  template <typename... OtherArgs>
  decltype(auto) operator/(const MetaVec<OtherArgs...> &other) const {
    return map([&](const auto &a, const auto &map_index) {
      return a / other.template get<INDEX(map_index)>();
    });
  }

  template <typename... OtherArgs>
  decltype(auto) dot(const MetaVec<OtherArgs...> &other) const {
    return reduce_aux<1>(
        [&](const auto &a, const auto &for_index, const auto &aggregate) {
          return aggregate + a * other.template get<INDEX(for_index)>();
        },
        get<0>() * other.template get<0>());
  }

  template <typename... OtherArgs>
  decltype(auto) cross3(const MetaVec<OtherArgs...> &other) const {
    MetaVec<> res;
    auto res1 = res.push(get<2>() * other.template get<1>() -
                         get<1>() * other.template get<2>());
    auto res2 = res1.push(get<0>() * other.template get<2>() -
                          get<2>() * other.template get<0>());
    return res2.push(get<1>() * other.template get<0>() -
                     get<0>() * other.template get<1>());
  }
};

template <typename... SomeArgs>
std::ostream &operator<<(std::ostream &s, const MetaVec<SomeArgs...> &expr) {
  if constexpr (sizeof...(SomeArgs) == 0) {
    return s;
  } else if constexpr (sizeof...(SomeArgs) == 1) {
    return s << expr.template get<0>();
  } else {
    return s << expr.template get<0>() << std::endl
             << expr.template slice<1, sizeof...(SomeArgs)>();
  }
}

template <int I, typename EigenType, typename... OtherArgs>
decltype(auto) eig_to_meta_vec(const MetaVec<OtherArgs...> &meta_vec,
                               const EigenType &vec) {
  if constexpr (I == TYPE(vec)::SizeAtCompileTime) {
    return meta_vec;
  } else {
    auto res = eig_to_meta_vec<I + 1>(meta_vec.push(vec(I)), vec);
    return res;
  }
}
template <typename EigenType>
decltype(auto) eig_to_meta_vec(const EigenType &vec) {
  return eig_to_meta_vec<0>(MetaVec<>(), vec);
}

template <typename... Args, typename T, int Rows, int Cols>
auto operator-(const MetaVec<Args...> &vec,
               const Eigen::Matrix<T, Rows, Cols> &mat) {
  return vec - eig_to_meta_vec(mat);
}
template <typename... Args, typename T, int Rows, int Cols>
auto operator-(const Eigen::Matrix<T, Rows, Cols> &mat,
               const MetaVec<Args...> &vec) {
  return eig_to_meta_vec(mat) - vec;
}
template <typename... Args, typename T, int Rows, int Cols>
auto operator+(const MetaVec<Args...> &vec,
               const Eigen::Matrix<T, Rows, Cols> &mat) {
  return vec + eig_to_meta_vec(mat);
}
template <typename... Args, typename T, int Rows, int Cols>
auto operator+(const Eigen::Matrix<T, Rows, Cols> &mat,
               const MetaVec<Args...> &vec) {
  return eig_to_meta_vec(mat) + vec;
}

} // namespace Optiz