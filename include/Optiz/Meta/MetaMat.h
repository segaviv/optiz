#pragma once
#include "MetaUtils.h"
#include "MetaVec.h"

#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace Optiz {

template <typename... Args> struct MetaMat {
  const std::tuple<Args...> exprs;
  MetaMat(Args... column_vecs) : exprs(column_vecs...) {}

  static constexpr int rows() {
    if constexpr (Cols == 0) {
      return 0;
    } else {
      return TYPE(std::get<0>(exprs))::Size;
    }
  }
  static constexpr int Rows = MetaMat::rows();
  static constexpr int Cols = sizeof...(Args);

  template <int I = 0, typename EigenType, typename... SomeArgs>
  static decltype(auto)
  from_eig_mat(const EigenType &mat,
               const MetaMat<SomeArgs...> &meta_mat = MetaMat<>()) {
    if constexpr (I == TYPE(mat)::ColsAtCompileTime) {
      return meta_mat;
    } else {
      auto res = eig_to_meta_mat<I + 1>(
          mat, meta_mat.push_col(eig_to_meta_vec(mat.col(I))));
      return res;
    }
  }

  template <int M, int N, typename Func, int J = 0, typename... OtherArgs>
  static decltype(auto)
  from_func(Func &&f, const MetaMat<OtherArgs...> &mat = MetaMat<>()) {
    if constexpr (J == N) {
      return mat;
    } else {
      // Build column J.
      auto vec = MetaVec<>::from_func<M>(
          [&]<int I>() { return f.template operator()<I, J>(); });
      // Add it and build the rest.
      auto res = from_func<M, N, Func, J + 1>(std::forward<Func>(f),
                                              mat.push_col(vec));
      return res;
    }
  }

  template <int I> decltype(auto) col() const { return std::get<I>(exprs); }

  template <typename T> decltype(auto) push_col(const T &t) const {
    if constexpr (Cols == 0) {
      return MetaMat<T>(t);
    } else {
      return std::apply(
          [&](auto &...args) { return MetaMat<Args..., T>(args..., t); },
          exprs);
    }
  }
  template <typename T> decltype(auto) push_col_at_start(const T &t) const {
    if constexpr (Cols == 0) {
      return MetaMat<T>(t);
    } else {
      return std::apply(
          [&](auto &...args) { return MetaMat<T, Args...>(t, args...); },
          exprs);
    }
  }

  template <int I, int J, typename... OtherArgs>
  decltype(auto) row_aux(const MetaVec<OtherArgs...> &vec) const {
    if constexpr (J == Cols) {
      return vec;
    } else {
      auto res = row_aux<I, J + 1>(vec.push(col<J>().template get<I>()));
      return res;
    }
  }
  template <int I> decltype(auto) row() const {
    auto res = row_aux<I, 0>(MetaVec<>());
    return res;
  }

  template <int J, typename... OtherArgs>
  decltype(auto) transpose_aux(const MetaMat<OtherArgs...> &mat) const {
    if constexpr (J == Rows) {
      return mat;
    } else {
      auto res = transpose_aux<J + 1>(mat.push_col(row<J>()));
      return res;
    }
  }
  decltype(auto) transpose() const {
    auto res = transpose_aux<0>(MetaMat<>());
    return res;
  }

  template <int I, int J, int K = 0, typename... OtherArgs>
  auto mul_mat_entry(const MetaMat<OtherArgs...> &other) const {
    if constexpr (K == Cols - 1) {
      return col<K>().template get<I>() *
             other.template col<J>().template get<K>();
    } else {
      return col<K>().template get<I>() *
                 other.template col<J>().template get<K>() +
             mul_mat_entry<I, J, K + 1>(other);
    }
  }

  template <int Col, int I = 0, typename... OtherArgs>
  auto mul_mat_col(const MetaMat<OtherArgs...> &other) const {
    if constexpr (I == Rows) {
      return MetaVec<>();
    } else {
      return mul_mat_col<Col, I + 1>(other).push_at_start(
          mul_mat_entry<I, Col>(other));
    }
  }
  template <int I = 0, int J = 0, typename... OtherArgs>
  auto mul_mat(const MetaMat<OtherArgs...> &other) const {
    static_assert(Cols == TYPE(other)::Rows);
    if constexpr (J == TYPE(other)::Cols) {
      return MetaMat<>();
    } else {
      return mul_mat<I, J + 1>(other).push_col_at_start(mul_mat_col<J>(other));
    }
  }

  template <typename... OtherArgs>
  decltype(auto) operator*(const MetaMat<OtherArgs...> &other) const {
    static_assert(Cols == TYPE(other)::Rows);
    auto res = mul_mat(other);
    return res;
  }

  template <typename... OtherArgs>
  decltype(auto) operator-(const MetaMat<OtherArgs...> &other) const {
    static_assert(Cols == TYPE(other)::Cols);
    static_assert(Rows == TYPE(other)::Rows);
    return map([&](const auto &elem, const auto &index) {
      return elem - other.template col<INDEX(index)>();
    });
  }

  template <typename... OtherArgs>
  decltype(auto) operator+(const MetaMat<OtherArgs...> &other) const {
    static_assert(Cols == TYPE(other)::Cols);
    static_assert(Rows == TYPE(other)::Rows);
    return map([&](const auto &elem, const auto &index) {
      return elem + other.template col<INDEX(index)>();
    });
  }

  template <typename... OtherArgs, typename T, int NumRows, int NumCols>
  decltype(auto)
  operator*(const Eigen::Matrix<T, NumRows, NumCols> &other) const {
    static_assert(NumRows >= 0 && NumCols >= 0);
    static_assert(Cols == NumRows);
    return operator*(from_eig_mat(other));
  }

  template <int RowStart, int RowEnd, int ColStart, int ColEnd,
            typename... OtherArgs>
  decltype(auto) submat(const MetaMat<OtherArgs...> &mat) const {
    if constexpr (ColStart == ColEnd) {
      return mat;
    } else {
      auto res = submat<RowStart, RowEnd, ColStart + 1, ColEnd>(
          mat.push_col(col<ColStart>().template slice<RowStart, RowEnd>()));
      return res;
    }
  }
  template <int RowStart, int RowEnd, int ColStart, int ColEnd>
  decltype(auto) submat() const {
    auto res = submat<RowStart, RowEnd, ColStart, ColEnd>(MetaMat<>());
    return res;
  }

  template <typename... OtherArgs>
  decltype(auto) concat(const MetaMat<OtherArgs...> &other) const {
    return std::apply(
        [&](auto &...args) {
          return std::apply(
              [&](auto &...other_args) {
                return MetaMat<Args..., OtherArgs...>(args..., other_args...);
              },
              other.exprs);
        },
        exprs);
  }

  template <int Start, int End> decltype(auto) slice() const {
    if constexpr (Start >= End) {
      return MetaMat<>();
    } else {
      auto elem_start = col<Start>();
      return MetaMat<TYPE(elem_start)>(elem_start)
          .concat(slice<Start + 1, End>());
    }
  }

  template <int I> decltype(auto) pop() const {
    return slice<0, I>().concat(slice<I + 1, Cols>());
  }

  // Map elements.
  template <int I = 0, typename Func, typename... OtherArgs>
  decltype(auto) map(Func &&f,
                     const MetaMat<OtherArgs...> &mat = MetaMat<>()) const {
    if constexpr (I == Cols) {
      return mat;
    } else {
      // Only wants the element.
      if constexpr (std::is_invocable_v<Func, decltype(col<I>())>) {
        auto res = map<I + 1>(std::forward<Func>(f), mat.push_col(f(col<I>())));
        return res;
      } else {
        // Also get the index.
        auto res = map<I + 1>(std::forward<Func>(f),
                              mat.push_col(f(col<I>(), ForIndex<I>())));
        return res;
      }
    }
  }

  template <int I, int J> decltype(auto) minor() const {
    return pop<J>().map([](auto &elem) { return elem.template pop<I>(); });
  }

  template <int I = 0> decltype(auto) determinant() const {
    if constexpr (0 == Cols) {
      return 1;
    } else if constexpr (Cols == 1 && Rows == 1) {
      return col<0>().template get<0>();
    } else if constexpr (I == Rows - 1) {
      if constexpr (I % 2 == 0) {
        return col<0>().template get<I>() * minor<I, 0>().determinant();
      } else {
        return -col<0>().template get<I>() * minor<I, 0>().determinant();
      }
    } else {
      if constexpr (I % 2 == 0) {
        return col<0>().template get<I>() * minor<I, 0>().determinant() +
               determinant<I + 1>();
      } else {
        return determinant<I + 1>() -
               col<0>().template get<I>() * minor<I, 0>().determinant();
      }
    }
  }

  template <int I = -1> auto inverse() const {
    auto det = determinant();
    return map([&](const auto &elem, const auto &j) {
      return elem.map([&](const auto &elem, const auto &i) {
        constexpr int sign = (INDEX(i) + INDEX(j)) % 2 == 0 ? 1 : -1;
        return sign * minor<INDEX(j), INDEX(i)>().determinant() / det;
      });
    });
  }

  template <int I = 0> decltype(auto) squaredNorm() const {
    if constexpr (0 == Cols) {
      return 0;
    } else if constexpr (I == Cols - 1) {
      return col<I>().squaredNorm();
    } else {
      return col<I>().squaredNorm() + squaredNorm<I + 1>();
    }
  }

  auto to_eigen() const {
    Eigen::Matrix<typename TYPE(col<0>().template get<0>()), Rows, Cols> mat;
    For<0, Cols>([&]<int j>() {
      For<0, Rows>([&]<int i>() { mat(i, j) = col<j>().template get<i>(); });
    });
    return mat;
  }
};

template <int I = 0, typename EigenType, typename... Args>
decltype(auto) eig_to_meta_mat(const EigenType &mat,
                               const MetaMat<Args...> &meta_mat = MetaMat<>()) {
  if constexpr (I == TYPE(mat)::ColsAtCompileTime) {
    return meta_mat;
  } else {
    auto res = eig_to_meta_mat<I + 1>(
        mat, meta_mat.push_col(eig_to_meta_vec(mat.col(I))));
    return res;
  }
}

template <typename... OtherArgs, typename T, int NumRows, int NumCols>
decltype(auto) operator*(const Eigen::Matrix<T, NumRows, NumCols> &other,
                         const MetaMat<OtherArgs...> &mat) {
  static_assert(NumRows >= 0 && NumCols >= 0);
  static_assert(TYPE(mat)::Rows == NumCols);
  return eig_to_meta_mat(other) * mat;
}

template <typename... OtherArgs, typename T, int NumRows, int NumCols>
decltype(auto) operator*(const Eigen::Matrix<T, NumRows, NumCols> &other,
                         const MetaVec<OtherArgs...> &mat) {
  static_assert(NumRows >= 0 && NumCols >= 0);
  static_assert(TYPE(mat)::Size == NumCols);
  auto res_mat = (eig_to_meta_mat(other) * MetaMat(mat));
  auto res_vec = res_mat.template col<0>();
  return res_vec;
}

template <typename... SomeArgs>
std::ostream &operator<<(std::ostream &s, const MetaMat<SomeArgs...> &expr) {
  if constexpr (TYPE(expr)::Cols == 0) {
    return s;
  }
  For<0, TYPE(expr)::Rows>([&]<int i>() {
    For<0, TYPE(expr)::Cols>(
        [&]<int j>() { s << expr.template col<j>().template get<i>() << " "; });
    if constexpr (i < TYPE(expr)::Rows - 1) {
      s << std::endl;
    }
  });
  return s;
}

} // namespace Optiz