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

  template <int I, int J, typename... OtherArgs, typename... Mat1,
            typename... Mat2>
  static decltype(auto) build_col_aux(const MetaVec<OtherArgs...> &vec,
                                      const MetaMat<Mat1...> &mat1,
                                      const MetaMat<Mat2...> &mat2) {
    if constexpr (I == TYPE(mat1)::Cols) {
      return vec;
    } else {
      auto res = build_col_aux<I + 1, J>(
          vec.push(mat1.template col<I>().dot(mat2.template col<J>())), mat1,
          mat2);
      return res;
    }
  }
  template <int J, typename... OtherArgs, typename... Mat1, typename... Mat2>
  static decltype(auto) mul_aux(const MetaMat<OtherArgs...> &mat,
                                const MetaMat<Mat1...> &mat1,
                                const MetaMat<Mat2...> &mat2) {
    if constexpr (J == TYPE(mat2)::Cols) {
      return mat;
    } else {
      auto res = mul_aux<J + 1>(
          mat.push_col(build_col_aux<0, J>(MetaVec<>(), mat1, mat2)), mat1,
          mat2);
      return res;
    }
  }

  template <typename... OtherArgs>
  decltype(auto) operator*(const MetaMat<OtherArgs...> &other) const {
    static_assert(Cols == TYPE(other)::Rows);
    auto mat1 = transpose();
    auto res = mul_aux<0>(MetaMat<>(), mat1, other);
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
      return 0;
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

  template <int I = -1> decltype(auto) inverse() const {
    auto det = determinant();
    return map([&](const auto &elem, const auto &j) {
      return elem.map([&](const auto &elem, const auto &i) {
        constexpr int sign = (INDEX(i) + INDEX(j)) % 2 == 0 ? 1 : -1;
        return sign * minor<INDEX(j), INDEX(i)>().determinant() / det;
      });
    });
  }

  // template <> decltype(auto) inverse<3>() const {
  //   MetaMat<> res;
  //   auto res1 = res.push_col(col<1>().cross3(col<2>()));
  //   auto res2 = res1.push_col(col<2>().cross3(col<0>()));
  //   auto res3 = res2.push_col(col<0>().cross3(col<1>()));
  //   auto det = res3.template col<0>().template dot(col<0>());

  //   return res3
  //       .map([&](const auto &elem, const auto &j) {
  //         return elem.template map(
  //             [&](const auto &elem, const auto &i) { return elem / det; });
  //       })
  //       .transpose();
  // }

  template <int I = 0> decltype(auto) squaredNorm() const {
    if constexpr (0 == Cols) {
      return 0;
    } else if constexpr (I == Cols - 1) {
      return col<I>().squaredNorm();
    } else {
      return col<I>().squaredNorm() + squaredNorm<I + 1>();
    }
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
  return (eig_to_meta_mat(other) * MetaMat(mat)).template col<0>();
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