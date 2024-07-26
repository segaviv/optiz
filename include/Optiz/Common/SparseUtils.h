#pragma once

#include <Eigen/Eigen>
#include <tuple>
#include <vector>

#include "../Linear/LinearExpression.h"

namespace Optiz {

template <typename T>
Eigen::SparseMatrix<T> spcat(
    const std::vector<std::vector<Eigen::SparseMatrix<T>>>& mats) {
  int n_rows = 0, n_cols = 0;
  for (auto& mat : mats) {
    n_rows += mat[0].rows();
  }
  for (auto& mat : mats[0]) {
    n_cols += mat.cols();
  }

  std::vector<Eigen::Triplet<T>> triplets;
  int row_offset = 0, col_offset = 0;
  for (auto& mat : mats) {
    col_offset = 0;
    for (auto& submat : mat) {
      for (int k = 0; k < submat.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(submat, k); it;
             ++it) {
          triplets.push_back(Eigen::Triplet<T>(
              it.row() + row_offset, it.col() + col_offset, it.value()));
        }
      }
      col_offset += submat.cols();
    }
    row_offset += mat[0].rows();
  }
  Eigen::SparseMatrix<T> res(n_rows, n_cols);
  res.setFromTriplets(triplets.begin(), triplets.end());
  return res;
}

template <typename T>
Eigen::SparseMatrix<T> spzero(int n) {
  return Eigen::SparseMatrix<T>(n, n);
}
template <typename T>
Eigen::SparseMatrix<T> spzero(int n_rows, int n_cols) {
  return Eigen::SparseMatrix<T>(n_rows, n_cols);
}

template <typename T>
Eigen::SparseMatrix<T> speye(int n) {
  Eigen::SparseMatrix<T> mat(n, n);
  mat.setIdentity();
  return mat;
}

template <typename T>
Eigen::SparseMatrix<T> sparse(
    const std::vector<int>& variables_shape, int nx, int ny,
    const std::function<Eigen::VectorX<LinearExpression<T>>(
        int, int, const Eigen::MatrixX<LinearExpression<T>>&)>& indices) {
  Eigen::MatrixX<LinearExpression<T>> vars =
      Eigen::MatrixX<LinearExpression<T>>::NullaryExpr(
          variables_shape[0],
          variables_shape.size() > 1 ? variables_shape[1] : 1,
          [&](int i, int j) {
            return LinearExpression<T>(i + j * variables_shape[0], 1);
          });

  int cur_row = 0;
  std::vector<Eigen::Triplet<T>> triplets;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      Eigen::VectorX<LinearExpression<T>> constraints = indices(i, j, vars);
      for (int j = 0; j < constraints.rows(); j++) {
        auto& expr = constraints(j);
        for (auto& [j, coeff] : expr.values()) {
          triplets.push_back({cur_row, j, coeff});
        }
        cur_row++;
      }
    }
  }
  Eigen::SparseMatrix<T> mat(cur_row, vars.size());
  mat.setFromTriplets(triplets.begin(), triplets.end());
  return mat;
}

template <typename T>
Eigen::SparseMatrix<T> sparse(
    const std::vector<int>& variables_shape, int num_elements,
    const std::function<Eigen::VectorX<LinearExpression<T>>(
        int, const Eigen::MatrixX<LinearExpression<T>>&)>& indices) {
  return sparse<T>(variables_shape, num_elements, 1,
                   [&](int i, int j, const auto& x) { return indices(i, x); });
}

inline Eigen::SparseMatrix<std::complex<double>> sparse_cd(
    const std::vector<int>& variables_shape, int num_elements,
    const std::function<Eigen::VectorX<LinearExpression<std::complex<double>>>(
        int, const Eigen::MatrixX<LinearExpression<std::complex<double>>>&)>&
        indices) {
  return sparse<std::complex<double>>(
      variables_shape, num_elements, 1,
      [&](int i, int j, const auto& x) { return indices(i, x); });
}

template <typename T, int Rows, int Cols>
Eigen::MatrixX<T> std_vec_to_eigen_mat(
    const std::vector<Eigen::Matrix<T, Rows, Cols>>& vec) {
  if (vec.size() == 0) {
    return Eigen::MatrixX<T>(0, 0);
  }
  Eigen::MatrixX<T> res(vec.size(), vec[0].size());
  for (int i = 0; i < vec.size(); i++) {
    res.row(i) = vec[i];
  }
  return res;
}

template <typename T>
std::tuple<Eigen::SparseMatrix<T>, Eigen::MatrixX<T>> sparse_with_rhs(
    const std::vector<int>& variables_shape, int nx, int ny,
    const std::function<Eigen::VectorX<LinearExpression<T>>(
        int, int, const Eigen::MatrixX<LinearExpression<T>>&)>& indices) {
  Eigen::MatrixX<LinearExpression<T>> vars =
      Eigen::MatrixX<LinearExpression<T>>::NullaryExpr(
          variables_shape[0],
          variables_shape.size() > 1 ? variables_shape[1] : 1,
          [&](int i, int j) {
            return LinearExpression<T>(i + j * variables_shape[0], 1);
          });

  int cur_row = 0;
  std::vector<Eigen::Triplet<T>> triplets;
  std::vector<Eigen::VectorX<T>> rhs;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      Eigen::VectorX<LinearExpression<T>> constraints = indices(i, j, vars);
      for (int j = 0; j < constraints.rows(); j++) {
        auto& expr = constraints(j);
        for (auto& [j, coeff] : expr.values()) {
          triplets.push_back({cur_row, j, coeff});
        }
        rhs.push_back(-expr.rhs_vector());
        cur_row++;
      }
    }
  }
  Eigen::SparseMatrix<T> mat(cur_row, vars.size());
  mat.setFromTriplets(triplets.begin(), triplets.end());
  return {mat, std_vec_to_eigen_mat(rhs)};
}

template <typename T>
std::tuple<Eigen::SparseMatrix<T>, Eigen::MatrixX<T>> sparse_with_rhs(
    const std::vector<int>& variables_shape, int num_elements,
    const std::function<Eigen::VectorX<LinearExpression<T>>(
        int, const Eigen::MatrixX<LinearExpression<T>>&)>& indices) {
  return sparse_with_rhs<T>(
      variables_shape, num_elements, 1,
      [&](int i, int j, const auto& x) { return indices(i, x); });
}

template <typename T>
inline Eigen::SparseMatrix<T> sparse_identity(int n) {
  Eigen::SparseMatrix<T> mat(n, n);
  mat.setIdentity();
  return mat;
}

template <typename T>
inline Eigen::SparseMatrix<T> sparse_diagonal(const Eigen::VectorXd& diagonal) {
  Eigen::SparseMatrix<T> W(diagonal.size(), diagonal.size());
  W.reserve(Eigen::VectorXi::Constant(diagonal.size(), 1));
  for (int i = 0; i < diagonal.size(); i++) {
    W.insert(i, i) = diagonal(i);
  }
  W.makeCompressed();
  return W;
}

}  // namespace Optiz
