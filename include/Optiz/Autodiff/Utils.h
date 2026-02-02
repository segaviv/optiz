#pragma once
#include "Var.h"
#include "VarGrad.h"
#include <Eigen/Eigen>

namespace Optiz {
template <typename T>
inline Eigen::MatrixXd dense_jacobian(const Eigen::VectorX<T> &func_values,
                                      int num_vars)
  requires(std::is_same_v<T, Var> || std::is_same_v<T, VarGrad>)
{
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(func_values.size(), num_vars);
  for (int i = 0; i < J.rows(); i++) {
    for (const auto &[row, val2] : func_values[i].grad()) {
      J(i, row) = val2;
    }
  }
  return J;
}

template <typename T>
inline Eigen::SparseMatrix<double>
sparse_jacobian(const Eigen::VectorX<T> &func_values, int num_vars)
  requires(std::is_same_v<T, Var> || std::is_same_v<T, VarGrad>)
{
  Eigen::SparseMatrix<double> J(func_values.size(), num_vars);
  std::vector<Eigen::Triplet<double>> triplets;
  for (int i = 0; i < J.rows(); i++) {
    for (const auto &[row, val2] : func_values[i].grad()) {
      triplets.emplace_back(i, row, val2);
    }
  }
  J.setFromTriplets(triplets.begin(), triplets.end());
  return J;
}

} // namespace Optiz