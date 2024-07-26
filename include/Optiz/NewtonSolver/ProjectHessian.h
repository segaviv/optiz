#pragma once
#include "SelfAdjointMapMatrix.h"
#include <Eigen/Eigen>

namespace Optiz {

SelfAdjointMapMatrix project_hessian(const SelfAdjointMapMatrix &hessian);

// Returns pair<dense, inds> such that the projected sparse hessian satisfies
// sparse[inds[i], inds[j]] = dense[i, j]
std::pair<Eigen::MatrixXd, std::vector<int>>
project_sparse_hessian(const SelfAdjointMapMatrix &hessian);

template <int k>
bool is_self_adjoint_positive_diagonally_dominant(
    const Eigen::Matrix<double, k, k> &dense) {
  for (int i = 0; i < dense.rows(); i++) {
    double non_diagonal_sum = 0.0;
    for (int j = 0; j < i; j++)
      non_diagonal_sum += std::abs(dense(i, j));
    for (int j = i + 1; j < dense.cols(); j++)
      non_diagonal_sum += std::abs(dense(j, i));

    if (dense(i, i) < non_diagonal_sum + 1e-6)
      return false;
  }
  return true;
}

template <int k>
void project_hessian(Eigen::Matrix<double, k, k> &hessian,
                     double epsilon = 1e-6) {
  if (is_self_adjoint_positive_diagonally_dominant(hessian))
    return;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, k, k>> eig(hessian);
  const auto &eigs = eig.eigenvalues();
  // Position of the first positive eigenvalue (eigen sorts them in ascending
  // order).
  auto selfadjview = hessian.template selfadjointView<Eigen::Lower>();
  for (int pos = 0; pos < hessian.rows() && eigs(pos) < epsilon; ++pos)
    selfadjview.rankUpdate(eig.eigenvectors().col(pos), epsilon - eigs(pos));
  // Which one is more efficient?
  // ;
  // if (pos == 0)
  //   return;
  // auto neg_vecs = eig.eigenvectors().leftCols(pos);
  // auto proj_vals = (epsilon - eigs.head(pos).array()).matrix().asDiagonal();
  // hessian += neg_vecs * proj_vals * neg_vecs.transpose();
}

extern template void project_hessian(Eigen::Matrix<double, -1, -1> &hessian,
                                     double epsilon = 1e-6);

} // namespace Optiz
