#include "ProjectHessian.h"

#include <tuple>
#include <unordered_map>
#include <vector>

namespace Optiz {

std::tuple<
    std::unordered_map<int, int>,
    std::vector<int>> static find_referenced_indices(const SelfAdjointMapMatrix
                                                         &mat) {
  std::unordered_map<int, int> sp_to_dense;
  int new_index = 0;
  for (const auto &val : mat.get_values()) {
    int row = val.first.row, col = val.first.col;
    if (sp_to_dense.try_emplace(row, new_index).second) {
      new_index++;
    }
    if (sp_to_dense.try_emplace(col, new_index).second) {
      new_index++;
    }
  }
  // Create a mapping from sparse indices to dense indices.
  std::vector<int> dense_to_sp(new_index);
  for (auto &el : sp_to_dense) {
    dense_to_sp[el.second] = el.first;
  }
  return {sp_to_dense, dense_to_sp};
}

static std::tuple<Eigen::MatrixX<double>, std::vector<int>>
sparse_to_dense(const SelfAdjointMapMatrix &mat) {
  auto [sp_to_dense, dense_to_sp] = find_referenced_indices(mat);

  Eigen::MatrixX<double> res =
      Eigen::MatrixX<double>::Zero(sp_to_dense.size(), sp_to_dense.size());
  for (const auto &val : mat.get_values()) {
    int row = val.first.row, col = val.first.col;
    int r = sp_to_dense[row], c = sp_to_dense[col];
    res(r, c) = val.second;
    res(c, r) = val.second;
  }
  return {res, dense_to_sp};
}

static SelfAdjointMapMatrix
dense_to_sparse_selfadj(const Eigen::MatrixX<double> &dense,
                        const std::vector<int> &dense_to_sp, int n) {
  std::vector<Eigen::Triplet<double>> triplets;
  SelfAdjointMapMatrix res(n);
  for (int i = 0; i < dense.rows(); i++) {
    for (int j = 0; j <= i; j++) {
      if (dense_to_sp[i] > dense_to_sp[j]) {
        res(dense_to_sp[i], dense_to_sp[j]) = dense(i, j);
      } else {
        res(dense_to_sp[j], dense_to_sp[i]) = dense(i, j);
      }
    }
  }
  return res;
}

SelfAdjointMapMatrix project_hessian(const SelfAdjointMapMatrix &hessian) {
  auto [dense, dense_to_sp] = sparse_to_dense(hessian);
  project_hessian(dense);
  return dense_to_sparse_selfadj(dense, dense_to_sp, hessian.rows());
}

std::pair<Eigen::MatrixXd, std::vector<int>>
project_sparse_hessian(const SelfAdjointMapMatrix &hessian) {
  auto [dense, dense_to_sp] = sparse_to_dense(hessian);
  project_hessian(dense);
  return {dense, dense_to_sp};
}

// template void project_hessian(Eigen::Matrix<double, -1, -1> &hessian,
//                               double epsilon);

} // namespace Optiz
