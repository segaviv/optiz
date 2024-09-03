#include "Utils.h"

#include <Eigen/Eigen>
#include <fstream>

namespace Optiz {

EnergyFunc element_func(int num, SparseEnergyFunc<Var> delegate,
                        bool project_hessian) {
  return [num, delegate, project_hessian](
             const TGenericVariableFactory<Var> &vars) -> ValueAndDerivatives {
    double f = 0.0;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(vars.num_vars());
    std::vector<Eigen::Triplet<double>> triplets;
// Parallel compute all the values.
#pragma omp declare reduction(                                                 \
        merge : std::vector<Eigen::Triplet<double>> : omp_out.insert(          \
                omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for schedule(static) reduction(+ : f)                     \
    reduction(merge : triplets)
    for (int i = 0; i < num; i++) {
      Var val = delegate(i, vars);
      f += val.val();

      for (const auto &[row, val2] : val.grad()) {
#pragma omp atomic
        grad(row) += val2;
      }

      if (project_hessian) {
        auto [dense, inds] = project_sparse_hessian(val.hessian());
        for (int i = 0; i < dense.rows(); i++) {
          for (int j = 0; j <= i; j++) {
            if (inds[i] >= inds[j]) {
              triplets.push_back(
                  Eigen::Triplet<double>(inds[i], inds[j], dense(i, j)));
            } else {
              triplets.push_back(
                  Eigen::Triplet<double>(inds[j], inds[i], dense(i, j)));
            }
          }
        }
      } else {
        for (const auto &[ind, val2] : val.hessian()) {
          long row = ind.row, col = ind.col;
          triplets.push_back(Eigen::Triplet<double>(row, col, val2));
        }
      }
    }

    return {f, grad, triplets};
  };
}

GenericEnergyFunc<double> val_func(int num, SparseEnergyFunc<double> delegate) {
  return [num, delegate](const TGenericVariableFactory<double> &vars) {
    double res = 0.0;
// Parallel compute all the values.
#pragma omp parallel for schedule(static) reduction(+ : res)
    for (int i = 0; i < num; i++) {
      res += delegate(i, vars);
    }
    return res;
  };
}

void write_matrix_to_file(const Eigen::MatrixXd &mat,
                          const std::string &file_name) {
  std::ofstream file(file_name);
  file << mat;
  file.close();
}

Eigen::MatrixXd read_matrix_from_file(const std::string &file_name) {
  std::ifstream file(file_name);

  int cols = 0;
  std::string line;
  std::vector<std::vector<double>> mat;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::vector<double> row;
    double val;
    while (ss >> val) {
      row.push_back(val);
    }
    mat.push_back(row);
  }
  Eigen::MatrixXd res(mat.size(), mat[0].size());
  for (int i = 0; i < mat.size(); i++) {
    for (int j = 0; j < mat[0].size(); j++) {
      res(i, j) = mat[i][j];
    }
  }

  file.close();
  return res;
}

} // namespace Optiz
