#pragma once
#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Eigen>
#include <string>
#include <vector>

#include "TDenseVar.h"
#include "Var.h"
#include "VarFactory.h"

#pragma omp declare reduction(                                                 \
        merge : std::vector<Eigen::Triplet<double>> : omp_out.insert(          \
                omp_out.end(), omp_in.begin(), omp_in.end()))

namespace Optiz {

using DenseValueAndDerivatives =
    std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd>;
using ValueAndDerivatives =
    std::tuple<double, Eigen::VectorXd, std::vector<Eigen::Triplet<double>>>;

template <typename T>
using SparseEnergyFunc =
    std::function<T(int index, const TGenericVariableFactory<T> &)>;
using SparseVarEnergyFunc =
    std::function<Var(int index, const TGenericVariableFactory<Var> &)>;
using EnergyFunc = std::function<ValueAndDerivatives(
    const TGenericVariableFactory<Var> &)>;
template <typename T>
using GenericEnergyFunc = std::function<T(const TGenericVariableFactory<T> &)>;

EnergyFunc element_func(int num, SparseEnergyFunc<Var> delegate,
                        bool project_hessian = true);

GenericEnergyFunc<double> val_func(int num, SparseEnergyFunc<double> delegate);

template <int k> class LocalVarFactory {
public:
  LocalVarFactory(const Eigen::Map<const Eigen::MatrixXd> &other)
      : cur(other) {}

  using Scalar = TDenseVar<k>;

  int get_local_index(int global) {
    for (int i = 0; i < num_referenced; i++) {
      if (local_to_global[i] == global)
        return i;
    }
    assert(num_referenced < k);
    local_to_global[num_referenced++] = global;
    return num_referenced - 1;
  }

  TDenseVar<k> operator()(int i) {
    int local_index = get_local_index(i);
    return TDenseVar<k>(cur(i), local_index);
  }

  TDenseVar<k> operator()(int i, int j) {
    return operator()(i + j * cur.rows());
  }

  int num_vars() const { return cur.size(); }

  Eigen::RowVectorX<TDenseVar<k>> row(int i) {
    Eigen::RowVectorX<TDenseVar<k>> result(cur.cols());
    int global_index = i;
    for (int j = 0; j < cur.cols(); j++, global_index += cur.rows()) {
      int ind = get_local_index(global_index);
      result(j).val() = cur(global_index);
      result(j).grad()(ind) = 1.0;
    }
    return result;
  }

  template <int m> Eigen::RowVector<TDenseVar<k>, m> row(int i) {
    Eigen::RowVector<TDenseVar<k>, m> result;
    int global_index = i;
    for (int j = 0; j < m; j++, global_index += cur.rows()) {
      int ind = get_local_index(global_index);
      result(j).val() = cur(global_index);
      result(j).grad()(ind) = 1.0;
    }
    return result;
  }

  Eigen::Map<const Eigen::MatrixXd> cur;
  int local_to_global[k];
  int num_referenced = 0;
};

template <int k>
using LocalEnergyFunction =
    std::function<TDenseVar<k>(int index, LocalVarFactory<k> &)>;

template <int k>
EnergyFunc element_func(int num, LocalEnergyFunction<k> delegate,
                        bool project_hessian = true) {
  return
      [num, delegate, project_hessian](const TGenericVariableFactory<Var> &vars)
          -> ValueAndDerivatives {
        int num_vars = vars.num_vars();
        double f = 0.0;
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_vars);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(k * k * num);
// Parallel compute all the values.
#pragma omp parallel for schedule(static) reduction(+ : f)                     \
    reduction(merge : triplets) num_threads(omp_get_max_threads() - 1)
        for (int i = 0; i < num; i++) {
          LocalVarFactory<k> local_vars(vars.current_mat());
          TDenseVar<k> res = delegate(i, local_vars);
          if (project_hessian) {
            res.projectHessian();
          }

          // Aggregate the result.
          auto &local_grad = res.grad();
          auto &local_hessian = res.hessian();
          // Value.
          f += res.val();

          // Grad.
          for (int j = 0; j < local_vars.num_referenced; j++) {
#pragma omp atomic
            grad(local_vars.local_to_global[j]) += local_grad(j);
          }
          // Hessian.
          for (int j = 0; j < local_vars.num_referenced; j++) {
            for (int h = 0; h <= j; h++) {
              int gj = local_vars.local_to_global[j],
                  gh = local_vars.local_to_global[h];
              double val = local_hessian(j, h);
              // Only fill the lower triangle of the hessian.
              if (gj <= gh) {
                triplets.emplace_back(gh, gj, val);
              } else {
                triplets.emplace_back(gj, gh, val);
              }
            }
          }
        }
        return {f, grad, triplets};
      };
};

void write_matrix_to_file(const Eigen::MatrixXd &mat,
                          const std::string &file_name);

Eigen::MatrixXd read_matrix_from_file(const std::string &file_name);

} // namespace Optiz