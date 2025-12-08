#include "ElementFunc.h"
#include "VarFactory.h"
#include "VarGrad.h"

#include <Eigen/Eigen>

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

EnergyFunc element_residual(int num, SparseEnergyFunc<VarGrad> delegate){
  return [num, delegate](
             const TGenericVariableFactory<Var> &vars) -> ValueAndDerivatives {
    int num_vars = vars.num_vars();
    double f = 0.0;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_vars);
    std::vector<Eigen::Triplet<double>> triplets;
    TVarFactory<VarGrad> var_factory(vars);
// Aggregate the values.
#pragma omp parallel for schedule(static) reduction(+ : f)                     \
    reduction(merge : triplets) num_threads(omp_get_max_threads() - 1)
    for (int i = 0; i < num; i++) {
      VarGrad res = delegate(i, var_factory);

      // Aggregate the result.
      auto &local_grad = res.grad();
      // Value.
      f += 0.5 * res.val() * res.val();

      // Grad.
      for (const auto &[row, val2] : res.grad()) {
#pragma omp atomic
        grad(row) += val2 * res.val();
      }
      // Approx Hessian.
      int num_referenced = res.grad().get_values().get_std_vector().size();
      for (int j = 0; j < num_referenced; j++) {
        for (int h = 0; h <= j; h++) {
          const auto &[gj, valj] = res.grad().get_values().get_std_vector()[j];
          const auto &[gh, valh] = res.grad().get_values().get_std_vector()[h];
          double val = valj * valh;
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

EnergyFunc grad_element_func(int num, SparseEnergyFunc<VarGrad> delegate) {
  return [num, delegate](
             const TGenericVariableFactory<Var> &vars) -> ValueAndDerivatives {
    double f = 0.0;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(vars.num_vars());
    std::vector<Eigen::Triplet<double>> triplets;
    TVarFactory<VarGrad> var_factory(vars);
// Parallel compute all the values.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num; i++) {
      VarGrad val = delegate(i, var_factory);
      f += val.val();

      for (const auto &[row, val2] : val.grad()) {
#pragma omp atomic
        grad(row) += val2;
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

} // namespace Optiz
