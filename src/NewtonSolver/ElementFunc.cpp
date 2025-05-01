#include "ElementFunc.h"

#include <Eigen/Eigen>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <atomic>
#include <tbb/concurrent_vector.h>

namespace Optiz {

EnergyFunc element_func(int num, SparseEnergyFunc<Var> delegate,
                        bool project_hessian) {
  return [num, delegate, project_hessian](
             const TGenericVariableFactory<Var> &vars) -> ValueAndDerivatives {
    std::atomic<double> f = 0.0;
    tbb::concurrent_vector<double> grad_vec(vars.num_vars(), 0.0);
    tbb::concurrent_vector<Eigen::Triplet<double>> triplets;

    tbb::parallel_for(tbb::blocked_range<int>(0, num),
      [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
          Var val = delegate(i, vars);
          f.fetch_add(val.val(), std::memory_order_relaxed);

          for (const auto &[row, val2] : val.grad()) {
            grad_vec[row] += val2;
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
      });

    // Convert concurrent_vector to Eigen::VectorXd and std::vector
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(grad_vec.size());
    for (size_t i = 0; i < grad_vec.size(); ++i) {
      grad(i) = grad_vec[i];
    }
    std::vector<Eigen::Triplet<double>> triplets_vec(triplets.begin(), triplets.end());
    return {f.load(std::memory_order_relaxed), grad, triplets_vec};
  };
}

GenericEnergyFunc<double> val_func(int num, SparseEnergyFunc<double> delegate) {
  return [num, delegate](const TGenericVariableFactory<double> &vars) {
    std::atomic<double> res = 0.0;
    
    tbb::parallel_for(tbb::blocked_range<int>(0, num),
      [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
          res.fetch_add(delegate(i, vars), std::memory_order_relaxed);
        }
      });

    return res.load(std::memory_order_relaxed);
  };
}

} // namespace Optiz
