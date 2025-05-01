#pragma once
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <atomic>
#include <tbb/concurrent_vector.h>

#include <Eigen/Eigen>
#include <vector>
#include <memory>

#include "../Autodiff/MetaVar.h"
#include "../Autodiff/TDenseVar.h"
#include "../Autodiff/Var.h"
#include "VarFactory.h"

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
using EnergyFunc =
    std::function<ValueAndDerivatives(const TGenericVariableFactory<Var> &)>;
template <typename T>
using GenericEnergyFunc = std::function<T(const TGenericVariableFactory<T> &)>;

EnergyFunc element_func(int num, SparseEnergyFunc<Var> delegate,
                        bool project_hessian = true);

GenericEnergyFunc<double> val_func(int num, SparseEnergyFunc<double> delegate);

template <int k> class LocalVarFactory {
public:
  LocalVarFactory(const Eigen::Map<const Eigen::MatrixXd> &other,
                  const std::shared_ptr<void> &state = nullptr)
      : cur(other), state(state) {}

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

  template <typename State> const State &get_state() const {
    return *static_cast<const State *>(state.get());
  }

  Eigen::Map<const Eigen::MatrixXd> cur;
  std::shared_ptr<void> state;
  int local_to_global[k];
  int num_referenced = 0;
};

template <int k>
using LocalEnergyFunction =
    std::function<TDenseVar<k>(int index, LocalVarFactory<k> &)>;

template <int k>
EnergyFunc element_func(int num, LocalEnergyFunction<k> delegate,
                        bool project_hessian = true) {
  return [num, delegate, project_hessian](
             const TGenericVariableFactory<Var> &vars) -> ValueAndDerivatives {
    int num_vars = vars.num_vars();
    std::atomic<double> f = 0.0;
    tbb::concurrent_vector<double> grad_vec(num_vars, 0.0);
    tbb::concurrent_vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(k * k * num);

    tbb::parallel_for(tbb::blocked_range<int>(0, num),
      [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
          LocalVarFactory<k> local_vars(vars.current_mat(), vars.get_state());
          TDenseVar<k> res = delegate(i, local_vars);
          if (project_hessian) {
            res.projectHessian();
          }

          // Aggregate the result.
          auto &local_grad = res.grad();
          auto &local_hessian = res.hessian();
          // Value.
          f.fetch_add(res.val(), std::memory_order_relaxed);

          // Grad.
          for (int j = 0; j < local_vars.num_referenced; j++) {
            grad_vec[local_vars.local_to_global[j]] += local_grad(j);
          }
          // Hessian.
          for (int j = 0; j < local_vars.num_referenced; j++) {
            for (int h = 0; h <= j; h++) {
              int gj = local_vars.local_to_global[j],
                  gh = local_vars.local_to_global[h];
              double val = local_hessian(j, h);
              // Only fill the lower triangle of the hessian.
              if (gj <= gh) {
                triplets.push_back(Eigen::Triplet<double>(gh, gj, val));
              } else {
                triplets.push_back(Eigen::Triplet<double>(gj, gh, val));
              }
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
};

template <auto Id> struct meta_var_counter {
  struct generator {
    friend consteval auto is_defined(meta_var_counter) { return true; }
  };
  friend consteval auto is_defined(meta_var_counter);

  template <typename Tag = meta_var_counter, auto = is_defined(Tag{})>
  static consteval auto exists(auto) {
    return true;
  }

  static consteval auto exists(...) { return generator(), false; }
};

template <auto Id = int{}, typename = decltype([] {})>
consteval auto next_meta_var_id() {
  if constexpr (!meta_var_counter<Id>::exists(Id))
    return Id;
  else
    return next_meta_var_id<Id + 1>();
}

class LocalMetaVarFactory {
public:
  template <auto ind = int{}, typename = decltype([] {})>
  LocalMetaVarFactory(const Eigen::Map<const Eigen::MatrixXd> &other,
                      const std::shared_ptr<void> &state = nullptr)
      : cur(other), state(state) {}

  int num_vars() const { return cur.size(); }

  template <auto ind = int{}, typename = decltype([] {})>
  decltype(auto) operator()(int i) {
    local_to_global[num_referenced++] = i;
    return Optiz::MetaVar<next_meta_var_id<ind>()>(cur(i));
  }

  template <auto ind = int{}, typename = decltype([] {})>
  decltype(auto) operator()(int i, int j) {
    local_to_global[num_referenced++] = i + j * cur.rows();
    return Optiz::MetaVar<next_meta_var_id<ind>()>(cur(i + j * cur.rows()));
  }

  template <int m, int start = 0, typename... Args, auto ind = int{},
            typename = decltype([] {})>
  decltype(auto) row(int i, const MetaVec<Args...> &vec = MetaVec<>()) {
    if constexpr (m == start) {
      return vec;
    } else {
      auto var = operator()<ind>(i, start);
      auto res = row<m, start + 1>(i, vec.push(var));
      return res;
    }
  }

  template <typename State> const State &get_state() const {
    return *static_cast<const State *>(state.get());
  }

  Eigen::Map<const Eigen::MatrixXd> cur;
  std::shared_ptr<void> state;
  int local_to_global[20];
  int num_referenced = 0;
};

EnergyFunc meta_element_func(int num, auto delegate, bool hessian_proj = true) {
  return [num, delegate, hessian_proj](
             const TGenericVariableFactory<Var> &vars) -> ValueAndDerivatives {
    int num_vars = vars.num_vars();
    std::atomic<double> f = 0.0;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_vars);
    std::vector<Eigen::Triplet<double>> triplets;

    tbb::parallel_for(tbb::blocked_range<int>(0, num),
      [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
          LocalMetaVarFactory local_vars(vars.current_mat(), vars.get_state());
          auto res = delegate(i, local_vars);
          f.fetch_add(res.val(), std::memory_order_relaxed);
          // Grad.
          auto local_grad = res.meta_grad();
          constexpr int M = decltype(local_grad)::first_var();
          local_grad.for_each([&](const auto &elem) {
            grad(local_vars.local_to_global[TYPE(elem)::Index - M]) += elem.val;
          });

          // Hessian.
          auto local_hessian = res.squeezed_hessian();
          if (hessian_proj) {
            project_hessian(local_hessian);
          }
          for (int j = 0; j < local_hessian.cols(); j++) {
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
      });

    return {f.load(std::memory_order_relaxed), grad, triplets};
  };
};

class MetaValFactory {
public:
  template <auto ind = int{}, typename = decltype([] {})>
  MetaValFactory(const Eigen::Map<const Eigen::MatrixXd> &other,
                 const std::shared_ptr<void> &state = nullptr)
      : cur(other), state(state) {}

  int num_vars() const { return cur.size(); }

  decltype(auto) operator()(int i) { return cur(i); }

  decltype(auto) operator()(int i, int j) { return cur(i + j * cur.rows()); }

  template <int m, int start = 0, typename... Args>
  decltype(auto) row(int i, const MetaVec<Args...> &vec = MetaVec<>()) {
    if constexpr (m == start) {
      return vec;
    } else {
      auto var = operator()(i, start);
      auto res = row<m, start + 1>(i, vec.push(var));
      return res;
    }
  }

  template <typename State> const State &get_state() const {
    return *static_cast<const State *>(state.get());
  }

  Eigen::Map<const Eigen::MatrixXd> cur;
  std::shared_ptr<void> state;
};

GenericEnergyFunc<double> meta_val_func(int num, auto delegate) {
  return [num, delegate](const TGenericVariableFactory<double> &vars) {
    std::atomic<double> res = 0.0;
    MetaValFactory fac(vars.current_mat(), vars.get_state());
    
    tbb::parallel_for(tbb::blocked_range<int>(0, num),
      [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
          res.fetch_add(delegate(i, fac), std::memory_order_relaxed);
        }
      });

    return res.load(std::memory_order_relaxed);
  };
}

} // namespace Optiz