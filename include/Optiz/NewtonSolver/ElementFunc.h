#pragma once
#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Eigen>
#include <memory>
#include <vector>

#include "../Autodiff/MetaVar.h"
#include "../Autodiff/TDenseVar.h"
#include "../Autodiff/Var.h"
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
using EnergyFunc =
    std::function<ValueAndDerivatives(const TGenericVariableFactory<Var> &)>;
template <typename T>
using GenericEnergyFunc = std::function<T(const TGenericVariableFactory<T> &)>;

EnergyFunc element_func(int num, SparseEnergyFunc<Var> delegate,
                        bool project_hessian = true);

GenericEnergyFunc<double> val_func(int num, SparseEnergyFunc<double> delegate);

template <int k, bool compute_hessian = true> class LocalVarBlockFactory;

template <int k, bool compute_hessian = true> class LocalVarFactory {
public:
  LocalVarFactory(const Eigen::Map<const Eigen::MatrixXd> &other,
                  const std::shared_ptr<void> &state = nullptr,
                  const std::vector<std::pair<int, int>> &block_shapes = {},
                  const std::vector<int> &offsets = {})
      : cur(other), state(state) {
    for (int i = 0; i < block_shapes.size(); i++) {
      sub_factories.push_back(LocalVarBlockFactory<k, compute_hessian>(
          *this, offsets[i], block_shapes[i]));
    }
  }

  using Scalar = TDenseVar<k, compute_hessian>;

  int get_local_index(int global) {
    for (int i = 0; i < num_referenced; i++) {
      if (local_to_global[i] == global)
        return i;
    }
    assert(num_referenced < k);
    local_to_global[num_referenced++] = global;
    return num_referenced - 1;
  }

  Scalar operator()(int i) {
    int local_index = get_local_index(i);
    return Scalar(cur(i), local_index);
  }

  Scalar operator()(int i, int j) { return operator()(i + j * cur.rows()); }

  int num_vars() const { return cur.size(); }

  Eigen::RowVectorX<Scalar> row(int i) {
    Eigen::RowVectorX<Scalar> result(cur.cols());
    int global_index = i;
    for (int j = 0; j < cur.cols(); j++, global_index += cur.rows()) {
      int ind = get_local_index(global_index);
      result(j).val() = cur(global_index);
      result(j).grad()(ind) = 1.0;
    }
    return result;
  }

  template <int m> Eigen::RowVector<Scalar, m> row(int i) {
    Eigen::RowVector<Scalar, m> result;
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

  LocalVarBlockFactory<k, compute_hessian> &var_block(int i) {
    return sub_factories[i];
  }

  Eigen::Map<const Eigen::MatrixXd> cur;
  std::shared_ptr<void> state;
  int local_to_global[k];
  int num_referenced = 0;
  std::vector<LocalVarBlockFactory<k, compute_hessian>> sub_factories;
};

template <int k, bool compute_hessian> class LocalVarBlockFactory {
public:
  using Scalar = TDenseVar<k, compute_hessian>;
  LocalVarFactory<k, compute_hessian> &parent_factory;
  int offset;
  const std::pair<int, int> block_shape;
  LocalVarBlockFactory(LocalVarFactory<k, compute_hessian> &parent_factory,
                       int offset, const std::pair<int, int> &block_shape)
      : parent_factory(parent_factory), offset(offset),
        block_shape(block_shape) {}

  Scalar operator()(int i) { return parent_factory(i + offset); }

  Scalar operator()(int i, int j) {
    return parent_factory(i + j * block_shape.first + offset);
  }

  Eigen::RowVectorX<Scalar> row(int i) {
    Eigen::RowVectorX<Scalar> result(block_shape.second);
    int global_index = i;
    for (int j = 0; j < block_shape.second;
         j++, global_index += block_shape.first) {
      int ind = parent_factory.get_local_index(offset + global_index);
      result(j).val() = parent_factory.cur(offset + global_index);
      result(j).grad()(ind) = 1.0;
    }
    return result;
  }

  template <int m> Eigen::RowVector<Scalar, m> row(int i) {
    Eigen::RowVector<Scalar, m> result;
    int global_index = i;
    for (int j = 0; j < m; j++, global_index += block_shape.first) {
      int ind = parent_factory.get_local_index(offset + global_index);
      result(j).val() = parent_factory.cur(offset + global_index);
      result(j).grad()(ind) = 1.0;
    }
    return result;
  }
};

template <int k, bool compute_hessian = true>
EnergyFunc element_func(int num, auto delegate, bool project_hessian = true) {
  return [num, delegate, project_hessian](
             const TGenericVariableFactory<Var> &vars) -> ValueAndDerivatives {
    int num_vars = vars.num_vars();
    double f = 0.0;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_vars);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(k * k * num);
// Aggregate the values.
#pragma omp parallel for schedule(static) reduction(+ : f)                     \
    reduction(merge : triplets) num_threads(omp_get_max_threads() - 1)
    for (int i = 0; i < num; i++) {
      LocalVarFactory<k, compute_hessian> local_vars(
          vars.current_mat(), vars.get_state(), vars.block_shapes,
          vars.offsets);
      TDenseVar<k, compute_hessian> res = delegate(i, local_vars);
      if constexpr (compute_hessian) {
        if (project_hessian) {
          res.projectHessian();
        }
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
      if constexpr (compute_hessian) {
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
    }
    return {f, grad, triplets};
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

class LocalMetaVarBlockFactory {
public:
  using Scalar = MetaVarScalar;
  int offset;
  const std::pair<int, int> block_shape;
  int *local_to_global;
  int &num_referenced;
  Eigen::Map<const Eigen::MatrixXd> cur;

  template <auto ind = int{}, typename = decltype([] {})>
  LocalMetaVarBlockFactory(int offset, const std::pair<int, int> &block_shape,
                           int *local_to_global, int &num_referenced,
                           const Eigen::Map<const Eigen::MatrixXd> &cur)
      : offset(offset), block_shape(block_shape),
        local_to_global(local_to_global), num_referenced(num_referenced),
        cur(cur) {}

  template <auto ind = int{}, typename = decltype([] {})>
  auto operator()(int i) {
    local_to_global[num_referenced++] = i + offset;
    return Optiz::MetaVar<next_meta_var_id<ind>()>(cur(i + offset));
  }

  template <auto ind = int{}, typename = decltype([] {})>
  auto operator()(int i, int j) {
    int index = i + j * block_shape.first + offset;
    local_to_global[num_referenced++] = index;
    return Optiz::MetaVar<next_meta_var_id<ind>()>(cur(index));
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
};

class LocalMetaVarFactory {
public:
  using Scalar = MetaVarScalar;
  template <auto ind = int{}, typename = decltype([] {})>
  LocalMetaVarFactory(const Eigen::Map<const Eigen::MatrixXd> &other,
                      const std::shared_ptr<void> &state = nullptr,
                      const std::vector<std::pair<int, int>> &block_shapes = {},
                      const std::vector<int> &offsets = {})
      : cur(other), state(state) {
    for (int i = 0; i < block_shapes.size(); i++) {
      sub_factories.push_back(LocalMetaVarBlockFactory(
          offsets[i], block_shapes[i], local_to_global, num_referenced, cur));
    }
  }

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

  LocalMetaVarBlockFactory &var_block(int i) { return sub_factories[i]; }

  Eigen::Map<const Eigen::MatrixXd> cur;
  std::shared_ptr<void> state;
  int local_to_global[20];
  int num_referenced = 0;
  std::vector<LocalMetaVarBlockFactory> sub_factories;
};

EnergyFunc meta_element_func(int num, auto delegate, bool hessian_proj = true) {
  return [num, delegate, hessian_proj](
             const TGenericVariableFactory<Var> &vars) -> ValueAndDerivatives {
    int num_vars = vars.num_vars();
    double f = 0.0;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_vars);
    std::vector<Eigen::Triplet<double>> triplets;
// Parallel compute all the values.
#pragma omp parallel for schedule(static) reduction(+ : f)                     \
    reduction(merge : triplets) num_threads(omp_get_max_threads() - 1)
    for (int i = 0; i < num; i++) {
      LocalMetaVarFactory local_vars(vars.current_mat(), vars.get_state(),
                                     vars.block_shapes, vars.offsets);
      auto res = delegate(i, local_vars);
      f += res.val();
      // Grad.
      auto local_grad = res.meta_grad();
      constexpr int M = decltype(local_grad)::first_var();
      local_grad.for_each([&](const auto &elem) {
#pragma omp atomic
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
    return {f, grad, triplets};
  };
};

class MetaValFactory {
public:
  using Scalar = MetaVarScalar;
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
  auto &var_block(int i) { return *this; }

  Eigen::Map<const Eigen::MatrixXd> cur;
  std::shared_ptr<void> state;
};

GenericEnergyFunc<double> meta_val_func(int num, auto delegate) {
  return [num, delegate](const TGenericVariableFactory<double> &vars) {
    double res = 0.0;
    MetaValFactory fac(vars.current_mat(), vars.get_state());
// Aggregate the values.
#pragma omp parallel for schedule(static) reduction(+ : res)
    for (int i = 0; i < num; i++) {
      res += delegate(i, fac);
    }
    return res;
  };
}

} // namespace Optiz