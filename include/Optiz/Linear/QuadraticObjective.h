#pragma once
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <Eigen/Eigen>
#include <functional>
#include <memory>
#include <vector>

#include "../Common/SparseUtils.h"
#include "LinearExpression.h"

namespace Optiz {
namespace Linear {

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

struct QuadraticTerm;

template <typename T>
class QuadraticObjective {
  struct HardConstraint {
    Eigen::SparseMatrix<T> C;
    Eigen::MatrixXd d;
  };
  struct QuadraticTerm {
    using SparseMatrix = Eigen::SparseMatrix<T>;
    using MatType = Eigen::MatrixX<T>;

    SparseMatrix A;
    double weight = 1;
    SparseMatrix W = sparse_identity<T>(A.rows());
    std::function<MatType()> b = [&]() { return b_mat; };
    MatType b_mat = MatType::Zero(A.rows(), 1);

    // Cached values.
    // Cached matrix of weight * unknown(A).transpose() * W.
    SparseMatrix weight_unknownA_W;
    // Cached matrix of weight * unknown(A).transpose() * W * known(A).
    SparseMatrix weight_unknownA_W_knownA;
    // Cached rhs.
    MatType rhs_b;
    MatType rhs_known;

    QuadraticTerm* set_b(const MatType& b_mat) {
      this->b_mat = b_mat;
      rhs_b = weight_unknownA_W * b_mat;
      return this;
    }
    QuadraticTerm* set_element_weight(const Eigen::VectorXd& diagonal_weights) {
      W = sparse_diagonal<T>(diagonal_weights);
      return this;
    }
  };

 public:
  using SparseMatrix = Eigen::SparseMatrix<T>;
  using MatType = Eigen::MatrixX<T>;
  QuadraticObjective(int n, int dim = -1)
      : n(n), dim(dim), variable_shape({n}) {}
  QuadraticObjective(const std::vector<int>& shape, int dim = -1)
      : variable_shape(shape), dim(dim) {
    n = 1;
    for (auto& s : shape) {
      n *= s;
    }
  }

  QuadraticTerm* add_quadratic_energy(double weight, const SparseMatrix& A,
                                      const std::function<MatType()>& b,
                                      const Eigen::VectorXd& diagonal_weights) {
    return add_quadratic_energy(weight, A, b,
                                sparse_diagonal<T>(diagonal_weights));
  }

  /*
   * Add a quadratic energy of the form:
   *             weight * ||sqrt(W) (Ax - b)||^2
   */
  QuadraticTerm* add_quadratic_energy(
      const SparseMatrix& A, const Eigen::VectorXd& weights = Eigen::VectorXd(),
      double weight = 1) {
    if (weights.size() == 0) {
      _quadratic_terms.emplace_back(std::make_unique<QuadraticTerm>(A, weight));
    } else {
      _quadratic_terms.emplace_back(std::make_unique<QuadraticTerm>(
          A, weight, sparse_diagonal<T>(weights)));
    }
    update_cache(_quadratic_terms.back().get());
    return _quadratic_terms.back().get();
  }

  QuadraticTerm* add_quadratic_energy(
      int num_elements,
      const std::function<Eigen::VectorX<LinearExpression<T>>(
          int, const Eigen::MatrixX<LinearExpression<T>>&)>& indices) {
    auto [A, b] =
        Optiz::sparse_with_rhs<T>(variable_shape, num_elements, indices);
    QuadraticTerm* term = add_quadratic_energy(A);
    if (b.size() > 0) {
      term->set_b(b);
    }
    return term;
  }

  // Add hard constraints of the form Cx = d
  QuadraticObjective& add_hard_constraints(const SparseMatrix& C,
                                           const MatType& d) {
    _hard_constraints.push_back({C, d});
    return *this;
  }

  QuadraticObjective& add_hard_constraints(
      int num_elements,
      const std::function<Eigen::VectorX<LinearExpression<T>>(
          int, const Eigen::MatrixX<LinearExpression<T>>&)>& indices) {
    auto [A, b] =
        Optiz::sparse_with_rhs<T>(variable_shape, num_elements, indices);
    return add_hard_constraints(A, b);
  }

  QuadraticObjective& update_known_values(const MatType& d, int offset = 0) {
    _knowns_vals.block(offset, 0, d.rows(), d.cols()) = d;
    update_known_values_rhs();
    return *this;
  }

  // Set the known indiecs.
  QuadraticObjective& set_known_indices(const Eigen::VectorXi& indices) {
    _known_indices =
        std::vector<int>(indices.data(), indices.data() + indices.size());
    std::vector<int> _sorted_eq_indices = _known_indices;
    std::sort(_sorted_eq_indices.begin(), _sorted_eq_indices.end());

    int offset = 0;
    _unknown_indices = std::vector<int>(n - _known_indices.size());
    for (int i = 0; i < n; i++) {
      if (offset < _sorted_eq_indices.size() &&
          i == _sorted_eq_indices[offset]) {
        offset++;
      } else {
        _unknown_indices[i - offset] = i;
      }
    }
    _knowns_vals = MatType::Zero(_known_indices.size(), n_cols());
    update_cache();
    return *this;
  }
  // Set known values.
  QuadraticObjective& set_known_values(const MatType& d) {
    _knowns_vals = d;
    update_known_values_rhs();
    return *this;
  }
  // Set known indices and values.
  QuadraticObjective& set_knowns(const Eigen::VectorXi& indices,
                                 const MatType& d) {
    set_known_indices(indices);
    set_known_values(d);
    update_cache();
    return *this;
  }

  /**
   * Prefactorized the system.
   */
  QuadraticObjective& prefactor() {
    // Aggregate the lhs of the terms.
    SparseMatrix Q(n_free(), n_free());
    for (auto& term : _quadratic_terms) {
      auto unknown_A = unknown(term->A);
      Q += term->weight * unknown_A.adjoint() * term->W * unknown_A;
    }
    if (_hard_constraints.size() == 0) {
      _psd_solver.compute(Q);
      return *this;
    }

    std::vector<std::vector<SparseMatrix>> mats;
    mats.push_back({{Q}});
    for (auto& constraint : _hard_constraints) {
      auto unknown_C = unknown(constraint.C);
      mats[0].push_back(unknown_C.adjoint());
      mats.push_back({{unknown_C}});
    }
    SparseMatrix Q_aug = spcat<T>(mats);
    _lu_solver.compute(Q_aug);
    return *this;
  }

  MatType solve() {
    // Number of hard constraints.
    int n_hard_constraints = 0;
    for (auto& constraint : _hard_constraints) {
      n_hard_constraints += constraint.d.rows();
    }
    // Number of free varialbes.
    int n_free = n - _known_indices.size();
    MatType b = MatType::Zero(n_free + n_hard_constraints, n_cols());
    if (n_free == n) {
      // Make sure known_vals is initialized.
      _knowns_vals = MatType(0, b.cols());
    }

    for (auto& term : _quadratic_terms) {
      b.topRows(n_free) += term->rhs_b - term->rhs_known;
    }

    // Add hard constraints equations.
    int row_offset = n_free;
    for (auto& constraint : _hard_constraints) {
      // If the constraint is a scalar, then we need to broadcast it.
      if (constraint.d.cols() != b.cols()) {
        b.block(row_offset, 0, constraint.d.rows(), b.cols()).colwise() =
            constraint.d.col(0);
      } else {
        b.block(row_offset, 0, constraint.d.rows(), b.cols()) =
            constraint.d - known(constraint.C) * _knowns_vals;
      }
      row_offset += constraint.d.rows();
    }

    if (requires_precompute) {
      prefactor();
      requires_precompute = false;
    }

    MatType sol;
    if (_hard_constraints.size() == 0) {
      sol = _psd_solver.solve(b);
    } else {
      sol = _lu_solver.solve(b).eval();
    }
    if (n_free == n) {
      return sol.cols() == 1 && variable_shape.size() > 1
                 ? sol.topRows(n)
                       .reshaped(variable_shape[0], variable_shape[1])
                       .eval()
                 : sol.topRows(n);
    }
    // Add the known vars.
    MatType sol_full(n, sol.cols());
    sol_full(_unknown_indices, Eigen::all) = sol.topRows(n_free);
    sol_full(_known_indices, Eigen::all) = _knowns_vals;
    return sol_full.cols() == 1 && variable_shape.size() > 1
               ? sol_full.reshaped(variable_shape[0], variable_shape[1]).eval()
               : sol_full;
  }
  MatType solve(const MatType& b) {
    _quadratic_terms[0]->set_b(b);
    return solve();
  }

  QuadraticTerm& get_quadratic_term(int index) {
    return *_quadratic_terms[index];
  }

  inline int n_free() const { return n - _known_indices.size(); }
  int n_cols() const {
    return dim > 0 ? dim : _quadratic_terms[0]->b_mat.cols();
  }

  double get_sqrd_error(const MatType& x) const {
    double res = 0;
    for (auto& term : _quadratic_terms) {
      res += (term->A * x - term->b()).squaredNorm();
    }
    return res;
  }

 private:
  void update_known_values_rhs() {
    for (auto& term : _quadratic_terms) {
      term->rhs_known = term->weight_unknownA_W_knownA * _knowns_vals;
    }
  }
  void update_cache(QuadraticTerm* term) {
    if (n_free() == n) {
      _knowns_vals = MatType(0, n_cols());
    }
    term->weight_unknownA_W =
        term->weight * unknown(term->A).adjoint() * term->W;
    term->weight_unknownA_W_knownA = term->weight_unknownA_W * known(term->A);
    term->rhs_b = term->weight_unknownA_W * term->b_mat;
    term->rhs_known = term->weight_unknownA_W_knownA * _knowns_vals;
    requires_precompute = true;
  }

  void update_cache() {
    for (auto& term : _quadratic_terms) {
      update_cache(term.get());
    }
  }
  SparseMatrix known(const SparseMatrix& mat) {
    std::vector<Eigen::Triplet<T>> triplets;
    for (int k = 0; k < _known_indices.size(); ++k) {
      for (typename SparseMatrix::InnerIterator it(mat, _known_indices[k]); it;
           ++it) {
        triplets.push_back(Eigen::Triplet<T>(it.row(), k, it.value()));
      }
    }
    SparseMatrix res(mat.rows(), _known_indices.size());
    res.setFromTriplets(triplets.begin(), triplets.end());
    return res;
  }

  SparseMatrix unknown(const SparseMatrix& mat) {
    if (_known_indices.size() == 0) {
      return mat;
    }
    std::vector<Eigen::Triplet<T>> triplets;
    for (int k = 0; k < _unknown_indices.size(); ++k) {
      for (typename SparseMatrix::InnerIterator it(mat, _unknown_indices[k]);
           it; ++it) {
        triplets.push_back(Eigen::Triplet<T>(it.row(), k, it.value()));
      }
    }
    SparseMatrix res(mat.rows(), _unknown_indices.size());
    res.setFromTriplets(triplets.begin(), triplets.end());
    return res;
  }

 private:
  int n;
  std::vector<int> variable_shape;
  int dim;
  // Equality constraints.
  std::vector<int> _unknown_indices;
  std::vector<int> _known_indices;
  MatType _knowns_vals;

  std::vector<std::unique_ptr<QuadraticTerm>> _quadratic_terms;
  std::vector<HardConstraint> _hard_constraints;

  bool requires_precompute = true;
  Eigen::SimplicialLDLT<SparseMatrix> _psd_solver;
  Eigen::SparseLU<SparseMatrix> _lu_solver;
};

}  // namespace Linear
}  // namespace Optiz
