#include "QuadraticObjectiveCD.h"

namespace Optiz {

QuadraticObjectiveCD::QuadraticObjectiveCD(const std::vector<int> &shape,
                                           int dim)
    : variable_shape(shape), dim(dim) {
  n = 1;
  for (auto &s : shape) {
    n *= s;
  }
}

QuadraticObjectiveCD::QuadraticTerm *
QuadraticObjectiveCD::QuadraticTerm::update_b(
    int num_elements,
    const std::function<void(int, const AddRhsFunc &)> &b_func) {
  int cur_row = 0;
  auto add_rhs = [&](const VarType &val) {
    b_mat(cur_row) = val;
    cur_row++;
  };
  for (int i = 0; i < num_elements; i++) {
    b_func(i, add_rhs);
  }
  rhs_b = weight_unknownA_W * b_mat;
  return this;
}

QuadraticObjectiveCD::QuadraticTerm *QuadraticObjectiveCD::add_quadratic_energy(
    double weight, const SparseMatrix &A, const std::function<MatType()> &b,
    const Eigen::VectorXd &diagonal_weights) {
  return add_quadratic_energy(weight, A, b,
                              sparse_diagonal<double>(diagonal_weights));
}

QuadraticObjectiveCD::QuadraticTerm *QuadraticObjectiveCD::add_quadratic_energy(
    const SparseMatrix &A, const Eigen::VectorXd &weights, double weight) {
  if (weights.size() == 0) {
    _quadratic_terms.emplace_back(std::make_unique<QuadraticTerm>(A, weight));
  } else {
    _quadratic_terms.emplace_back(std::make_unique<QuadraticTerm>(
        A, weight, sparse_diagonal<std::complex<double>>(weights)));
  }
  update_cache(_quadratic_terms.back().get());
  return _quadratic_terms.back().get();
}

QuadraticObjectiveCD &
QuadraticObjectiveCD::add_hard_constraints(const SparseMatrix &C,
                                           const MatType &d) {
  _hard_constraints.push_back({C, d});
  return *this;
}

QuadraticObjectiveCD &
QuadraticObjectiveCD::update_known_values(const MatType &d, int offset) {
  _knowns_vals.block(offset, 0, d.rows(), d.cols()) = d;
  update_known_values_rhs();
  return *this;
}

QuadraticObjectiveCD &
QuadraticObjectiveCD::set_known_indices(const Eigen::VectorXi &indices) {
  _known_indices =
      std::vector<int>(indices.data(), indices.data() + indices.size());
  std::vector<int> _sorted_eq_indices = _known_indices;
  std::sort(_sorted_eq_indices.begin(), _sorted_eq_indices.end());

  int offset = 0;
  _unknown_indices = std::vector<int>(n - _known_indices.size());
  for (int i = 0; i < n; i++) {
    if (offset < _sorted_eq_indices.size() && i == _sorted_eq_indices[offset]) {
      offset++;
    } else {
      _unknown_indices[i - offset] = i;
    }
  }
  _knowns_vals = MatType::Zero(_known_indices.size(), n_cols());
  update_cache();
  return *this;
}

QuadraticObjectiveCD &QuadraticObjectiveCD::set_known_values(const MatType &d) {
  _knowns_vals = d;
  update_known_values_rhs();
  return *this;
}

QuadraticObjectiveCD &
QuadraticObjectiveCD::set_knowns(const Eigen::VectorXi &indices,
                                 const MatType &d) {
  set_known_indices(indices);
  set_known_values(d);
  update_cache();
  return *this;
}

QuadraticObjectiveCD &QuadraticObjectiveCD::prefactor() {
  requires_precompute = false;
  // Aggregate the lhs of the terms.
  SparseMatrix Q(n_free(), n_free());
  for (auto &term : _quadratic_terms) {
    auto unknown_A = unknown(term->A);
    Q += term->weight * unknown_A.adjoint() * term->W * unknown_A;
  }
  if (_hard_constraints.size() == 0) {
    _psd_solver.compute(Q);
    return *this;
  }

  std::vector<std::vector<SparseMatrix>> mats;
  mats.push_back({{Q}});
  for (auto &constraint : _hard_constraints) {
    auto unknown_C = unknown(constraint.C);
    mats[0].push_back(unknown_C.adjoint());
    mats.push_back({{unknown_C}});
  }
  SparseMatrix Q_aug = spcat<std::complex<double>>(mats);
  _lu_solver.compute(Q_aug);
  return *this;
}

QuadraticObjectiveCD::MatType QuadraticObjectiveCD::solve() {
  // Number of hard constraints.
  int n_hard_constraints = 0;
  for (auto &constraint : _hard_constraints) {
    n_hard_constraints += constraint.d.rows();
  }
  // Number of free varialbes.
  int n_free = n - _known_indices.size();
  MatType b = MatType::Zero(n_free + n_hard_constraints, n_cols());
  if (n_free == n) {
    // Make sure known_vals is initialized.
    _knowns_vals = MatType(0, b.cols());
  }

  for (auto &term : _quadratic_terms) {
    if (n_free == n) {
      b.topRows(n) += term->rhs_b;
    } else {
      b.topRows(n_free) += term->rhs_b - term->rhs_known;
    }
  }

  // Add hard constraints equations.
  int row_offset = n_free;
  for (auto &constraint : _hard_constraints) {
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

QuadraticObjectiveCD::MatType QuadraticObjectiveCD::solve(const MatType &b) {
  _quadratic_terms[0]->set_b(b);
  return solve();
}

double QuadraticObjectiveCD::get_sqrd_error(const MatType &x) const {
  double res = 0;
  for (auto &term : _quadratic_terms) {
    res += (term->A * x - term->b()).squaredNorm();
  }
  return res;
}

void QuadraticObjectiveCD::update_known_values_rhs() {
  for (auto &term : _quadratic_terms) {
    term->rhs_known = term->weight_unknownA_W_knownA * _knowns_vals;
  }
}

void QuadraticObjectiveCD::update_cache(QuadraticTerm *term) {
  if (n_free() == n) {
    _knowns_vals = MatType(0, n_cols());
  }
  term->weight_unknownA_W = term->weight * unknown(term->A).adjoint() * term->W;
  term->weight_unknownA_W_knownA = term->weight_unknownA_W * known(term->A);
  term->rhs_b = term->weight_unknownA_W * term->b_mat;
  term->rhs_known = term->weight_unknownA_W_knownA * _knowns_vals;
  requires_precompute = true;
}

void QuadraticObjectiveCD::update_cache() {
  for (auto &term : _quadratic_terms) {
    update_cache(term.get());
  }
}

QuadraticObjectiveCD::SparseMatrix
QuadraticObjectiveCD::known(const SparseMatrix &mat) {
  std::vector<Eigen::Triplet<std::complex<double>>> triplets;
  for (int k = 0; k < _known_indices.size(); ++k) {
    for (typename SparseMatrix::InnerIterator it(mat, _known_indices[k]); it;
         ++it) {
      triplets.push_back(
          Eigen::Triplet<std::complex<double>>(it.row(), k, it.value()));
    }
  }
  SparseMatrix res(mat.rows(), _known_indices.size());
  res.setFromTriplets(triplets.begin(), triplets.end());
  return res;
}

QuadraticObjectiveCD::SparseMatrix
QuadraticObjectiveCD::unknown(const SparseMatrix &mat) {
  if (_known_indices.size() == 0) {
    return mat;
  }
  std::vector<Eigen::Triplet<std::complex<double>>> triplets;
  for (int k = 0; k < _unknown_indices.size(); ++k) {
    for (typename SparseMatrix::InnerIterator it(mat, _unknown_indices[k]); it;
         ++it) {
      triplets.push_back(
          Eigen::Triplet<std::complex<double>>(it.row(), k, it.value()));
    }
  }
  SparseMatrix res(mat.rows(), _unknown_indices.size());
  res.setFromTriplets(triplets.begin(), triplets.end());
  return res;
}

} // namespace Optiz