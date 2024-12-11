#include <NewtonSolver/Problem.h>

#include <Common/Utils.h>
#include <chrono>
#include <tuple>
using namespace std::chrono;

namespace Optiz {

#define VAL_FACTORY(x, state)                                                  \
  block_start_indices.empty()                                                  \
      ? ValFactory<double>(x, _cur_shape, state)                               \
      : VecValFactory(x, block_start_indices, block_shapes)

#define REPORT_ITER_START(i)                                                   \
  time_point<high_resolution_clock> start, stop, stop2, stop3;                 \
  if (_options.report_level == Options::EVERY_STEP) {                          \
    std::cout << "Iteration " << i << " => ";                                  \
    start = high_resolution_clock::now();                                      \
  }

#define REPORT_CALC_TIME(f, filter)                                            \
  if (_options.report_level == Options::EVERY_STEP) {                          \
    stop = high_resolution_clock::now();                                       \
    if (constraints_energies.empty()) {                                        \
      std::cout << "f: " << f << " ("                                          \
                << duration_cast<microseconds>(stop - start).count()           \
                << " us) ";                                                    \
    } else {                                                                   \
      std::cout << "f: " << filter.back().first                                \
                << " cf: " << filter.back().second << " ("                     \
                << duration_cast<microseconds>(stop - start).count()           \
                << " us) ";                                                    \
    }                                                                          \
  }

#define REPORT_SOLVE()                                                         \
  if (_options.report_level == Options::EVERY_STEP) {                          \
    stop2 = high_resolution_clock::now();                                      \
    std::cout << "| Solve ("                                                   \
              << duration_cast<microseconds>(stop2 - stop).count() << " us) "; \
  }

#define REPORT_LINE_SEARCH(step, decrease)                                     \
  if (_options.report_level == Options::EVERY_STEP) {                          \
    stop3 = high_resolution_clock::now();                                      \
    std::cout << (step == 0 ? "| LINE SEARCH FAILED "                          \
                            : ("| step size: " + std::to_string(step)))        \
              << " | Decrease: " << decrease << " ("                           \
              << duration_cast<microseconds>(stop3 - stop2).count() << " us)"  \
              << std::endl;                                                    \
  }

#define REPORT_CONVERGENCE(iter, f, filter)                                    \
  if (_options.report_level != Options::NONE) {                                \
    std::cout << std::endl                                                     \
              << "Converged after " << iter << " iterations"                   \
              << "\n";                                                         \
    std::cout << "Current energy: " << f << "\n";                              \
    if (!constraints_energies.empty()) {                                       \
      std::cout << "Current cf: " << filter.back().second << "\n";             \
      std::cout << "Current f: " << filter.back().first << "\n";               \
      std::cout << "Current f - gamma_theta * cf: "                            \
                << filter.back().first - _options.gamma_theta * f << "\n";     \
    }                                                                          \
  }
#define REPORT_NOT_CONVERGED(iter, f, filter)                                  \
  if (_options.report_level != Options::NONE) {                                \
    std::cout << std::endl                                                     \
              << "No convergence after " << iter << " iterations"              \
              << "\n";                                                         \
    std::cout << "Current energy: " << f << "\n";                              \
    if (!constraints_energies.empty()) {                                       \
      std::cout << "Current cf: " << filter.back().second << "\n";             \
      std::cout << "Current f: " << filter.back().first << "\n";               \
      std::cout << "Current f - gamma_theta * cf: "                            \
                << filter.back().first - _options.gamma_theta * f << "\n";     \
    }                                                                          \
  }

Problem::Problem(const Eigen::MatrixXd &init) : Problem(init, {}) {}
Problem::Problem(const Eigen::MatrixXd &init, const Problem::Options &options)
    : _options(options), first_solve(true),
      _cur(Eigen::Map<const Eigen::VectorXd>(init.data(), init.size())),
      _cur_shape({init.rows(), init.cols()}) {}

Problem::Problem(const std::vector<Eigen::MatrixXd> &init)
    : Problem(init, {}) {}
Problem::Problem(const std::vector<Eigen::MatrixXd> &init,
                 const Options &options)
    : _options(options), first_solve(true) {
  int total_size = 0;
  for (int i = 0; i < init.size(); i++) {
    total_size += init[i].size();
  }
  _cur.resize(total_size, 1);
  // Calculate the start index for each of the blocks.
  block_start_indices.resize(init.size());
  block_shapes.resize(init.size());
  int block_start_index = 0;
  for (int i = 0; i < init.size(); i++) {
    // Store the start index.
    block_start_indices[i] = block_start_index;
    block_shapes[i] = {init[i].rows(), init[i].cols()};
    // And copy the block to the vector.
    _cur.block(block_start_index, 0, init[i].size(), 1) = init[i].reshaped();
    block_start_index += init[i].size();
  }
}

void extract_free_varaibles(Eigen::VectorXd &grad,
                            Eigen::SparseMatrix<double> &hessian,
                            const std::vector<int> &remove_fixed_indices,
                            const std::vector<int> &free_variable_indices) {
  // Update the gradient.
  grad = grad(free_variable_indices).eval();
  // Update the hessian.
  std::vector<Eigen::Triplet<double>> triplets;
  for (int i = 0; i < hessian.outerSize(); i++) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(hessian, i); it; ++it) {
      auto [new_col, new_row] = minmax(remove_fixed_indices[it.row()],
                                       remove_fixed_indices[it.col()]);
      if (new_col != -1) {
        triplets.push_back(
            Eigen::Triplet<double>(new_row, new_col, it.value()));
      }
    }
  }
  hessian.resize(free_variable_indices.size(), free_variable_indices.size());
  hessian.setFromTriplets(triplets.begin(), triplets.end());
}

void compress(Eigen::VectorXd &grad, Eigen::SparseMatrix<double> &hessian,
              std::vector<int> &compress_inds,
              std::vector<int> &uncompress_inds) {
  bool build_inds = uncompress_inds.empty();
  std::vector<Eigen::Triplet<double>> triplets;
  for (int k = 0; k < hessian.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<double>::InnerIterator it(hessian, k); it;
         ++it) {
      if (build_inds && compress_inds[it.row()] == -1) {
        compress_inds[it.row()] = uncompress_inds.size();
        uncompress_inds.push_back(it.row());
      }
      if (build_inds && compress_inds[it.col()] == -1) {
        compress_inds[it.col()] = uncompress_inds.size();
        uncompress_inds.push_back(it.col());
      }
      int new_row_ind =
          std::max(compress_inds[it.row()], compress_inds[it.col()]);
      int new_col_ind =
          std::min(compress_inds[it.row()], compress_inds[it.col()]);
      triplets.push_back(
          Eigen::Triplet<double>(new_row_ind, new_col_ind, it.value()));
    }
  }
  grad = grad(uncompress_inds).eval();
  hessian.resize(uncompress_inds.size(), uncompress_inds.size());
  hessian.setFromTriplets(triplets.begin(), triplets.end());
}

Eigen::VectorXd
uncompress(const Eigen::VectorXd &direction,
           const std::vector<int> &compressed_index_to_uncompressed, int n) {
  Eigen::VectorXd res = Eigen::VectorXd::Zero(n);
  res(compressed_index_to_uncompressed) = direction;
  return res;
}

std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>
calc_energy_with_derivatives(
    const std::vector<Problem::InternalEnergy> &energies,
    const TGenericVariableFactory<Var> &factory) {
  double combined_f = 0.0;
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(factory.num_vars());
  std::vector<Eigen::Triplet<double>> triplets;
  Eigen::SparseMatrix<double> hessian;
  hessian.resize(factory.num_vars(), factory.num_vars());
  for (const auto &energy : energies) {
    const auto &[f, e_grad, e_hessian] = energy.derivatives_func(factory);
    combined_f += f;
    grad.head(e_grad.size()) += e_grad;
    triplets.insert(triplets.end(), e_hessian.begin(), e_hessian.end());
  }
  hessian.setFromTriplets(triplets.begin(), triplets.end());
  return {combined_f, grad, hessian};
}

double calc_energy(const std::vector<Problem::InternalEnergy> &energies,
                   const ValFactory<double> &factory) {
  double combined_f = 0.0;
  for (const auto &energy : energies) {
    combined_f += energy.value_func(factory);
  }
  return combined_f;
}

double calc_non_constraints_energy(
    const std::vector<Problem::InternalEnergy> &energies,
    const std::vector<int> &constraints_energies,
    const ValFactory<double> &factory) {
  double combined_f = 0.0;
  int cur = 0;
  for (int i = 0; i < energies.size(); i++) {
    if (cur < constraints_energies.size() && constraints_energies[cur] == i) {
      cur++;
      continue;
    }
    combined_f += energies[i].value_func(factory);
  }
  return combined_f;
}

double calc_hard_constraints_energy(
    const std::vector<Problem::InternalEnergy> &energies,
    const std::vector<int> &constraints_energies,
    const ValFactory<double> &factory) {
  double combined_f = 0.0;
  for (auto energy_index : constraints_energies) {
    combined_f += energies[energy_index].value_func(factory);
  }
  return combined_f;
}

void Problem::analyze_pattern() {
  if (!constraints_energies.empty()) {
    lu_solver.analyzePattern(_last_hessian);
  } else {
    solver.analyzePattern(_last_hessian);
  }
}

Eigen::VectorXd Problem::factorize_and_solve() {
  if (!constraints_energies.empty()) {
    Eigen::SparseMatrix<double> full_hessian =
        _last_hessian.selfadjointView<Eigen::Lower>();
    lu_solver.factorize(full_hessian);
    return lu_solver.solve(-_last_grad);
  } else {
    solver.factorize(_last_hessian);
    return solver.solve(-_last_grad);
  }
}

Problem &Problem::optimize() {
  // Filter for constrained optimization.
  std::vector<std::pair<double, double>> filter;
  if (!constraints_energies.empty()) {
    filter.push_back(
        {calc_non_constraints_energy(energies, constraints_energies,
                                     VAL_FACTORY(_cur, _state)),
         calc_hard_constraints_energy(energies, constraints_energies,
                                      VAL_FACTORY(_cur, _state))});
  }
  int i = 0;
  std::vector<int> compress_inds, uncompress_inds;
  if (_options.remove_unreferenced) {
    compress_inds = std::vector<int>(_cur.size(), -1);
    uncompress_inds.reserve(_cur.size());
  }
  for (; i < _options.num_iterations; i++) {
    REPORT_ITER_START(i);
    // Calculate the function and its derivatives.
    std::tie(_last_f, _last_grad, _last_hessian) =
        block_start_indices.empty()
            ? calc_energy_with_derivatives(energies,
                                           VarFactory(_cur, _cur_shape, _state))
            : calc_energy_with_derivatives(
                  energies,
                  VecVarFactory(_cur, block_start_indices, block_shapes));
    REPORT_CALC_TIME(_last_f, filter);
    // If remove unreferenced is true, adjust the gradient and hessian.
    if (_options.remove_unreferenced) {
      compress(_last_grad, _last_hessian, compress_inds, uncompress_inds);
    }

    // Check if there's fixed variables.
    if (!free_variables_indices.empty()) {
      extract_free_varaibles(_last_grad, _last_hessian, remove_fixed_mapping,
                             free_variables_indices);
    }

    // Find direction.
    if (!_options.cache_pattern || (_options.cache_pattern && first_solve)) {
      analyze_pattern();
      first_solve = false;
    }
    Eigen::VectorXd d = factorize_and_solve();
    REPORT_SOLVE();
    if (_options.remove_unreferenced && !uncompress_inds.empty()) {
      d = uncompress(d, uncompress_inds, _cur.size());
      _last_grad = uncompress(_last_grad, uncompress_inds, _cur.size());
    }
    if (!free_variables_indices.empty()) {
      d = uncompress(d, free_variables_indices, _cur.size());
      _last_grad = uncompress(_last_grad, free_variables_indices, _cur.size());
    }

    // Find new value.
    double step_size, new_f, decrease;
    if (constraints_energies.empty()) {
      double dir_dot_grad = d.dot(_last_grad);
      std::tie(_cur, _state) =
          line_search(_cur, _last_f, d, dir_dot_grad, step_size, new_f);
      decrease = abs((new_f - _last_f)) / (abs(_last_f) + 1e-9);
    } else {
      auto [prev_f, prev_cf] = filter.back();
      _cur = line_search_constrained(_cur, d, step_size, new_f, filter);
      if (step_size == 0) {
        decrease = 0;
      } else {
        auto [nf, ncf] = filter.back();
        decrease = abs((nf - prev_f) - _options.gamma_theta * (ncf - prev_cf)) /
                   (abs(prev_f) + 1e-9);
      }
    }
    REPORT_LINE_SEARCH(step_size, decrease);
    if (step_size == 0) {
      break;
    }
    if (decrease < _options.relative_change_tolerance) {
      REPORT_CONVERGENCE(i, new_f, filter);
      return *this;
    }
    _end_iteration_callback();
  }
  REPORT_NOT_CONVERGED(i, _last_f, filter);
  return *this;
}

std::tuple<double, Eigen::VectorXd &, Eigen::SparseMatrix<double> &>
Problem::calc_derivatives() {
  std::tie(_last_f, _last_grad, _last_hessian) =
      block_start_indices.empty()
          ? calc_energy_with_derivatives(energies,
                                         VarFactory(_cur, _cur_shape, _state))
          : calc_energy_with_derivatives(
                energies,
                VecVarFactory(_cur, block_start_indices, block_shapes));
  return {_last_f, _last_grad, _last_hessian};
}

double Problem::calc_value(int i) {
  if (i >= 0 && i < energies.size()) {
    return block_start_indices.empty()
               ? energies[i].value_func(ValFactory<double>(_cur, _cur_shape))
               : energies[i].value_func(
                     VecValFactory(_cur, block_start_indices, block_shapes));
  }
  return block_start_indices.empty()
             ? calc_energy(energies, ValFactory<double>(_cur, _cur_shape))
             : calc_energy(energies, VecValFactory(_cur, block_start_indices,
                                                   block_shapes));
}

void Problem::set_end_iteration_callback(std::function<void()> callback) {
  _end_iteration_callback = callback;
}

void Problem::set_fixed_variarbles(const std::vector<int> &indices,
                                   const std::vector<double> &vals) {
  free_variables_indices.clear();
  free_variables_indices.reserve(_cur.size() - indices.size());
  // Marked fixed.
  remove_fixed_mapping = std::vector<int>(_cur.size());
  for (int i = 0; i < indices.size(); i++) {
    remove_fixed_mapping[indices[i]] = -1;
    if (!vals.empty()) {
      _cur(indices[i]) = vals[i];
    }
  }
  // Assign new indices.
  int new_index = 0;
  for (int i = 0; i < remove_fixed_mapping.size(); i++) {
    if (remove_fixed_mapping[i] == -1)
      continue;
    free_variables_indices.push_back(i);
    remove_fixed_mapping[i] = new_index++;
  }
}

void Problem::set_fixed_rows(const std::vector<int> &rows_indices,
                             const Eigen::MatrixXd &vals) {
  if (vals.rows() == rows_indices.size() && vals.cols() == _cur_shape.second) {
    Eigen::Map<Eigen::MatrixXd>(_cur.data(), _cur_shape.first,
                                _cur_shape.second)(rows_indices, Eigen::all) =
        vals;
  }
  std::vector<int> indices(rows_indices.size() * _cur_shape.second);
  for (int j = 0; j < _cur_shape.second; j++) {
    for (int i = 0; i < rows_indices.size(); i++) {
      indices[j * rows_indices.size() + i] =
          j * _cur_shape.first + rows_indices[i];
    }
  }
  set_fixed_variarbles(indices);
}

Eigen::Map<Eigen::MatrixXd> Problem::x() {
  return Eigen::Map<Eigen::MatrixXd>(_cur.data(), _cur_shape.first,
                                     _cur_shape.second);
}

Eigen::Map<Eigen::MatrixXd> Problem::x(int index) {
  return Eigen::Map<Eigen::MatrixXd>(_cur.data() + block_start_indices[index],
                                     block_shapes[index].first,
                                     block_shapes[index].second);
}

Problem::Options &Problem::options() { return _options; }

bool Problem::armijo_cond(double f_curr, double f_x, double step_size,
                          double dir_dot_grad, double armijo_const) {
  return f_x <= f_curr + armijo_const * step_size * dir_dot_grad;
}

std::tuple<Eigen::VectorXd, std::shared_ptr<void>>
Problem::line_search(const Eigen::VectorXd &cur, double f,
                     const Eigen::VectorXd &dir, double dir_dot_grad,
                     double &step_size, double &new_f) {
  step_size = 1.0;
  for (int i = 0; i < _options.line_search_iterations; i++) {
    auto [x, state] = _advance_func(cur, _state, dir, step_size);
    new_f =
        block_start_indices.empty()
            ? calc_energy(energies, ValFactory<double>(x, _cur_shape, state))
            : calc_energy(energies,
                          VecValFactory(x, block_start_indices, block_shapes));
    if (!constraints_energies.empty() ||
        armijo_cond(f, new_f, step_size, dir_dot_grad, 1e-6)) {
      return {x, state};
    }
    step_size *= _options.step_decrease_factor;
  }
  step_size = 0.0;
  return {_cur, _state};
}

Eigen::VectorXd Problem::line_search_constrained(
    const Eigen::VectorXd &cur, const Eigen::VectorXd &dir, double &step_size,
    double &new_f, std::vector<std::pair<double, double>> &filter) {
  // Each iteration should improve either f or cf for each entry in the filter.
  step_size = 1.0;
  double gt = _options.gamma_theta;
  for (int i = 0; i < _options.line_search_iterations; i++) {
    auto [x, state] = _advance_func(cur, _state, dir, step_size);
    auto x_factory = VAL_FACTORY(x, state);
    // Calculate new_f and new_cf.
    new_f =
        calc_non_constraints_energy(energies, constraints_energies, x_factory);
    double new_cf =
        calc_hard_constraints_energy(energies, constraints_energies, x_factory);
    // Check if the new x is acceptible for the filter.
    bool accepted = true;
    for (auto &[f, cf] : filter) {
      if (new_f > f - gt * cf && new_cf > (1 - gt) * cf) {
        accepted = false;
        break;
      }
    }
    // If it is, update the filter and return the new x.
    if (accepted) {
      std::erase_if(filter, [&](auto &p) {
        return new_f - gt * new_cf <= p.first - gt * p.second &&
               new_cf <= p.second;
      });
      filter.push_back({new_f, new_cf});
      return x;
    }
    step_size *= _options.step_decrease_factor;
  }
  step_size = 0.0;
  return _cur;
}

} // namespace Optiz