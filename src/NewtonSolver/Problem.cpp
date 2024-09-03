#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include "Problem.h"

#include <chrono>
#include <tuple>
#include <unordered_map>
using namespace std::chrono;

namespace Optiz {

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

Problem::Problem(const std::vector<Eigen::VectorXd> &init)
    : Problem(init, {}) {}
Problem::Problem(const std::vector<Eigen::VectorXd> &init,
                 const Options &options)
    : _options(options), first_solve(true) {
  int total_size = 0;
  for (int i = 0; i < init.size(); i++) {
    total_size += init[i].size();
  }
  _cur.resize(total_size, 1);
  // Calculate the start index for each of the blocks.
  block_start_indices.resize(init.size());
  int block_start_index = 0;
  for (int i = 0; i < init.size(); i++) {
    // Store the start index.
    block_start_indices[i] = block_start_index;
    // And copy the block to the vector.
    _cur.block(block_start_index, 0, init[i].size(), 1) = init[i];
    block_start_index += init[i].size();
  }
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
    grad += e_grad;
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
    filter.push_back({calc_non_constraints_energy(
                          energies, constraints_energies, val_factory(_cur)),
                      calc_hard_constraints_energy(
                          energies, constraints_energies, val_factory(_cur))});
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
                                           VarFactory(_cur, _cur_shape))
            : calc_energy_with_derivatives(
                  energies, VecVarFactory(_cur, block_start_indices));
    REPORT_CALC_TIME(_last_f, filter);
    // If remove unreferenced is true, adjust the gradient and hessian.
    if (_options.remove_unreferenced) {
      compress(_last_grad, _last_hessian, compress_inds, uncompress_inds);
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

    // Find new value.
    double step_size, new_f, decrease;
    if (constraints_energies.empty()) {
      double dir_dot_grad = d.dot(_last_grad);
      _cur = line_search(_cur, _last_f, d, dir_dot_grad, step_size, new_f);
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
          ? calc_energy_with_derivatives(energies, VarFactory(_cur, _cur_shape))
          : calc_energy_with_derivatives(
                energies, VecVarFactory(_cur, block_start_indices));
  return {_last_f, _last_grad, _last_hessian};
}

double Problem::calc_value(int i) {
  if (i >= 0 && i < energies.size()) {
    return block_start_indices.empty()
               ? energies[i].value_func(ValFactory<double>(_cur, _cur_shape))
               : energies[i].value_func(
                     VecValFactory<double>(_cur, block_start_indices));
  }
  return block_start_indices.empty()
             ? calc_energy(energies, ValFactory<double>(_cur, _cur_shape))
             : calc_energy(energies,
                           VecValFactory<double>(_cur, block_start_indices));
}

void Problem::set_end_iteration_callback(std::function<void()> callback) {
  _end_iteration_callback = callback;
}

Eigen::Map<Eigen::MatrixXd> Problem::x() {
  return Eigen::Map<Eigen::MatrixXd>(_cur.data(), _cur_shape.first,
                                     _cur_shape.second);
}

Problem::Options &Problem::options() { return _options; }

bool Problem::armijo_cond(double f_curr, double f_x, double step_size,
                          double dir_dot_grad, double armijo_const) {
  return f_x <= f_curr + armijo_const * step_size * dir_dot_grad;
}

ValFactory<double> Problem::val_factory(const Eigen::VectorXd &x) const {
  return block_start_indices.empty()
             ? ValFactory<double>(x, _cur_shape)
             : VecValFactory<double>(x, block_start_indices);
}

Eigen::VectorXd Problem::line_search(const Eigen::VectorXd &cur, double f,
                                     const Eigen::VectorXd &dir,
                                     double dir_dot_grad, double &step_size,
                                     double &new_f) {
  step_size = 1.0;
  for (int i = 0; i < _options.line_search_iterations; i++) {
    Eigen::VectorXd x = cur + step_size * dir;
    new_f = calc_energy(energies, val_factory(x));
    if (!constraints_energies.empty() ||
        armijo_cond(f, new_f, step_size, dir_dot_grad, 1e-6)) {
      return x;
    }
    step_size *= _options.step_decrease_factor;
  }
  step_size = 0.0;
  return _cur;
}

Eigen::VectorXd Problem::line_search_constrained(
    const Eigen::VectorXd &cur, const Eigen::VectorXd &dir, double &step_size,
    double &new_f, std::vector<std::pair<double, double>> &filter) {
  // Each iteration should improve either f or cf for each entry in the filter.
  step_size = 1.0;
  double gt = _options.gamma_theta;
  for (int i = 0; i < _options.line_search_iterations; i++) {
    Eigen::VectorXd x = cur + step_size * dir;
    auto x_factory = val_factory(x);
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