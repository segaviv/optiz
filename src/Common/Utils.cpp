#include <Common/Utils.h>
#include <iomanip>
#include <map>
#include <ostream>
#include <set>
#include <fstream>
#include <sstream>

namespace Optiz {
std::map<std::string, Var> vars_mapping;
std::vector<std::string> names;

Var &get_var(const std::string &name) {
  if (vars_mapping.count(name) == 0) {
    int s = vars_mapping.size();
    vars_mapping[name] = Var(0.0, s);
    names.push_back(name);
    return vars_mapping[name];
  }
  return vars_mapping[name];
}

Eigen::MatrixX<Var> create_variables(const Eigen::MatrixXd &values) {
  Eigen::MatrixX<Var> res(values.rows(), values.cols());
  for (int i = 0; i < values.rows(); i++) {
    for (int j = 0; j < values.cols(); j++) {
      res(i, j) = Var(values(i, j), j * values.rows() + i);
    }
  }
  return res;
}

std::string get_var_name(int val) {
  std::stringstream ss;
  ss << "d/d";
  if (val < names.size()) {
    ss << names[val];
  } else {
    ss << val;
  }
  return ss.str();
}

std::string pad(const std::string& s, int n) {
  std::stringstream ss;
  for (int i = 0; i < (n - s.size()) / 2; i++) {
    ss << " ";
  }
  ss << s;
  return ss.str();
}

std::string str(const Var &var) {
  std::set<int> sorted_indices;
  for (auto val : var.grad().get_values())
    sorted_indices.insert(val.first);

  std::stringstream ss;
  ss << "Val: " << var.val() << std::endl;

  ss << std::endl << "Grad: " << std::endl;
  for (auto val : sorted_indices) {
    ss << std::setw(10) << std::left << get_var_name(val) + ": ";
    ss << std::setw(10) << std::left << std::setprecision(4) << var.grad()(val)
       << std::endl;
  }

  ss << std::endl << "Hessian: " << std::endl;
  ss << std::setw(10) << "";
  for (auto val : sorted_indices) {
    ss << std::setw(10) << std::left << pad(get_var_name(val), 10);
  }
  ss << std::endl;
  for (auto val : sorted_indices) {
    ss <<std::setw(10) << std::left << get_var_name(val) + ": ";
    for (auto val2 : sorted_indices) {
      ss << std::setw(10) << std::setprecision(4) << var.hessian()(val, val2)
         << " ";
    }
    ss << std::endl;
  }
  return ss.str();
}

std::pair<long, long> minmax(long a, long b) {
  if (a < b)
    return {a, b};
  return {b, a};
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