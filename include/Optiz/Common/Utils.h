#pragma once
#include "../Autodiff/Var.h"
#include <Eigen/Eigen>

#define VAR1(x) Optiz::get_var(#x)
#define VAR2(x, val) Optiz::get_var(#x) = val
#define GET_MACRO(_1,_2,NAME,...) NAME
#define VAR(...) GET_MACRO(__VA_ARGS__, VAR2, VAR1)(__VA_ARGS__)

namespace Optiz {
  Var& get_var(const std::string& name);
  Eigen::MatrixX<Var> create_variables(const Eigen::MatrixXd& values);
  std::string str(const Var& var);
  std::pair<long, long> minmax(long a, long b);
  void write_matrix_to_file(const Eigen::MatrixXd &mat,
                          const std::string &file_name);

Eigen::MatrixXd read_matrix_from_file(const std::string &file_name);
}