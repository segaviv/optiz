#include "test_utils.h"

Eigen::VectorXd finite_difference_gradient(
    const std::function<double(const Eigen::VectorXd &)> &func,
    const Eigen::VectorXd &x, double h) {
  Eigen::VectorXd grad(x.size());
  double f_x = func(x);
  for (int i = 0; i < x.size(); i++) {
    grad(i) = (func(x + Eigen::VectorXd::Unit(x.size(), i) * h) -
               func(x - Eigen::VectorXd::Unit(x.size(), i) * h)) /
              (2 * h);
  }
  return grad;
}

Eigen::MatrixXd finite_difference_hessian(
    const std::function<double(const Eigen::VectorXd &)> &func,
    const Eigen::VectorXd &x, double h) {
  Eigen::MatrixXd hessian(x.size(), x.size());
  for (int i = 0; i < x.size(); i++) {
    Eigen::VectorXd f_x = finite_difference_gradient(func, x - Eigen::VectorXd::Unit(x.size(), i) * h, h);
    hessian.col(i) = (finite_difference_gradient(
                          func, x + Eigen::VectorXd::Unit(x.size(), i) * h, h) -
                      f_x) /
                     (2 * h);
  }
  return hessian;
}