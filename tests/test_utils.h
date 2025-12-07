#pragma once
#include <Eigen/Eigen>

Eigen::VectorXd finite_difference_gradient(
    const std::function<double(const Eigen::VectorXd &)> &func,
    const Eigen::VectorXd &x, double h = 1e-6);

Eigen::MatrixXd finite_difference_hessian(
    const std::function<double(const Eigen::VectorXd &)> &func,
    const Eigen::VectorXd &x, double h = 1e-6);