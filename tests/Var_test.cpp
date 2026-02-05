#include <gtest/gtest.h>

#include "Optiz/Autodiff/Var.h"
#include "Optiz/NewtonSolver/Problem.h"
#include "Optiz/NewtonSolver/VarFactory.h"
#include "test_utils.h"

using Optiz::Var;

TEST(Var, constructor) {
  Var x(5, 2);

  EXPECT_EQ(x.grad().size(), 3);
  EXPECT_EQ(x.grad().get_values().size(), 1);
  EXPECT_EQ(x.grad()(2), 1);
  EXPECT_EQ(x.hessian().get_values().size(), 0);
}

TEST(Var, multiplication) {
  Var x(5, 2);
  Var y(3, 3);

  Var z = 3 * x * y;

  EXPECT_EQ(z.val(), 45);
  EXPECT_EQ(z.grad().size(), 4);
  EXPECT_EQ(z.grad().get_values().size(), 2);
  EXPECT_EQ(z.grad()(2), 9);
  EXPECT_EQ(z.grad()(3), 15);
  EXPECT_EQ(z.hessian().get_values().size(), 1);
  EXPECT_EQ(z.hessian()(3, 2), 3);
}

TEST(Var, division) {
  Var x(6, 2);
  Var y(3, 3);

  Var z = x / y;

  EXPECT_EQ(z.val(), 2);
  EXPECT_EQ(z.grad().size(), 4);
  EXPECT_EQ(z.grad().get_values().size(), 2);
  EXPECT_DOUBLE_EQ(z.grad()(2), 1.0 / 3.0);
  EXPECT_DOUBLE_EQ(z.grad()(3), -6.0 / 9.0);
  EXPECT_EQ(z.hessian().get_values().size(), 2);
  EXPECT_DOUBLE_EQ(z.hessian()(3, 2), -1.0 / 9.0);
  EXPECT_DOUBLE_EQ(z.hessian()(2, 2), 0.0);
  EXPECT_DOUBLE_EQ(z.hessian()(3, 3), 12.0 / 27.0);
}

TEST(Var, addition) {
  Var x(6, 2);
  Var y(3, 3);

  Var z = x + y + 5.0;

  EXPECT_EQ(z.val(), 14);
  EXPECT_EQ(z.grad().size(), 4);
  EXPECT_EQ(z.grad().get_values().size(), 2);
  EXPECT_DOUBLE_EQ(z.grad()(2), 1.0);
  EXPECT_DOUBLE_EQ(z.grad()(3), 1.0);
  EXPECT_EQ(z.hessian().get_values().size(), 0);
}

TEST(Var, subtraction) {
  Var x(6, 2);
  Var y(3, 3);

  Var z = x - y - 2.0;

  EXPECT_EQ(z.val(), 1);
  EXPECT_EQ(z.grad().size(), 4);
  EXPECT_EQ(z.grad().get_values().size(), 2);
  EXPECT_DOUBLE_EQ(z.grad()(2), 1.0);
  EXPECT_DOUBLE_EQ(z.grad()(3), -1.0);
  EXPECT_EQ(z.hessian().get_values().size(), 0);
}

TEST(Var, chain) {
  Var x(0.5, 2);

  Var y = exp(x);

  EXPECT_DOUBLE_EQ(y.val(), std::exp(0.5));
  EXPECT_DOUBLE_EQ(y.grad()(2), std::exp(0.5));
  EXPECT_DOUBLE_EQ(y.hessian()(2, 2), std::exp(0.5));
}

TEST(Var, sin_cos) {
  Var x(M_PI / 4, 2);

  Var y = sin(x) + cos(x);

  EXPECT_DOUBLE_EQ(y.val(), std::sin(M_PI / 4) + std::cos(M_PI / 4));
  EXPECT_DOUBLE_EQ(y.grad()(2), std::cos(M_PI / 4) - std::sin(M_PI / 4));
  EXPECT_DOUBLE_EQ(y.hessian()(2, 2), -std::sin(M_PI / 4) - std::cos(M_PI / 4));
}

TEST(Var, pow_sqr) {
  Var x(2.0, 2);

  Var y = pow(x, 3) + sqr(x);

  EXPECT_DOUBLE_EQ(y.val(), 8.0 + 4.0);
  EXPECT_DOUBLE_EQ(y.grad()(2), 3 * 4.0 + 2 * 2.0);
  EXPECT_DOUBLE_EQ(y.hessian()(2, 2), 3 * 2 * 2.0 + 2);
}

TEST(Var, rotation) {
  // Setup random axis and angle variables.
  Eigen::Vector3d axis = Eigen::Vector3d::Random().normalized();
  Eigen::Vector3d point = Eigen::Vector3d::Random();
  double angle = M_PI / 3;
  Eigen::VectorXd vars(7);
  vars << axis, point, angle;
  // Autodiff variable factory.
  Optiz::VarFactory var_factory(vars, {7, 0});

  auto func = [&](int i, auto &x) {
    using T = FACTORY_TYPE(x);
    Eigen::AngleAxis<T> angle_axis(
        x(6), Eigen::Vector3<T>(x(0), x(1), x(2)).normalized());
    Eigen::Vector3<T> rotated_point =
        angle_axis * Eigen::Vector3<T>(x(3), x(4), x(5));
    return rotated_point(i);
  };

  for (int i = 0; i < 3; i++) {
    auto autodiff = func(i, var_factory);
    // Value check.
    EXPECT_NEAR(autodiff.val(), func(i, vars), 1e-9);

    // Finite difference gradient.
    Eigen::VectorXd fd_grad = finite_difference_gradient(
        [&](const Eigen::VectorXd &v) { return func(i, v); }, vars, 1e-8);
    for (int j = 0; j < 7; j++) {
      EXPECT_NEAR(autodiff.grad()(j), fd_grad(j), 1e-6);
    }
    // Finite difference hessian.
    Eigen::MatrixXd fd_hessian = finite_difference_hessian(
        [&](const Eigen::VectorXd &v) { return func(i, v); }, vars, 1e-4);
    for (int j = 0; j < 7; j++) {
      for (int k = 0; k < 7; k++) {
        EXPECT_NEAR(autodiff.hessian()(j, k), fd_hessian(j, k), 1e-4);
      }
    }
  }
}