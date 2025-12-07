#include <gtest/gtest.h>

#include "Optiz/Autodiff/TDenseVar.h"
#include "Optiz/NewtonSolver/Problem.h"
#include "Optiz/NewtonSolver/VarFactory.h"
#include "test_utils.h"

using Optiz::TDenseVar;

TEST(TDenseVar, constructor) {
  TDenseVar<3> x(5, 2);

  EXPECT_EQ(x.grad().size(), 3);
  EXPECT_EQ(x.grad()(2), 1);
  EXPECT_DOUBLE_EQ(x.hessian().norm(), 0);
}

TEST(TDenseVar, multiplication) {
  TDenseVar<4> x(5, 2);
  TDenseVar<4> y(3, 3);

  TDenseVar<4> z = 3 * x * y;

  EXPECT_EQ(z.val(), 45);
  EXPECT_EQ(z.grad().size(), 4);
  EXPECT_EQ(z.grad()(2), 9);
  EXPECT_EQ(z.grad()(3), 15);
  EXPECT_EQ(z.hessian()(3, 2), 3);
}

TEST(TDenseVar, division) {
  TDenseVar<4> x(6, 2);
  TDenseVar<4> y(3, 3);

  TDenseVar<4> z = x / y;

  EXPECT_EQ(z.val(), 2);
  EXPECT_EQ(z.grad().size(), 4);
  EXPECT_DOUBLE_EQ(z.grad()(2), 1.0 / 3.0);
  EXPECT_DOUBLE_EQ(z.grad()(3), -6.0 / 9.0);
  EXPECT_DOUBLE_EQ(z.hessian()(3, 2), -1.0 / 9.0);
  EXPECT_DOUBLE_EQ(z.hessian()(2, 2), 0.0);
  EXPECT_DOUBLE_EQ(z.hessian()(3, 3), 12.0 / 27.0);
}

TEST(TDenseVar, addition) {
  TDenseVar<4> x(6, 2);
  TDenseVar<4> y(3, 3);

  TDenseVar<4> z = x + y + 5.0;

  EXPECT_EQ(z.val(), 14);
  EXPECT_EQ(z.grad().size(), 4);
  EXPECT_DOUBLE_EQ(z.grad()(2), 1.0);
  EXPECT_DOUBLE_EQ(z.grad()(3), 1.0);
  EXPECT_DOUBLE_EQ(z.hessian().norm(), 0);
}

TEST(TDenseVar, subtraction) {
  TDenseVar<4> x(6, 2);
  TDenseVar<4> y(3, 3);

  TDenseVar<4> z = x - y - 2.0;

  EXPECT_EQ(z.val(), 1);
  EXPECT_EQ(z.grad().size(), 4);
  EXPECT_DOUBLE_EQ(z.grad()(2), 1.0);
  EXPECT_DOUBLE_EQ(z.grad()(3), -1.0);
  EXPECT_DOUBLE_EQ(z.hessian().norm(), 0);
}

TEST(TDenseVar, chain) {
  TDenseVar<4> x(0.5, 2);

  TDenseVar<4> y = exp(x);

  EXPECT_DOUBLE_EQ(y.val(), std::exp(0.5));
  EXPECT_DOUBLE_EQ(y.grad()(2), std::exp(0.5));
  EXPECT_DOUBLE_EQ(y.hessian()(2, 2), std::exp(0.5));
}

TEST(TDenseVar, sin_cos) {
  TDenseVar<4> x(M_PI / 4, 2);

  TDenseVar<4> y = sin(x) + cos(x);

  EXPECT_DOUBLE_EQ(y.val(), std::sin(M_PI / 4) + std::cos(M_PI / 4));
  EXPECT_DOUBLE_EQ(y.grad()(2), std::cos(M_PI / 4) - std::sin(M_PI / 4));
  EXPECT_DOUBLE_EQ(y.hessian()(2, 2), -std::sin(M_PI / 4) - std::cos(M_PI / 4));
}

TEST(TDenseVar, pow_sqr) {
  TDenseVar<4> x(2.0, 2);

  TDenseVar<4> y = pow(x, 3) + sqr(x);

  EXPECT_DOUBLE_EQ(y.val(), 8.0 + 4.0);
  EXPECT_DOUBLE_EQ(y.grad()(2), 3 * 4.0 + 2 * 2.0);
  EXPECT_DOUBLE_EQ(y.hessian()(2, 2), 3 * 2 * 2.0 + 2);
}

TEST(TDenseVar, rotation) {
  // Setup random axis and angle variables.
  Eigen::Vector3d axis = Eigen::Vector3d::Random().normalized();
  Eigen::Vector3d point = Eigen::Vector3d::Random();
  double angle = M_PI / 3;
  Eigen::VectorXd vars(7);
  vars << axis, point, angle;
  // Autodiff variable factory.
  Eigen::Map<const Eigen::MatrixXd> vars_map(vars.data(), 7, 1);
  Optiz::LocalVarFactory<7> var_factory(vars_map);

  auto func = [&](int i, auto &x) {
    using T = FACTORY_TYPE(x);
    Eigen::AngleAxis<T> angle_axis(
        x(6), Eigen::Vector3<T>(x(0), x(1), x(2)).normalized());
    Eigen::Vector3<T> rotated_point =
        angle_axis * Eigen::Vector3<T>(x(3), x(4), x(5));
    return rotated_point(i);
  };

  for (int i = 0; i < 1; i++) {
    auto autodiff = func(i, var_factory);
    // Value check.
    EXPECT_NEAR(autodiff.val(), func(i, vars), 1e-9);

    // Finite difference gradient.
    Eigen::VectorXd fd_grad = finite_difference_gradient(
        [&](const Eigen::VectorXd &v) { return func(i, v); }, vars, 1e-8);
    for (int j = 0; j < 7; j++) {
      EXPECT_NEAR(autodiff.grad()(j), fd_grad(var_factory.local_to_global[j]),
                  1e-6);
    }
    // Finite difference hessian.
    Eigen::MatrixXd fd_hessian = finite_difference_hessian(
        [&](const Eigen::VectorXd &v) { return func(i, v); }, vars, 1e-6);
    for (int j = 0; j < 7; j++) {
      for (int k = 0; k < 7; k++) {
        EXPECT_NEAR(autodiff.hessian()(j, k),
                    fd_hessian(var_factory.local_to_global[j],
                               var_factory.local_to_global[k]),
                    1e-4);
      }
    }
  }
}