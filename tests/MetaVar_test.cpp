#include <gtest/gtest.h>

#include "Optiz/Autodiff/MetaVar.h"
#include "Optiz/NewtonSolver/ElementFunc.h"
#include "test_utils.h"
#include <Optiz/Meta/MetaMat.h>

using Optiz::MetaVar;

TEST(MetaVar, constructor) {
  Optiz::MetaVar<2> x(5);

  EXPECT_EQ(x.grad().size(), 3);
  EXPECT_EQ(x.grad()(2), 1);
  EXPECT_EQ(x.hessian().norm(), 0);
}

TEST(MetaVar, multiplication) {
  Optiz::MetaVar<2> x(5);
  Optiz::MetaVar<3> y(3);

  auto z = 3 * x * y;

  EXPECT_EQ(z.val(), 45);
  EXPECT_EQ(z.grad().size(), 4);
  EXPECT_EQ(z.grad()(2), 9);
  EXPECT_EQ(z.grad()(3), 15);
  EXPECT_EQ(z.hessian()(3, 2), 3);
}

TEST(MetaVar, division) {
  Optiz::MetaVar<2> x(6);
  Optiz::MetaVar<3> y(3);

  auto z = x / y;

  EXPECT_EQ(z.val(), 2);
  EXPECT_EQ(z.grad().size(), 4);
  EXPECT_DOUBLE_EQ(z.grad()(2), 1.0 / 3.0);
  EXPECT_DOUBLE_EQ(z.grad()(3), -6.0 / 9.0);
  EXPECT_DOUBLE_EQ(z.hessian()(3, 2), -1.0 / 9.0);
  EXPECT_DOUBLE_EQ(z.hessian()(2, 2), 0.0);
  EXPECT_DOUBLE_EQ(z.hessian()(3, 3), 12.0 / 27.0);
}

TEST(MetaVar, addition) {
  Optiz::MetaVar<2> x(6);
  Optiz::MetaVar<3> y(3);

  auto z = x + y + 5.0;

  EXPECT_EQ(z.val(), 14);
  EXPECT_EQ(z.grad().size(), 4);
  EXPECT_DOUBLE_EQ(z.grad()(2), 1.0);
  EXPECT_DOUBLE_EQ(z.grad()(3), 1.0);
  EXPECT_DOUBLE_EQ(z.hessian().norm(), 0);
}

TEST(Metavar, subtraction) {
  Optiz::MetaVar<2> x(6);
  Optiz::MetaVar<3> y(3);

  auto z = x - y - 2.0;

  EXPECT_EQ(z.val(), 1);
  EXPECT_EQ(z.grad().size(), 4);
  EXPECT_DOUBLE_EQ(z.grad()(2), 1.0);
  EXPECT_DOUBLE_EQ(z.grad()(3), -1.0);
  EXPECT_DOUBLE_EQ(z.hessian().norm(), 0);
}

TEST(MetaVar, chain) {
  Optiz::MetaVar<2> x(0.5);

  auto y = exp(x);

  EXPECT_DOUBLE_EQ(y.val(), std::exp(0.5));
  EXPECT_DOUBLE_EQ(y.grad()(2), std::exp(0.5));
  EXPECT_DOUBLE_EQ(y.hessian()(2, 2), std::exp(0.5));
}

TEST(MetaVar, sin_cos) {
  auto x = Optiz::MetaVar<2>(M_PI / 4);

  auto y = sin(x) + cos(x);

  EXPECT_DOUBLE_EQ(y.val(), std::sin(M_PI / 4) + std::cos(M_PI / 4));
  EXPECT_DOUBLE_EQ(y.grad()(2), std::cos(M_PI / 4) - std::sin(M_PI / 4));
  EXPECT_DOUBLE_EQ(y.hessian()(2, 2), -std::sin(M_PI / 4) - std::cos(M_PI / 4));
}

TEST(MetaVar, pow_sqr) {
  Optiz::MetaVar<2> x(2.0);

  auto y = pow(x, 3) + sqr(x);

  EXPECT_DOUBLE_EQ(y.val(), 8.0 + 4.0);
  EXPECT_DOUBLE_EQ(y.grad()(2), 3 * 4.0 + 2 * 2.0);
  EXPECT_DOUBLE_EQ(y.hessian()(2, 2), 3 * 2 * 2.0 + 2);
}

TEST(MetaVar, matrix_mul) {
  Optiz::MetaVec a =
      Optiz::MetaVec(MetaVar<0>(2.0), MetaVar<1>(3.0), MetaVar<2>(4.0));
  Eigen::Matrix<double, 4, 3> rand_mat = Eigen::Matrix<double, 4, 3>::Random();
  Eigen::VectorXd vars(3);
  vars << 2.0, 3.0, 4.0;

  auto res = (rand_mat * a).map([](const auto &a) { return cos(a); });
  auto sqrnorm = res.reduce_aux<1>(
      [](const auto &a, const auto &b) { return Optiz::sqr(a) + b; },
      Optiz::sqr(res.get<0>()));
  auto func = [&](const Eigen::VectorXd &v) {
    return (rand_mat * v).array().cos().square().sum();
  };

  Eigen::VectorXd fd_grad = finite_difference_gradient(func, vars, 1e-8);
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(sqrnorm.grad()(i), fd_grad(i), 1e-6);
  }
  Eigen::MatrixXd fd_hessian = finite_difference_hessian(func, vars, 1e-4);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_NEAR(sqrnorm.hessian()(i, j), fd_hessian(i, j), 1e-6);
    }
  }
}