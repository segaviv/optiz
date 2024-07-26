#include <gtest/gtest.h>

#include "Optiz/NewtonSolver/Var.h"

TEST(Var, constructor) {
  Optiz::Var::set_num_vars(4);
  Optiz::Var x(5, 2);

  EXPECT_EQ(x.grad().size(), 4);
  EXPECT_EQ(x.grad().get_values().size(), 1);
  EXPECT_EQ(x.grad()(2), 1);
  EXPECT_EQ(x.hessian().get_values().size(), 0);
}

TEST(Var, multiplication) {
  Optiz::Var::set_num_vars(4);
  Optiz::Var x(5, 2);
  Optiz::Var y(3, 3);

  Optiz::Var z = 3 * x * y;

  EXPECT_EQ(z.val(), 45);
  EXPECT_EQ(z.grad().size(), 4);
  EXPECT_EQ(z.grad().get_values().size(), 2);
  EXPECT_EQ(z.grad()(2), 9);
  EXPECT_EQ(z.grad()(3), 15);
  EXPECT_EQ(z.hessian().get_values().size(), 1);
  EXPECT_EQ(z.hessian()(3, 2), 3);
}