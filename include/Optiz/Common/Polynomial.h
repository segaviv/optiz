#pragma once

#include <Eigen/Eigen>
#include <complex>
#include <iostream>
#include <type_traits>

#define POLY(poly)                                                             \
  [] {                                                                         \
    Optiz::Polynomial x(Eigen::RowVector<double, 2>{0, 1}),                    \
        x2(Eigen::RowVector<double, 3>{0, 0, 1}),                              \
        x3(Eigen::RowVector<double, 4>{0, 0, 0, 1}),                           \
        x4(Eigen::RowVector<double, 5>{0, 0, 0, 0, 1}),                        \
        x5(Eigen::RowVector<double, 6>{0, 0, 0, 0, 0, 1});                     \
    return poly;                                                               \
  }()

namespace Optiz {

struct Polynomial {

  Polynomial() {}

  explicit Polynomial(const Eigen::MatrixXd &coefs) : coefs(coefs) {}
  explicit Polynomial(double x) { coefs = Eigen::Matrix<double, 1, 1>::Constant(x); }

  Polynomial pow(int n);

  Eigen::VectorXd at(double t);
  Eigen::VectorXcd at(const std::complex<double> &t);

  Eigen::VectorXcd roots(double threshold = 1e-6);
  Eigen::VectorXd real_roots();

  Polynomial dx();

  // Each col is a point coefficient for some power.
  // The first col is x^0.
  Eigen::MatrixXd coefs = Eigen::MatrixXd::Zero(1, 1);
};
std::ostream &operator<<(std::ostream &s, const Polynomial &p);
Polynomial operator*(const Polynomial &p1, const Polynomial &p2);
Polynomial operator*(const Polynomial &p1, const Eigen::VectorXd &p2);
Polynomial operator+(const Polynomial &p1, const Polynomial &p2);
Polynomial operator-(const Polynomial &p1, const Polynomial &p2);
Polynomial pow(const Polynomial &p, int n);
} // namespace Optiz
