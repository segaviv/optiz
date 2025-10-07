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
    return Optiz::Polynomial(poly);                                            \
  }()

namespace Optiz {

struct Polynomial {

  Polynomial() {}

  explicit Polynomial(const Eigen::MatrixXd &coefs) : coefs(coefs) {}
  explicit Polynomial(double x) {
    coefs = Eigen::Matrix<double, 1, 1>::Constant(x);
  }

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
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
inline Polynomial operator*(const Polynomial &p1, const T &scalar) {
  return p1 * Polynomial(scalar);
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
inline Polynomial operator*(const T &scalar, const Polynomial &p1) {
  return Polynomial(scalar) * p1;
}
Polynomial operator*(const Polynomial &p1, const Eigen::VectorXd &p2);
Polynomial operator+(const Polynomial &p1, const Polynomial &p2);
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
inline Polynomial operator+(const Polynomial &p1, const T &scalar) {
  return p1 + Polynomial(scalar);
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
inline Polynomial operator+(const T &scalar, const Polynomial &p1) {
  return Polynomial(scalar) + p1;
}
Polynomial operator-(const Polynomial &p1, const Polynomial &p2);
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
inline Polynomial operator-(const Polynomial &p1, const T &scalar) {
  return p1 - Polynomial(scalar);
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
inline Polynomial operator-(const T &scalar, const Polynomial &p1) {
  return Polynomial(scalar) - p1;
}

inline Polynomial operator-(const Polynomial &p1) {
  return Polynomial(-p1.coefs);
}
Polynomial pow(const Polynomial &p, int n);
} // namespace Optiz

namespace Eigen {

/**
 * See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
 * and https://eigen.tuxfamily.org/dox/structEigen_1_1NumTraits.html
 */
template <> struct NumTraits<Optiz::Polynomial> : NumTraits<double> {
  typedef Optiz::Polynomial Real;
  typedef Optiz::Polynomial NonInteger;
  typedef Optiz::Polynomial Nested;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 9,
    MulCost = 9,
  };
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<Optiz::Polynomial, double, BinaryOp> {
  typedef Optiz::Polynomial ReturnType;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, Optiz::Polynomial, BinaryOp> {
  typedef Optiz::Polynomial ReturnType;
};

} // namespace Eigen