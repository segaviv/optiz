#pragma once
#include <Eigen/Eigen>
#include <iostream>
#include <tuple>

#include "ProjectHessian.h"
#include "SelfAdjointMapMatrix.h"

namespace Optiz {

class Var {
public:
  Var();
  explicit Var(double val);
  Var(const double val, int index);
  Var(const Var &);
  Var(Var &&) noexcept;

  Var &operator=(const Var &other) = default;
  Var &operator=(Var &&);
  Var &operator=(double val);

  // Getters.
  double val() const;
  inline double &val() { return _val; }
  inline SparseVector &grad() { return _grad; }
  inline SelfAdjointMapMatrix &hessian() { return _hessian; };
  const SparseVector &grad() const;
  Eigen::VectorXd dense_grad() const;
  const SelfAdjointMapMatrix &hessian() const;
  using Tup =
      std::tuple<double, Eigen::VectorXd, std::vector<Eigen::Triplet<double>>>;
  operator Tup() const;

  int n_vars() const;

  Var &projectHessian();
  void reserve(int n);

  friend std::ostream &operator<<(std::ostream &s, const Var &var);
  std::string referenced_str() const;

  // Operators.
  Var &operator*=(const Var &b);
  Var &operator*=(double b);
  Var &operator/=(const Var &b);
  Var &operator/=(double b);
  Var &operator+=(const Var &b);
  Var &operator+=(double b);
  Var &operator-=(const Var &b);
  Var &operator-=(double b);
  Var &chain_this(double val, double grad, double hessian);
  Var chain(double val, double grad, double hessian) const;

  Var inv() const;
  Var &inv_self();
  Var &neg();

  // Mul operator between two variables.
  friend Var operator*(const Var &a, const Var &b);
  friend Var operator*(Var &&a, const Var &b);
  friend Var operator*(const Var &a, Var &&b);
  friend Var operator*(Var &&a, Var &&b);
  friend Var operator*(double b, const Var &a);
  friend Var operator*(const Var &a, double b);
  friend Var operator*(double b, Var &&a);
  friend Var operator*(Var &&a, double b);

  // Div operator between two variables.
  friend Var operator/(const Var &a, const Var &b);
  friend Var operator/(Var &&a, const Var &b);
  friend Var operator/(const Var &a, Var &&b);
  friend Var operator/(Var &&a, Var &&b);
  friend Var operator/(double b, const Var &a);
  friend Var operator/(const Var &a, double b);
  friend Var operator/(double b, Var &&a);
  friend Var operator/(Var &&a, double b);

  // Add operator between two variables.
  friend Var operator+(const Var &a, const Var &b);
  friend Var operator+(Var &&a, const Var &b);
  friend Var operator+(const Var &a, Var &&b);
  friend Var operator+(Var &&a, Var &&b);
  // Add operator between Var and double
  friend Var operator+(double b, const Var &a);
  friend Var operator+(double b, Var &&a);
  friend Var operator+(const Var &a, double b);
  friend Var operator+(Var &&a, double b);

  // Sub operator between two variables.
  friend Var operator-(const Var &a, const Var &b);
  friend Var operator-(Var &&a, const Var &b);
  friend Var operator-(const Var &a, Var &&b);
  friend Var operator-(Var &&a, Var &&b);
  // Sub operator between Var and double
  friend Var operator-(double b, const Var &a);
  friend Var operator-(const Var &a, double b);
  friend Var operator-(double b, Var &&a);
  friend Var operator-(Var &&a, double b);

  Var operator^(const Var &p) { return pow(*this, p.val()); }
  friend Var operator-(const Var &a);
  friend Var operator-(Var &&a);
  friend Var operator+(const Var &a);
  friend Var operator+(Var &&a);
  friend Var sqrt(const Var &a);
  friend Var sqrt(Var &&a);
  friend Var abs(const Var &a);
  friend Var abs(Var &&a);
  friend Var pow(const Var &a, const double exponent);
  friend Var pow(const Var &a, const int exponent);
  friend Var pow(Var &&a, const int exponent);
  friend Var pow(Var &&a, const double exponent);
  friend Var exp(const Var &a);
  friend Var exp(Var &&a);
  friend Var log(const Var &a);
  friend Var log(Var &&a);
  friend Var cos(const Var &a);
  friend Var cos(Var &&a);
  friend Var sin(const Var &a);
  friend Var sin(Var &&a);
  friend Var atan(const Var &x);
  friend Var atan2(const Var &y, const Var &x);
  friend Var chain2(const Var &x, const Var &y, double val, double dx,
                    double dy, double dxdx, double dxdy, double dydy);
  friend bool isfinite(const Var &x);
  friend bool isinf(const Var &x);

  // ----------------------- Comparisons -----------------------
  friend bool operator<(const Var &a, const Var &b);
  friend bool operator<=(const Var &a, const Var &b);
  friend bool operator>(const Var &a, const Var &b);
  friend bool operator>=(const Var &a, const Var &b);
  friend bool operator==(const Var &a, const Var &b);
  friend bool operator!=(const Var &a, const Var &b);

public:
  double _val;
  SparseVector _grad;
  SelfAdjointMapMatrix _hessian;
};

Var sqr(const Var &a);
Var sqr(Var &&a);

} // namespace Optiz

#pragma omp declare reduction(+ : Optiz::Var : omp_out += omp_in)

namespace Eigen {

/**
 * See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
 * and https://eigen.tuxfamily.org/dox/structEigen_1_1NumTraits.html
 */
template <> struct NumTraits<Optiz::Var> : NumTraits<double> {
  typedef Optiz::Var Real;
  typedef Optiz::Var NonInteger;
  typedef Optiz::Var Nested;

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
struct ScalarBinaryOpTraits<Optiz::Var, double, BinaryOp> {
  typedef Optiz::Var ReturnType;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, Optiz::Var, BinaryOp> {
  typedef Optiz::Var ReturnType;
};

} // namespace Eigen