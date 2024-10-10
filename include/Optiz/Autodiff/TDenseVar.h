#pragma once

#include <Eigen/Eigen>
#include <iostream>
#include <tuple>

#include "ProjectHessian.h"

namespace Optiz {

template <int k> class TDenseVar {
public:
  using KVEC = Eigen::Matrix<double, k, 1>;
  using KMAT = Eigen::Matrix<double, k, k>;

  TDenseVar() = default;
  TDenseVar(double val, const KVEC &vec, const KMAT &mat)
      : _val(val), _grad(vec), _hessian(mat) {}

  TDenseVar(double val) : _val(val) {}
  TDenseVar(double val, int index) : _val(val) { _grad(index) = 1.0; }

  // Getters.
  inline double val() const { return _val; }
  inline double &val() { return _val; }
  inline const KVEC &grad() const { return _grad; }
  inline KVEC &grad() { return _grad; }
  inline const KMAT &hessian() const { return _hessian; }
  inline KMAT &hessian() { return _hessian; }
  using Tup = std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd>;
  inline operator Tup() const { return {_val, _grad, _hessian}; }

  TDenseVar &projectHessian() {
    project_hessian(_hessian);
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &s, const TDenseVar &var) {
    s << "Val: " << var._val << std::endl
      << "Grad: " << std::endl
      << var._grad << std::endl
      << "Hessian: " << std::endl
      << var._hessian << std::endl;
    return s;
  }

  TDenseVar &operator*=(const TDenseVar &b) {
    _hessian *= b._val;
    _hessian += _grad * b._grad.transpose() + b._grad * _grad.transpose() +
                _val * b._hessian;

    _grad *= b._val;
    _grad += _val * b._grad;
    _val *= b._val;
    return *this;
  }
  TDenseVar &operator*=(double b) {
    _val *= b;
    _grad *= b;
    _hessian *= b;
    return *this;
  }
  TDenseVar &operator/=(const TDenseVar &b) {
    _val /= b._val;
    _grad /= b._val;
    _grad -= b._grad * (_val / b._val);
    _hessian -= _grad * b._grad.transpose() + b._grad * _grad.transpose() +
                _val * b._hessian;
    _hessian /= b._val;
    return *this;
  }
  TDenseVar &operator/=(double b) {
    _val /= b;
    _grad /= b;
    _hessian /= b;
    return *this;
  }
  TDenseVar &operator+=(const TDenseVar &b) {
    _val += b._val;
    _grad += b._grad;
    _hessian += b._hessian;
    return *this;
  }
  TDenseVar &operator+=(double b) {
    _val += b;
    return *this;
  }
  TDenseVar &operator-=(const TDenseVar &b) {
    _val -= b._val;
    _grad -= b._grad;
    _hessian -= b._hessian;
    return *this;
  }
  TDenseVar &operator-=(double b) {
    _val -= b;
    return *this;
  }
  TDenseVar &chain_self(double val, double grad, double hessian) {
    _val = val;
    _hessian *= grad;
    _hessian += hessian * _grad * _grad.transpose();
    _grad *= grad;
    return *this;
  }
  TDenseVar chain(double val, double grad, double hessian) const {
    return TDenseVar(val, _grad * grad,
                     _hessian * grad + hessian * _grad * _grad.transpose());
  }

  TDenseVar inv() const {
    double valsqr = _val * _val;
    double valcube = valsqr * _val;
    return chain(1 / _val, -1 / valsqr, 2 / valcube);
  }
  TDenseVar &inv_self() {
    double valsqr = _val * _val;
    double valcube = valsqr * _val;
    chain_self(1 / _val, -1 / valsqr, 2 / valcube);
    return *this;
  }
  TDenseVar &neg() {
    chain_self(-_val, -1.0, 0.0);
    return *this;
  }

  // Mul operator between two TDenseVars.
  friend TDenseVar operator*(const TDenseVar &a, const TDenseVar &b) {
    // return TDenseVar(a._val * b._val, a._grad * b._val + a._val * b._grad,
    //                  a._grad * b._grad.transpose() +
    //                      b._grad * a._grad.transpose() + a._hessian * b._val
    //                      + a._val * b._hessian);
    return TDenseVar(a) *= b;
  }
  friend TDenseVar operator*(double b, const TDenseVar &a) {

    return TDenseVar(a) *= b;
  }
  friend TDenseVar operator*(const TDenseVar &a, double b) { return b * a; }

  // Div operator between two TDenseVars.
  friend TDenseVar operator/(const TDenseVar &a, const TDenseVar &b) {
    return TDenseVar(a) /= b;
  }
  friend TDenseVar operator/(double b, const TDenseVar &a) {
    return a.inv() * b;
  }
  friend TDenseVar operator/(const TDenseVar &a, double b) {
    TDenseVar res(a);
    res /= b;
    return res;
  }

  // Add operator between two TDenseVars.
  friend TDenseVar operator+(const TDenseVar &a, const TDenseVar &b) {
    // return TDenseVar(a._val + b._val, a._grad + b._grad,
    //                  a._hessian + b._hessian);
    return TDenseVar(a) += b;
  }
  // Add operator between TDenseVar and double
  friend TDenseVar operator+(double b, const TDenseVar &a) { return a + b; }

  friend TDenseVar operator+(const TDenseVar &a, double b) {
    return TDenseVar(a) += b;
  }

  // Sub operator between two TDenseVars.
  friend TDenseVar operator-(const TDenseVar &a, const TDenseVar &b) {
    return TDenseVar(a) -= b;
  }
  // Sub operator between TDenseVar and double
  friend TDenseVar operator-(double b, const TDenseVar &a) {
    TDenseVar res(-a);
    res += b;
    return res;
  }
  friend TDenseVar operator-(const TDenseVar &a, double b) {
    TDenseVar res(a);
    res -= b;
    return res;
  }

  friend TDenseVar operator-(const TDenseVar &a) {
    return a.chain(-a._val, -1.0, 0.0);
  }
  friend TDenseVar operator+(const TDenseVar &a) {
    TDenseVar res(a);
    res.projectHessian();
    return res;
  }
  friend TDenseVar sqrt(const TDenseVar &a) {
    const auto &sqrt_a = std::sqrt(a._val);
    return a.chain(sqrt_a, 0.5 / sqrt_a, -0.25 / (sqrt_a * a._val));
  }
  friend TDenseVar abs(const TDenseVar &a) {
    return a.chain(a._val, a._val >= 0 ? 1 : -1, 0);
  }
  friend TDenseVar pow(const TDenseVar &a, double exponent) {
    double f2 = std::pow(a.val(), exponent - 2);
    double f1 = f2 * a.val();
    double f = f1 * a.val();

    return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
  }
  friend TDenseVar pow(const TDenseVar &a, const int exponent) {
    double f2 = std::pow(a.val(), exponent - 2);
    double f1 = f2 * a.val();
    double f = f1 * a.val();

    return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
  }
  friend TDenseVar exp(const TDenseVar &a) {
    double val = std::exp(a._val);
    return a.chain(val, val, val);
  }
  friend TDenseVar log(const TDenseVar &a) {
    return a.chain(std::log(a._val), 1 / a._val, -1 / (a._val * a._val));
  }
  friend TDenseVar sin(const TDenseVar &a) {
    double sinval = std::sin(a._val);
    return a.chain(sinval, std::cos(a._val), -sinval);
  }
  friend TDenseVar cos(const TDenseVar &a) {
    double cosval = std::cos(a._val);
    return a.chain(cosval, -std::sin(a._val), -cosval);
  }
  friend TDenseVar atan2(const TDenseVar &y, const TDenseVar &x) {
    double val = std::atan2(y._val, x._val);
    double x_sqr = x._val * x._val, y_sqr = y._val * y._val;
    double denom = x_sqr + y_sqr;
    double denom_sqr = denom * denom;
    // First order derivatives.
    double dx = -y._val / denom;
    double dy = x._val / denom;
    // Second order derivatives.
    double dxdx = 2 * x._val * y._val / denom_sqr;
    double dydy = -dxdx;
    double dxdy = (y_sqr - x_sqr) / denom_sqr;
    return chain2(x, y, val, dx, dy, dxdx, dxdy, dydy);
  }

  // Chain rule for multivariate function (with 2 variables, R^2 -> R).
  friend TDenseVar chain2(const TDenseVar &a, const TDenseVar &b, double val,
                          double da, double db, double dada, double dadb,
                          double dbdb) {
    TDenseVar res(val);
    res._grad = da * a._grad + db * b._grad;
    res._hessian =
        da * a._hessian + db * b._hessian +
        dada * a._grad * a._grad.transpose() +
        dadb * (a._grad * b._grad.transpose() + b._grad * a._grad.transpose()) +
        dbdb * b._grad * b._grad.transpose();
    return res;
  }

  // Chain rule for multivariate function (R^n -> R).
  // chain(vars, f(v1, v2, ..., vn), df/dvi, d2f/dvidvj).
  friend TDenseVar chain(const Eigen::VectorX<TDenseVar> &vars, double val,
                         const Eigen::VectorXd &grad,
                         const Eigen::MatrixXd &hessian) {
    TDenseVar res(val);
    for (int i = 0; i < vars.size(); ++i) {
      res._grad += grad(i) * vars[i]._grad;
      res._hessian += grad(i) * vars[i]._hessian;
      for (int j = 0; j < vars.size(); ++j) {
        res._hessian +=
            hessian(i, j) * vars[i]._grad * vars[j]._grad.transpose();
      }
    }
    return res;
  }

  // ----------------------- Comparisons -----------------------
  friend bool operator<(const TDenseVar &a, const TDenseVar &b) {
    return a._val < b._val;
  }
  friend bool operator<=(const TDenseVar &a, const TDenseVar &b) {
    return a._val <= b._val;
  }
  friend bool operator>(const TDenseVar &a, const TDenseVar &b) {
    return a._val > b._val;
  }
  friend bool operator>=(const TDenseVar &a, const TDenseVar &b) {
    return a._val >= b._val;
  }
  friend bool operator==(const TDenseVar &a, const TDenseVar &b) {
    return a._val == b._val;
  }
  friend bool operator!=(const TDenseVar &a, const TDenseVar &b) {
    return a._val != b._val;
  }

private:
  double _val = 0.0;
  KVEC _grad = KVEC::Zero();
  KMAT _hessian = KMAT::Zero();
};

template<int k>
TDenseVar<k> sqr(const TDenseVar<k> &a) {
    return a.chain(a.val() * a.val(), 2 * a.val(), 2);
  }

} // namespace Optiz

namespace Eigen {

/**
 * See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
 * and https://eigen.tuxfamily.org/dox/structEigen_1_1NumTraits.html
 */
template <int k> struct NumTraits<Optiz::TDenseVar<k>> : NumTraits<double> {
  typedef Optiz::TDenseVar<k> Real;
  typedef Optiz::TDenseVar<k> NonInteger;
  typedef Optiz::TDenseVar<k> Nested;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = k * k,
    AddCost = k * k,
    MulCost = k * k * k,
  };
};

template <typename BinaryOp, int k>
struct ScalarBinaryOpTraits<Optiz::TDenseVar<k>, double, BinaryOp> {
  typedef Optiz::TDenseVar<k> ReturnType;
};

template <typename BinaryOp, int k>
struct ScalarBinaryOpTraits<double, Optiz::TDenseVar<k>, BinaryOp> {
  typedef Optiz::TDenseVar<k> ReturnType;
};

} // namespace Eigen