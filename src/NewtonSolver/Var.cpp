#include "Var.h"
#include <iomanip>

#include "ProjectHessian.h"

namespace Optiz {

Var::Var(Var &&other) noexcept = default;
Var::Var(const Var &other) = default;
Var::Var() : Var(0.0) {}
Var::Var(double val) : _val(val) {}
Var::Var(const double val, int index) : Var(val) { _grad.insert(index) = 1.0; }

Var &Var::operator=(Var &&) = default;
Var &Var::operator=(double val) {
  _val = val;
  return *this;
}

double Var::val() const { return _val; }
const SparseVector &Var::grad() const { return _grad; }
Eigen::VectorXd Var::dense_grad() const { return _grad.to_dense(); }
const SelfAdjointMapMatrix &Var::hessian() const { return _hessian; }
int Var::n_vars() const { return _grad.rows(); }

Var::operator Var::Tup() const { return {_val, _grad.to_dense(), _hessian}; }

std::ostream &operator<<(std::ostream &s, const Var &var) {
  s << "Val: " << var._val << std::endl
    << "Grad: " << std::endl
    << var._grad.to_dense() << std::endl
    << "Hessian: " << std::endl
    << var._hessian.to_dense() << std::endl;
  return s;
}

std::string Var::referenced_str() const {
  std::map<int, int> old_to_new;
  std::vector<int> new_to_old;
  int new_index = 0;
  for (auto val : _grad.get_values()) {
    old_to_new[val.first] = new_index;
    new_to_old.push_back(val.first);
    new_index++;
  }

  std::stringstream ss;
  ss << "Val: " << _val << std::endl;
  ss << std::endl << "Grad: " << std::endl;
  for (auto val : _grad.get_values()) {
    ss << val.first << ": " << val.second << std::endl;
  }
  Eigen::MatrixXd hess =
      Eigen::MatrixXd::Zero(old_to_new.size(), old_to_new.size());
  for (auto val : _hessian.get_values()) {
    hess(old_to_new[val.first.row], old_to_new[val.first.col]) = val.second;
    hess(old_to_new[val.first.col], old_to_new[val.first.row]) = val.second;
  }
  ss << std::endl << "Hessian: " << std::endl;
  for (int i = 0; i < hess.rows(); i++) {
    ss << new_to_old[i] << ": ";
    for (int j = 0; j < hess.cols(); j++) {
      ss << std::setw(10) << hess(i, j) << " ";
    }
    ss << std::endl;
  }
  return ss.str();
}

Var &Var::operator*=(const Var &b) {
  if (&b == this) {
    _hessian *= 2 * _val;
    _hessian.rank_update(_grad, b._grad);
    _grad *= 2 * _val;
  } else {
    _hessian *= b._val;
    _hessian.add(b._hessian, _val);
    _hessian.rank_update(_grad, b._grad);
    _grad *= b._val;
    _grad.add(b._grad, _val);
  }
  _val *= b._val;
  return *this;
}
Var &Var::operator*=(double b) {
  _val *= b;
  _grad *= b;
  _hessian *= b;
  return *this;
}
Var &Var::operator/=(const Var &b) { return operator*=(b.inv()); }
Var &Var::operator/=(double b) {
  _val /= b;
  _grad /= b;
  _hessian /= b;
  return *this;
}
Var &Var::operator+=(const Var &b) {
  _val += b._val;
  _grad += b._grad;
  _hessian += b._hessian;
  return *this;
}
Var &Var::operator+=(double b) {
  _val += b;
  return *this;
}
Var &Var::operator-=(const Var &b) {
  _val -= b._val;
  _grad -= b._grad;
  _hessian -= b._hessian;
  return *this;
}
Var &Var::operator-=(double b) {
  _val -= b;
  return *this;
}
Var &Var::chain_this(double val, double grad, double hessian) {
  _val = val;
  _hessian *= grad;
  _hessian.rank_update(_grad, hessian);
  _grad *= grad;
  return *this;
}

Var Var::chain(double val, double grad, double hessian) const {
  Var res(*this);
  res.chain_this(val, grad, hessian);
  return res;
}

Var Var::inv() const {
  double valsqr = _val * _val;
  double valcube = valsqr * _val;
  return chain(1 / _val, -1 / valsqr, 2 / valcube);
}

Var &Var::inv_self() {
  double valsqr = _val * _val;
  double valcube = valsqr * _val;
  chain_this(1 / _val, -1 / valsqr, 2 / valcube);
  return *this;
}

Var &Var::neg() {
  chain_this(-_val, -1.0, 0.0);
  return *this;
}

Var &Var::projectHessian() {
  _hessian = project_hessian(_hessian);
  return *this;
}

void Var::reserve(int n) {
  // _grad.reserve(n);
  // _hessian.reserve(n*n);
}

Var operator*(const Var &a, const Var &b) {
  Var res(a);
  res *= b;
  return res;
}
Var operator*(Var &&a, const Var &b) {
  a *= b;
  return a;
}
Var operator*(const Var &a, Var &&b) { return std::move(b) * a; }
Var operator*(Var &&a, Var &&b) { return std::move(a) * b; }

Var operator*(double b, const Var &a) {
  Var res = a;
  res *= b;
  return res;
}
Var operator*(const Var &a, double b) { return b * a; }
Var operator*(double b, Var &&a) {
  a *= b;
  return a;
}
Var operator*(Var &&a, double b) {
  a *= b;
  return a;
}

Var operator/(const Var &a, const Var &b) {
  Var res(a);
  res /= b;
  return res;
}
Var operator/(Var &&a, const Var &b) {
  a /= b;
  return a;
}
Var operator/(const Var &a, Var &&b) { return a / b; }
Var operator/(Var &&a, Var &&b) { return std::move(a) / b; }
Var operator/(double b, const Var &a) {
  Var res = a.inv();
  res *= b;
  return res;
}
Var operator/(const Var &a, double b) {
  Var res(a);
  res /= b;
  return res;
}
Var operator/(double b, Var &&a) {
  a.inv_self() *= b;
  return a;
}
Var operator/(Var &&a, double b) {
  a /= b;
  return a;
}
/* ------------------------ Operator+ ------------------------ */
Var operator+(const Var &a, const Var &b) {
  Var res(a);
  res += b;
  return res;
}
Var operator+(Var &&a, const Var &b) {
  a += b;
  return a;
}

Var operator+(const Var &a, Var &&b) { return std::move(b) + a; }
Var operator+(Var &&a, Var &&b) { return std::move(a) + b; }

Var operator+(const Var &a, double b) {
  Var res(a);
  res += b;
  return res;
}
Var operator+(double b, const Var &a) { return a + b; }
Var operator+(double b, Var &&a) {
  a += b;
  return a;
}

Var operator+(Var &&a, double b) {
  a += b;
  return a;
}

/* ------------------------ Operator- ------------------------ */
Var operator-(const Var &a, const Var &b) {
  Var res(a);
  res -= b;
  return res;
}
Var operator-(Var &&a, const Var &b) {
  a -= b;
  return a;
}
Var operator-(const Var &a, Var &&b) {
  b.neg() += a;
  return b;
}
Var operator-(Var &&a, Var &&b) { return std::move(a) - b; }

Var operator-(double b, const Var &a) {
  Var res(-a);
  res += b;
  return res;
}
Var operator-(const Var &a, double b) {
  Var res(a);
  res -= b;
  return res;
}
Var operator-(double b, Var &&a) {
  a.neg() += b;
  return a;
}
Var operator-(Var &&a, double b) {
  a -= b;
  return a;
}
Var operator-(const Var &a) { return a.chain(-a._val, -1.0, 0.0); }
Var operator-(Var &&a) {
  a.neg();
  return a;
}
Var operator+(const Var &a) {
  Var res(a);
  res.projectHessian();
  return res;
}
Var operator+(Var &&a) { return a.projectHessian(); }
Var sqrt(const Var &a) {
  const auto &sqrt_a = std::sqrt(a._val);
  return a.chain(sqrt_a, 0.5 / sqrt_a, -0.25 / (sqrt_a * a._val));
}
Var sqrt(Var &&a) {
  const auto &sqrt_a = std::sqrt(a._val);
  a.chain_this(sqrt_a, 0.5 / sqrt_a, -0.25 / (sqrt_a * a._val));
  return a;
}
Var sqr(const Var &a) {
  // return a * a;
  return a.chain(a._val * a._val, 2 * a._val, 2);
}
Var sqr(Var &&a) {
  a.chain_this(a._val * a._val, 2 * a._val, 2);
  return a;
}
Var abs(const Var &a) { return a.chain(a._val, a._val >= 0 ? 1 : -1, 0); }
Var abs(Var &&a) {
  a.chain_this(a._val, a._val >= 0 ? 1 : -1, 0);
  return a;
}
Var pow(const Var &a, const int exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();

  return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
}
Var pow(const Var &a, const double exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();

  return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
}
Var pow(Var &&a, const int exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();
  a.chain_this(f, exponent * f1, exponent * (exponent - 1) * f2);
  return a;
}
Var pow(Var &&a, const double exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();
  a.chain_this(f, exponent * f1, exponent * (exponent - 1) * f2);
  return a;
}

Var exp(const Var &a) {
  const double val = std::exp(a._val);
  return a.chain(val, val, val);
}
Var exp(Var &&a) {
  const double val = std::exp(a._val);
  a.chain_this(val, val, val);
  return a;
}
Var log(const Var &a) {
  return a.chain(std::log(a._val), 1 / a._val, -1 / (a._val * a._val));
}
Var log(Var &&a) {
  a.chain_this(std::log(a._val), 1 / a._val, -1 / (a._val * a._val));
  return a;
}
Var cos(const Var &a) {
  double cosval = std::cos(a._val);
  return a.chain(cosval, -std::sin(a._val), -cosval);
}
Var cos(Var &&a) {
  double cosval = std::cos(a._val);
  a.chain_this(cosval, -std::sin(a._val), -cosval);
  return a;
}
Var sin(const Var &a) {
  double sinval = std::sin(a._val);
    return a.chain(sinval, std::cos(a._val), -sinval);
}
Var sin(Var &&a) {
  double sinval = std::sin(a._val);
  a.chain_this(sinval, std::cos(a._val), -sinval);
  return a;
}
Var atan(const Var &x) {
  double denom = (x._val * x._val + 1);
  return x.chain(std::atan(x._val), 1 / denom, -2 * x._val / (denom * denom));
}
Var atan2(const Var &y, const Var &x) {
  // return 2 * atan(y / (sqrt(sqr(x) + sqr(y)) + x));
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
Var chain2(const Var &x, const Var &y, double val, double dx, double dy,
           double dxdx, double dxdy, double dydy) {
  Var res(val);
  res._grad.add(x._grad, dx);
  res._grad.add(y._grad, dy);

  res._hessian.add(x._hessian, dx);
  res._hessian.add(y._hessian, dy);
  res._hessian.rank_update(x._grad, dxdx);
  res._hessian.rank_update(x._grad, y._grad, dxdy);
  res._hessian.rank_update(y._grad, dydy);
  return res;
}
bool isfinite(const Var &x) { return std::isfinite(x._val); }
bool isinf(const Var &x) { return std::isinf(x._val); }

// ----------------------- Comparisons -----------------------
bool operator<(const Var &a, const Var &b) { return a._val < b._val; }
bool operator<=(const Var &a, const Var &b) { return a._val <= b._val; }
bool operator>(const Var &a, const Var &b) { return a._val > b._val; }
bool operator>=(const Var &a, const Var &b) { return a._val >= b._val; }
bool operator==(const Var &a, const Var &b) { return a._val == b._val; }
bool operator!=(const Var &a, const Var &b) { return a._val != b._val; }

} // namespace Optiz