#include "VarGrad.h"
#include <iomanip>

namespace Optiz {

VarGrad::VarGrad(VarGrad &&other) noexcept = default;
VarGrad::VarGrad(const VarGrad &other) = default;
VarGrad::VarGrad() : VarGrad(0.0) {}
VarGrad::VarGrad(double val) : _val(val) {}
VarGrad::VarGrad(const double val, int index) : VarGrad(val) {
  _grad.insert(index) = 1.0;
}

VarGrad &VarGrad::operator=(VarGrad &&) = default;
VarGrad &VarGrad::operator=(double val) {
  _val = val;
  return *this;
}

double VarGrad::val() const { return _val; }
const SparseVector &VarGrad::grad() const { return _grad; }
Eigen::VectorXd VarGrad::dense_grad() const { return _grad.to_dense(); }
int VarGrad::n_vargrads() const { return _grad.rows(); }

std::ostream &operator<<(std::ostream &s, const VarGrad &vargrad) {
  s << "Val: " << vargrad._val << std::endl
    << "Grad: " << std::endl
    << vargrad._grad.to_dense() << std::endl;
  return s;
}

std::string VarGrad::referenced_str() const {
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
  return ss.str();
}

VarGrad &VarGrad::operator*=(const VarGrad &b) {
  if (&b == this) {
    _grad *= 2 * _val;
  } else {
    _grad *= b._val;
    _grad.add(b._grad, _val);
  }
  _val *= b._val;
  return *this;
}
VarGrad &VarGrad::operator*=(double b) {
  _val *= b;
  _grad *= b;
  return *this;
}
VarGrad &VarGrad::operator/=(const VarGrad &b) { return operator*=(b.inv()); }
VarGrad &VarGrad::operator/=(double b) {
  _val /= b;
  _grad /= b;
  return *this;
}
VarGrad &VarGrad::operator+=(const VarGrad &b) {
  _val += b._val;
  _grad += b._grad;
  return *this;
}
VarGrad &VarGrad::operator+=(double b) {
  _val += b;
  return *this;
}
VarGrad &VarGrad::operator-=(const VarGrad &b) {
  _val -= b._val;
  _grad -= b._grad;
  return *this;
}
VarGrad &VarGrad::operator-=(double b) {
  _val -= b;
  return *this;
}
VarGrad &VarGrad::chain_this(double val, double grad) {
  _val = val;
  _grad *= grad;
  return *this;
}

VarGrad VarGrad::chain(double val, double grad) const {
  VarGrad res(val, _grad);
  res._grad *= grad;
  return res;
}

VarGrad VarGrad::inv() const {
  double valsqr = _val * _val;
  return chain(1 / _val, -1 / valsqr);
}

VarGrad &VarGrad::inv_self() {
  double valsqr = _val * _val;
  chain_this(1 / _val, -1 / valsqr);
  return *this;
}

VarGrad &VarGrad::neg() {
  chain_this(-_val, -1.0);
  return *this;
}

VarGrad operator*(const VarGrad &a, const VarGrad &b) {
  VarGrad res(a);
  res *= b;
  return res;
}
VarGrad operator*(VarGrad &&a, const VarGrad &b) {
  a *= b;
  return a;
}
VarGrad operator*(const VarGrad &a, VarGrad &&b) { return std::move(b) * a; }
VarGrad operator*(VarGrad &&a, VarGrad &&b) { return std::move(a) * b; }

VarGrad operator*(double b, const VarGrad &a) {
  VarGrad res = a;
  res *= b;
  return res;
}
VarGrad operator*(const VarGrad &a, double b) { return b * a; }
VarGrad operator*(double b, VarGrad &&a) {
  a *= b;
  return a;
}
VarGrad operator*(VarGrad &&a, double b) {
  a *= b;
  return a;
}

VarGrad operator/(const VarGrad &a, const VarGrad &b) {
  VarGrad res(a);
  res /= b;
  return res;
}
VarGrad operator/(VarGrad &&a, const VarGrad &b) {
  a /= b;
  return a;
}
VarGrad operator/(const VarGrad &a, VarGrad &&b) { return a / b; }
VarGrad operator/(VarGrad &&a, VarGrad &&b) { return std::move(a) / b; }
VarGrad operator/(double b, const VarGrad &a) {
  VarGrad res = a.inv();
  res *= b;
  return res;
}
VarGrad operator/(const VarGrad &a, double b) {
  VarGrad res(a);
  res /= b;
  return res;
}
VarGrad operator/(double b, VarGrad &&a) {
  a.inv_self() *= b;
  return a;
}
VarGrad operator/(VarGrad &&a, double b) {
  a /= b;
  return a;
}
/* ------------------------ Operator+ ------------------------ */
VarGrad operator+(const VarGrad &a, const VarGrad &b) {
  VarGrad res(a);
  res += b;
  return res;
}
VarGrad operator+(VarGrad &&a, const VarGrad &b) {
  a += b;
  return a;
}

VarGrad operator+(const VarGrad &a, VarGrad &&b) { return std::move(b) + a; }
VarGrad operator+(VarGrad &&a, VarGrad &&b) { return std::move(a) + b; }

VarGrad operator+(const VarGrad &a, double b) {
  VarGrad res(a);
  res += b;
  return res;
}
VarGrad operator+(double b, const VarGrad &a) { return a + b; }
VarGrad operator+(double b, VarGrad &&a) {
  a += b;
  return a;
}

VarGrad operator+(VarGrad &&a, double b) {
  a += b;
  return a;
}

/* ------------------------ Operator- ------------------------ */
VarGrad operator-(const VarGrad &a, const VarGrad &b) {
  VarGrad res(a);
  res -= b;
  return res;
}
VarGrad operator-(VarGrad &&a, const VarGrad &b) {
  a -= b;
  return a;
}
VarGrad operator-(const VarGrad &a, VarGrad &&b) {
  b.neg() += a;
  return b;
}
VarGrad operator-(VarGrad &&a, VarGrad &&b) { return std::move(a) - b; }

VarGrad operator-(double b, const VarGrad &a) {
  VarGrad res(-a);
  res += b;
  return res;
}
VarGrad operator-(const VarGrad &a, double b) {
  VarGrad res(a);
  res -= b;
  return res;
}
VarGrad operator-(double b, VarGrad &&a) {
  a.neg() += b;
  return a;
}
VarGrad operator-(VarGrad &&a, double b) {
  a -= b;
  return a;
}
VarGrad operator-(const VarGrad &a) { return a.chain(-a._val, -1.0); }
VarGrad operator-(VarGrad &&a) {
  a.neg();
  return a;
}

VarGrad sqrt(const VarGrad &a) {
  const auto &sqrt_a = std::sqrt(a._val);
  return a.chain(sqrt_a, 0.5 / sqrt_a);
}
VarGrad sqrt(VarGrad &&a) {
  const auto &sqrt_a = std::sqrt(a._val);
  a.chain_this(sqrt_a, 0.5 / sqrt_a);
  return a;
}
VarGrad sqr(const VarGrad &a) {
  // return a * a;
  return a.chain(a._val * a._val, 2 * a._val);
}
VarGrad sqr(VarGrad &&a) {
  a.chain_this(a._val * a._val, 2 * a._val);
  return a;
}
VarGrad abs(const VarGrad &a) {
  return a.chain(a._val, a._val >= 0 ? 1 : -1);
}
VarGrad abs(VarGrad &&a) {
  a.chain_this(a._val, a._val >= 0 ? 1 : -1);
  return a;
}
VarGrad pow(const VarGrad &a, const int exponent) {
  const double f1 = std::pow(a.val(), exponent - 1);
  const double f = f1 * a.val();

  return a.chain(f, exponent * f1);
}
VarGrad pow(const VarGrad &a, const double exponent) {
  const double f1 = std::pow(a.val(), exponent - 1);
  const double f = f1 * a.val();

  return a.chain(f, exponent * f1);
}
VarGrad pow(VarGrad &&a, const int exponent) {
  const double f1 = std::pow(a.val(), exponent - 1);
  const double f = f1 * a.val();
  a.chain_this(f, exponent * f1);
  return a;
}
VarGrad pow(VarGrad &&a, const double exponent) {
  const double f1 = std::pow(a.val(), exponent - 1);
  const double f = f1 * a.val();
  a.chain_this(f, exponent * f1);
  return a;
}

VarGrad exp(const VarGrad &a) {
  const double val = std::exp(a._val);
  return a.chain(val, val);
}
VarGrad exp(VarGrad &&a) {
  const double val = std::exp(a._val);
  a.chain_this(val, val);
  return a;
}
VarGrad log(const VarGrad &a) {
  return a.chain(std::log(a._val), 1 / a._val);
}
VarGrad log(VarGrad &&a) {
  a.chain_this(std::log(a._val), 1 / a._val);
  return a;
}
VarGrad cos(const VarGrad &a) {
  double cosval = std::cos(a._val);
  return a.chain(cosval, -std::sin(a._val));
}
VarGrad cos(VarGrad &&a) {
  double cosval = std::cos(a._val);
  a.chain_this(cosval, -std::sin(a._val));
  return a;
}
VarGrad sin(const VarGrad &a) {
  double sinval = std::sin(a._val);
  return a.chain(sinval, std::cos(a._val));
}
VarGrad sin(VarGrad &&a) {
  double sinval = std::sin(a._val);
  a.chain_this(sinval, std::cos(a._val));
  return a;
}
VarGrad atan(const VarGrad &x) {
  double denom = (x._val * x._val + 1);
  return x.chain(std::atan(x._val), 1 / denom);
}
VarGrad atan2(const VarGrad &y, const VarGrad &x) {
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
VarGrad chain2(const VarGrad &x, const VarGrad &y, double val, double dx,
               double dy, double dxdx, double dxdy, double dydy) {
  VarGrad res(val);
  res._grad.add(x._grad, dx);
  res._grad.add(y._grad, dy);
  return res;
}
bool isfinite(const VarGrad &x) { return std::isfinite(x._val); }
bool isinf(const VarGrad &x) { return std::isinf(x._val); }

// ----------------------- Comparisons -----------------------
bool operator<(const VarGrad &a, const VarGrad &b) { return a._val < b._val; }
bool operator<=(const VarGrad &a, const VarGrad &b) { return a._val <= b._val; }
bool operator>(const VarGrad &a, const VarGrad &b) { return a._val > b._val; }
bool operator>=(const VarGrad &a, const VarGrad &b) { return a._val >= b._val; }
bool operator==(const VarGrad &a, const VarGrad &b) { return a._val == b._val; }
bool operator!=(const VarGrad &a, const VarGrad &b) { return a._val != b._val; }

} // namespace Optiz