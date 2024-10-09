#pragma once

#include "MetaUtils.h"
#include <Eigen/Eigen>
#include <iostream>
#include <string>

namespace Optiz {

template <typename Derived> class ReverseGradMetaVarBase;
template <typename Derived> class ReverseGradMetaVarSin;
template <typename Derived> class ReverseGradMetaVarCos;
template <typename Derived1, typename Derived2> class ReverseGradMetaVarMul;

/**
 * @brief Calculates gradient in reverse mode.
 * Probably useless to calculate the hessian. Unless it's possible to figure out
 * a way to simplify the d_d(i) expression at compile time and share the
 * expression nodes of the graph.
 *
 * @param Derived the derived expression
 */
template <typename Derived> class ReverseGradMetaVarBase {
public:
  static constexpr bool is_leaf = false;

  int n_vars() const { return static_cast<const Derived *>(this)->n_vars(); }

  double val() const { return static_cast<const Derived *>(this)->val(); }

  Eigen::VectorXd grad(int n = -1) {
    if (n == -1) {
      n = n_vars() + 1;
    }
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(n);
    static_cast<const Derived *>(this)->add_to_grad(grad);
    return grad;
  }

  template <int N> Eigen::Matrix<double, N, 1> grad() {
    Eigen::Matrix<double, N, 1> grad = Eigen::Matrix<double, N, 1>::Zero();
    static_cast<const Derived *>(this)->add_to_grad(grad);
    return grad;
  }

  template <int N> Eigen::Matrix<double, N, N> hessian() {
    Eigen::Matrix<double, N, N> hessian = Eigen::Matrix<double, N, N>::Zero();
    Optiz::For<0, N>([&]<int i>(const auto &ind) {
      auto res = static_cast<const Derived *>(this)->template d_d<i>();
      Optiz::For<0, i + 1>([&, res]<int j>(const auto &ind2) {
        hessian(i, j) = res.template d_d<j>().val();
        hessian(j, i) = hessian(i, j);
      });
    });
    return hessian;
  }

  Eigen::MatrixXd hessian(int n = -1) {
    if (n == -1) {
      n = n_vars() + 1;
    }
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; i++) {
      auto d_di = static_cast<const Derived *>(this)->d_d(i);
      for (int j = 0; j <= i; j++) {
        hessian(i, j) = d_di.d_d(j).val();
        hessian(j, i) = hessian(i, j);
      }
    }
    return hessian;
  }

  decltype(auto) d_d(int i) const {
    return static_cast<const Derived *>(this)->d_d(i);
  }

  template <int T> decltype(auto) d_d() const {
    return static_cast<const Derived *>(this)->d_d(T);
  }

  std::string print() const {
    return static_cast<const Derived *>(this)->print();
  }

  friend std::ostream &operator<<(std::ostream &s,
                                  ReverseGradMetaVarBase const &expr) {
    return s << "ReverseGradMetaVar(" << expr.print() << ")";
  }
};

class ReverseGradMetaVarZero
    : public ReverseGradMetaVarBase<ReverseGradMetaVarZero> {
public:
  static constexpr bool is_leaf = false;

  ReverseGradMetaVarZero() {}

  int n_vars() const { return 0; }

  double val() const { return 0.0; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {}

  decltype(auto) d_d(int i) const { return *this; }

  template <int T> decltype(auto) d_d() const { return *this; }

  std::string print() const { return "0"; }
};

class ReverseGradMetaVarScalar
    : public ReverseGradMetaVarBase<ReverseGradMetaVarScalar> {
public:
  static constexpr bool is_leaf = false;
  double _value;

  ReverseGradMetaVarScalar(double value) : _value(value) {}

  int n_vars() const { return 0; }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {}

  decltype(auto) d_d(int i) const { return ReverseGradMetaVarZero(); }

  template <int T> decltype(auto) d_d() const {
    return ReverseGradMetaVarZero();
  }

  std::string print() const { return std::to_string(_value); }
};

class ReverseGradMetaVarLeaf
    : public ReverseGradMetaVarBase<ReverseGradMetaVarLeaf> {
public:
  static constexpr bool is_leaf = true;
  int _index;
  double _value;
  ReverseGradMetaVarLeaf(double value, int index)
      : _value(value), _index(index) {}

  int n_vars() const { return _index; }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {
    grad(_index) += mul;
  }

  decltype(auto) d_d(int i) const {
    return ReverseGradMetaVarScalar((i == _index) ? 1.0 : 0.0);
  }

  template <int T> decltype(auto) d_d() const { return d_d(T); }

  std::string print() const {
    std::ostringstream out;
    out << "x" << _index;
    return out.str();
  }
};

template <int V>
class ReverseGradMetaVarLeafT
    : public ReverseGradMetaVarBase<ReverseGradMetaVarLeafT<V>> {
public:
  static constexpr bool is_leaf = false;
  double _value;

  ReverseGradMetaVarLeafT(double value) : _value(value) {}

  int n_vars() const { return V; }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {
    grad(V) += mul;
  }

  decltype(auto) d_d(int i) const {
    return ReverseGradMetaVarScalar((i == V) ? 1.0 : 0.0);
  }

  template <int T> decltype(auto) d_d() const {
    if constexpr (V == T) {
      return ReverseGradMetaVarScalar(1.0);
    } else {
      return ReverseGradMetaVarZero();
    }
  }

  std::string print() const {
    std::ostringstream out(std::ios::fixed);
    out << "x" << V;
    return out.str();
  }
};

template <typename Derived1>
class ReverseGradMetaVarScalarMul
    : public ReverseGradMetaVarBase<ReverseGradMetaVarScalarMul<Derived1>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  double _value;
  double scalar;

  ReverseGradMetaVarScalarMul(const Derived1 &derived1, double scalar)
      : _derived1(derived1), _value(derived1.val() * scalar), scalar(scalar) {}

  int n_vars() const { return _derived1.n_vars(); }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {
    _derived1.add_to_grad(grad, mul * scalar);
  }

  auto d_d(int i) const {
    auto a = _derived1.d_d(i);
    return ReverseGradMetaVarScalarMul<TYPE(a)>(a, scalar).simplified();
  }

  template <int T> decltype(auto) d_d() const {
    auto a = _derived1.template d_d<T>();
    return ReverseGradMetaVarScalarMul<TYPE(a)>(a, scalar).simplified();
  }

  decltype(auto) simplified() const {
    if constexpr (std::is_same_v<TYPE(_derived1), ReverseGradMetaVarZero>) {
      return ReverseGradMetaVarZero();
    } else if constexpr (std::is_same_v<TYPE(_derived1),
                                        ReverseGradMetaVarScalar>) {
      return ReverseGradMetaVarScalar(_derived1.val() * scalar);
    } else if constexpr (is_specialization<
                             TYPE(_derived1),
                             ReverseGradMetaVarScalarMul>::value) {
      return ReverseGradMetaVarScalarMul<TYPE(_derived1._derived1)>(
          _derived1._derived1, _derived1.scalar * scalar);
    } else {
      return ReverseGradMetaVarScalarMul(_derived1, scalar);
    }
  }

  std::string print() const {
    return _derived1.print() + " * " + std::to_string(scalar);
  }
};

template <typename Derived1, typename T>
class ReverseGradMetaVarPow
    : public ReverseGradMetaVarBase<ReverseGradMetaVarPow<Derived1, T>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  double _value;
  T scalar;

  ReverseGradMetaVarPow(const Derived1 &derived1, T scalar)
      : _derived1(derived1), _value(std::pow(derived1.val(), scalar)),
        scalar(scalar) {}

  int n_vars() const { return _derived1.n_vars(); }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {
    _derived1.add_to_grad(grad,
                          mul * scalar * std::pow(_derived1.val(), scalar - 1));
  }

  auto d_d(int i) const {
    auto a = _derived1.d_d(i);
    return ReverseGradMetaVarMul(ReverseGradMetaVarScalarMul(a, scalar),
                                 ReverseGradMetaVarPow(_derived1, scalar - 1));
  }

  template <int G> decltype(auto) d_d() const {
    auto a = _derived1.template d_d<G>();
    if constexpr (std::is_same_v<decltype(a), ReverseGradMetaVarZero>) {
      return a;
    } else if constexpr (std::is_same_v<decltype(a),
                                        ReverseGradMetaVarScalar>) {
      return ReverseGradMetaVarScalarMul(
          ReverseGradMetaVarPow(_derived1, scalar - 1), a.val() * scalar);
    } else {
      return ReverseGradMetaVarMul(
          ReverseGradMetaVarScalarMul(a, scalar),
          ReverseGradMetaVarPow(_derived1, scalar - 1));
    }
  }

  std::string print() const {
    return "pow(" + _derived1.print() + ", " + std::to_string(scalar) + ")";
  }
};

template <typename Derived1>
class ReverseGradMetaVarSqr
    : public ReverseGradMetaVarBase<ReverseGradMetaVarSqr<Derived1>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  double _value;

  ReverseGradMetaVarSqr(const Derived1 &derived1)
      : _derived1(derived1), _value(derived1.val() * derived1.val()) {}

  int n_vars() const { return _derived1.n_vars(); }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {
    _derived1.add_to_grad(grad, mul * 2 * _derived1.val());
  }

  auto d_d(int i) const {
    auto a = _derived1.d_d(i);
    return ReverseGradMetaVarMul(ReverseGradMetaVarScalarMul(a, 2), _derived1);
  }

  template <int G> decltype(auto) d_d() const {
    auto a = _derived1.template d_d<G>();
    if constexpr (std::is_same_v<decltype(a), ReverseGradMetaVarZero>) {
      return a;
    } else if constexpr (std::is_same_v<decltype(a),
                                        ReverseGradMetaVarScalar>) {
      return ReverseGradMetaVarScalarMul(_derived1, a.val() * 2).simplified();
    } else {
      return ReverseGradMetaVarMul(
                 ReverseGradMetaVarScalarMul(a, 2).simplified(), _derived1)
          .simplified();
    }
  }

  std::string print() const { return "(" + _derived1.print() + ")^2"; }
};

template <typename Derived1, typename Derived2>
class ReverseGradMetaVarAdd
    : public ReverseGradMetaVarBase<ReverseGradMetaVarAdd<Derived1, Derived2>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  typename std::conditional<Derived2::is_leaf, const Derived2 &,
                            const Derived2>::type _derived2;
  double _value;

  ReverseGradMetaVarAdd(const Derived1 &derived1, const Derived2 &derived2)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() + derived2.val()) {}

  int n_vars() const {
    return std::max(_derived1.n_vars(), _derived2.n_vars());
  }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {
    _derived1.add_to_grad(grad, mul);
    _derived2.add_to_grad(grad, mul);
  }

  decltype(auto) d_d(int i) const {
    auto a = _derived1.d_d(i);
    auto b = _derived2.d_d(i);
    return ReverseGradMetaVarAdd<TYPE(a), TYPE(b)>(a, b);
  }

  template <int T> decltype(auto) d_d() const {
    auto a = _derived1.template d_d<T>();
    auto b = _derived2.template d_d<T>();
    return ReverseGradMetaVarAdd<TYPE(a), TYPE(b)>(a, b).simplified();
  }

  decltype(auto) simplified() const {
    if constexpr (std::is_same_v<TYPE(_derived1), ReverseGradMetaVarZero>) {
      return _derived2;
    } else if constexpr (std::is_same_v<TYPE(_derived2),
                                        ReverseGradMetaVarZero>) {
      return _derived1;
    } else if constexpr (std::is_same_v<TYPE(_derived1),
                                        ReverseGradMetaVarScalar> &&
                         std::is_same_v<TYPE(_derived2),
                                        ReverseGradMetaVarScalar>) {
      return ReverseGradMetaVarScalar(_derived1.val() + _derived2.val());
    } else {
      return ReverseGradMetaVarAdd(_derived1, _derived2);
    }
  }

  std::string print() const {
    return "(" + _derived1.print() + " + " + _derived2.print() + ")";
  }
};

template <typename Derived1, typename Derived2>
class ReverseGradMetaVarSub
    : public ReverseGradMetaVarBase<ReverseGradMetaVarSub<Derived1, Derived2>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  typename std::conditional<Derived2::is_leaf, const Derived2 &,
                            const Derived2>::type _derived2;

  double _value;

  ReverseGradMetaVarSub(const Derived1 &derived1, const Derived2 &derived2)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() - derived2.val()) {}

  int n_vars() const {
    return std::max(_derived1.n_vars(), _derived2.n_vars());
  }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {
    _derived1.add_to_grad(grad, mul);
    _derived2.add_to_grad(grad, -mul);
  }

  decltype(auto) d_d(int i) const {
    auto a = _derived1.d_d(i);
    auto b = _derived2.d_d(i);
    return ReverseGradMetaVarSub<TYPE(a), TYPE(b)>(a, b).simplified();
  }

  template <int T> decltype(auto) d_d() const {
    auto a = _derived1.template d_d<T>();
    auto b = _derived2.template d_d<T>();
    return ReverseGradMetaVarSub<TYPE(a), TYPE(b)>(a, b).simplified();
  }

  decltype(auto) simplified() const {
    if constexpr (std::is_same_v<TYPE(_derived1), ReverseGradMetaVarZero>) {
      if constexpr (std::is_same_v<TYPE(_derived2), ReverseGradMetaVarZero>) {
        return ReverseGradMetaVarZero();
      } else {
        // TODO: Replace with neg(_derived2).
        return ReverseGradMetaVarScalarMul(_derived2, -1.0).simplified();
      }
    } else if constexpr (std::is_same_v<TYPE(_derived2),
                                        ReverseGradMetaVarZero>) {
      return _derived1;
    } else if constexpr (std::is_same_v<TYPE(_derived1),
                                        ReverseGradMetaVarScalar> &&
                         std::is_same_v<TYPE(_derived2),
                                        ReverseGradMetaVarScalar>) {
      return ReverseGradMetaVarScalar(_derived1.val() - _derived2.val());
    } else {
      return ReverseGradMetaVarSub(_derived1, _derived2);
    }
  }

  std::string print() const {
    return "(" + _derived1.print() + " - " + _derived2.print() + ")";
  }
};

template <typename Derived1, typename Derived2>
class ReverseGradMetaVarMul
    : public ReverseGradMetaVarBase<ReverseGradMetaVarMul<Derived1, Derived2>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  typename std::conditional<Derived2::is_leaf, const Derived2 &,
                            const Derived2>::type _derived2;
  double _value;

  ReverseGradMetaVarMul(const Derived1 &derived1, const Derived2 &derived2)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() * derived2.val()) {}

  int n_vars() const {
    return std::max(_derived1.n_vars(), _derived2.n_vars());
  }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {
    _derived1.add_to_grad(grad, _derived2.val() * mul);
    _derived2.add_to_grad(grad, _derived1.val() * mul);
  }

  decltype(auto) d_d(int i) const {
    auto a = _derived1.d_d(i);
    auto b = _derived2.d_d(i);
    return ReverseGradMetaVarAdd(
        ReverseGradMetaVarMul<TYPE(a), TYPE(_derived2)>(a, _derived2),
        ReverseGradMetaVarMul<TYPE(_derived1), TYPE(b)>(_derived1, b));
  }

  template <int T> decltype(auto) d_d() const {
    auto a = _derived1.template d_d<T>();
    auto b = _derived2.template d_d<T>();
    return ReverseGradMetaVarAdd(
               ReverseGradMetaVarMul<TYPE(a), TYPE(_derived2)>(a, _derived2)
                   .simplified(),
               ReverseGradMetaVarMul<TYPE(_derived1), TYPE(b)>(_derived1, b)
                   .simplified())
        .simplified();
  }

  decltype(auto) simplified() const {
    if constexpr (std::is_same_v<TYPE(_derived1), ReverseGradMetaVarZero> ||
                  std::is_same_v<TYPE(_derived2), ReverseGradMetaVarZero>) {
      return ReverseGradMetaVarZero();
    } else if constexpr (std::is_same_v<TYPE(_derived1),
                                        ReverseGradMetaVarScalar>) {
      if constexpr (std::is_same_v<TYPE(_derived2), ReverseGradMetaVarScalar>) {
        return ReverseGradMetaVarScalar(_derived1.val() * _derived2.val());
      } else {
        return ReverseGradMetaVarScalarMul(_derived2, _derived1.val());
      }
    } else if constexpr (std::is_same_v<TYPE(_derived2),
                                        ReverseGradMetaVarScalar>) {
      return ReverseGradMetaVarScalarMul(_derived1, _derived2.val());
    } else {
      return ReverseGradMetaVarMul(_derived1, _derived2);
    }
  }

  std::string print() const {
    return _derived1.print() + " * " + _derived2.print();
  }
};

template <typename Derived1, typename Derived2>
class ReverseGradMetaVarDiv
    : public ReverseGradMetaVarBase<ReverseGradMetaVarDiv<Derived1, Derived2>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  typename std::conditional<Derived2::is_leaf, const Derived2 &,
                            const Derived2>::type _derived2;
  double _value;

  ReverseGradMetaVarDiv(const Derived1 &derived1, const Derived2 &derived2)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() / derived2.val()) {}

  int n_vars() const {
    return std::max(_derived1.n_vars(), _derived2.n_vars());
  }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {
    _derived1.add_to_grad(grad, mul / _derived2.val());
    _derived2.add_to_grad(grad, -_derived1.val() * mul /
                                    (_derived2.val() * _derived2.val()));
  }

  decltype(auto) d_d(int i) const {
    // TODO: use a more direct way?
    return ReverseGradMetaVarMul(_derived1,
                                 ReverseGradMetaVarPow(_derived2, -1))
        .d_d(i);
  }

  template <int T> decltype(auto) d_d() const {
    return ReverseGradMetaVarMul(_derived1,
                                 ReverseGradMetaVarPow(_derived2, -1))
        .template d_d<T>();
  }

  std::string print() const {
    return _derived1.print() + " / " + _derived2.print();
  }
};

template <typename Derived1>
class ReverseGradMetaVarCos
    : public ReverseGradMetaVarBase<ReverseGradMetaVarCos<Derived1>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  double _value;

  ReverseGradMetaVarCos(const Derived1 &derived1)
      : _derived1(derived1), _value(std::cos(derived1.val())) {}

  int n_vars() const { return _derived1.n_vars(); }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {
    _derived1.add_to_grad(grad, mul * -std::sin(_derived1.val()));
  }

  decltype(auto) d_d(int i) const {
    // TODO: use a more direct way?
    return ReverseGradMetaVarScalarMul(
        ReverseGradMetaVarMul(ReverseGradMetaVarSin(_derived1),
                              _derived1.d_d(i)),
        -1);
  }

  template <int T> decltype(auto) d_d() const {
    // TODO: use a more direct way?
    return ReverseGradMetaVarMul(ReverseGradMetaVarScalarMul(
                                     ReverseGradMetaVarSin(_derived1), -1),
                                 _derived1.template d_d<T>())
        .simplified();
  }

  std::string print() const { return "cos(" + _derived1.print() + ")"; }
};

template <typename Derived1>
class ReverseGradMetaVarSin
    : public ReverseGradMetaVarBase<ReverseGradMetaVarSin<Derived1>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  double _value;

  ReverseGradMetaVarSin(const Derived1 &derived1)
      : _derived1(derived1), _value(std::sin(derived1.val())) {}

  int n_vars() const { return _derived1.n_vars(); }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {
    _derived1.add_to_grad(grad, mul * std::cos(_derived1.val()));
  }

  decltype(auto) d_d(int i) const {
    // TODO: use a more direct way?
    return ReverseGradMetaVarMul(ReverseGradMetaVarCos(_derived1),
                                 _derived1.d_d(i));
  }

  template <int T> decltype(auto) d_d() const {
    // TODO: use a more direct way?
    return ReverseGradMetaVarMul(ReverseGradMetaVarCos(_derived1),
                                 _derived1.template d_d<T>())
        .simplified();
  }

  std::string print() const { return "sin(" + _derived1.print() + ")"; }
};

template <typename Derived1>
class ReverseGradMetaVarExp
    : public ReverseGradMetaVarBase<ReverseGradMetaVarExp<Derived1>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  double _value;

  ReverseGradMetaVarExp(const Derived1 &derived1)
      : _derived1(derived1), _value(std::exp(derived1.val())) {}

  int n_vars() const { return _derived1.n_vars(); }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {
    _derived1.add_to_grad(grad, mul * std::exp(_derived1.val()));
  }

  decltype(auto) d_d(int i) const {
    // TODO: use a more direct way?
    return ReverseGradMetaVarMul(ReverseGradMetaVarExp(_derived1),
                                 _derived1.d_d(i));
  }

  std::string print() const { return "exp(" + _derived1.print() + ")"; }
};

template <typename Derived1>
class ReverseGradMetaVarLog
    : public ReverseGradMetaVarBase<ReverseGradMetaVarLog<Derived1>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  double _value;

  ReverseGradMetaVarLog(const Derived1 &derived1)
      : _derived1(derived1), _value(std::log(derived1.val())) {}

  int n_vars() const { return _derived1.n_vars(); }

  double val() const { return _value; }

  template <typename Grad>
  void add_to_grad(Grad &grad, double mul = 1.0) const {
    _derived1.add_to_grad(grad, mul / _derived1.val());
  }

  decltype(auto) d_d(int i) const {
    return ReverseGradMetaVarDiv(_derived1.d_d(i), _derived1);
  }

  std::string print() const { return "log(" + _derived1.print() + ")"; }
};

// Operators.

template <typename Derived1, typename Derived2>
ReverseGradMetaVarAdd<Derived1, Derived2>
operator+(ReverseGradMetaVarBase<Derived1> const &derived1,
          ReverseGradMetaVarBase<Derived2> const &derived2) {
  return ReverseGradMetaVarAdd<Derived1, Derived2>(
             *static_cast<const Derived1 *>(&derived1),
             *static_cast<const Derived2 *>(&derived2))
      .simplified();
}

template <typename Derived1, typename Derived2>
ReverseGradMetaVarSub<Derived1, Derived2>
operator-(ReverseGradMetaVarBase<Derived1> const &derived1,
          ReverseGradMetaVarBase<Derived2> const &derived2) {
  return ReverseGradMetaVarSub<Derived1, Derived2>(
             *static_cast<const Derived1 *>(&derived1),
             *static_cast<const Derived2 *>(&derived2))
      .simplified();
}

template <typename Derived1, typename Derived2>
ReverseGradMetaVarMul<Derived1, Derived2>
operator*(ReverseGradMetaVarBase<Derived1> const &derived1,
          ReverseGradMetaVarBase<Derived2> const &derived2) {
  return ReverseGradMetaVarMul<Derived1, Derived2>(
             *static_cast<const Derived1 *>(&derived1),
             *static_cast<const Derived2 *>(&derived2))
      .simplified();
}

template <
    typename Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
ReverseGradMetaVarScalarMul<Derived1>
operator*(ReverseGradMetaVarBase<Derived1> const &derived1, const G &scalar) {
  return ReverseGradMetaVarScalarMul<Derived1>(
      *static_cast<const Derived1 *>(&derived1), scalar);
}

template <
    typename Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
ReverseGradMetaVarScalarMul<Derived1>
operator*(const G &scalar, ReverseGradMetaVarBase<Derived1> const &derived1) {
  return ReverseGradMetaVarScalarMul<Derived1>(
      *static_cast<const Derived1 *>(&derived1), scalar);
}

template <
    typename Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
ReverseGradMetaVarScalarMul<Derived1>
operator/(ReverseGradMetaVarBase<Derived1> const &derived1, const G &scalar) {
  return ReverseGradMetaVarScalarMul<Derived1>(
      *static_cast<const Derived1 *>(&derived1), 1.0 / scalar);
}

template <typename Derived1, typename Derived2>
ReverseGradMetaVarDiv<Derived1, Derived2>
operator/(ReverseGradMetaVarBase<Derived1> const &derived1,
          ReverseGradMetaVarBase<Derived2> const &derived2) {
  return ReverseGradMetaVarDiv<Derived1, Derived2>(
      *static_cast<const Derived1 *>(&derived1),
      *static_cast<const Derived2 *>(&derived2));
}

template <typename Derived1>
ReverseGradMetaVarSin<Derived1>
sin(ReverseGradMetaVarBase<Derived1> const &derived1) {
  return ReverseGradMetaVarSin<Derived1>(
      *static_cast<const Derived1 *>(&derived1));
}

template <typename Derived1>
ReverseGradMetaVarCos<Derived1>
cos(ReverseGradMetaVarBase<Derived1> const &derived1) {
  return ReverseGradMetaVarCos<Derived1>(
      *static_cast<const Derived1 *>(&derived1));
}

template <typename Derived1>
ReverseGradMetaVarExp<Derived1>
exp(ReverseGradMetaVarBase<Derived1> const &derived1) {
  double val = std::exp(derived1.val());
  return ReverseGradMetaVarExp<Derived1>(
      *static_cast<const Derived1 *>(&derived1));
}

template <typename Derived1>
ReverseGradMetaVarLog<Derived1>
log(ReverseGradMetaVarBase<Derived1> const &derived1) {
  return ReverseGradMetaVarLog<Derived1>(
      *static_cast<const Derived1 *>(&derived1));
}

template <
    typename Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
ReverseGradMetaVarPow<Derived1, G>
pow(ReverseGradMetaVarBase<Derived1> const &derived1, const G &exponent) {
  return ReverseGradMetaVarPow<Derived1, G>(
      *static_cast<const Derived1 *>(&derived1), exponent);
}

template <typename Derived1>
ReverseGradMetaVarSqr<Derived1>
sqr(ReverseGradMetaVarBase<Derived1> const &derived1) {
  return ReverseGradMetaVarSqr<Derived1>(
      *static_cast<const Derived1 *>(&derived1));
}

} // namespace Optiz