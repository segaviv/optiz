#pragma once

#include "MetaGrad.h"
#include "MetaUtils.h"
#include "MetaVec.h"
#include <Eigen/Eigen>
#include <iostream>
#include <string>
#include <utility>

namespace Optiz {

#define GRAD_TYPE(x) decltype(std::declval<x>()._grad)

/**
 * @brief Calculates gradient in forward mode, hessian in reverse.
 * Quick & dirty implementation. Maybe it works, who knows.
 * The variable indices have to be known at compile time.
 *
 * @param Derived the derived expression
 */
template <typename Derived> class MetaVarBase {
public:
  static constexpr bool is_leaf = false;

  double val() const { return static_cast<const Derived *>(this)->val(); }

  decltype(auto) first_var() const {
    auto _grad = static_cast<const Derived *>(this)->_grad;
    constexpr int N = decltype(_grad)::first_var();
    return N;
  }

  decltype(auto) grad() const {
    auto _grad = static_cast<const Derived *>(this)->_grad;
    constexpr int N = decltype(_grad)::num_vars();
    Eigen::Matrix<double, N, 1> res = Eigen::Matrix<double, N, 1>::Zero();
    _grad.for_each(
        [&](const auto &elem) { res(TYPE(elem)::Index) += elem.val; });
    return res;
  }

  decltype(auto) meta_grad() const {
    return static_cast<const Derived *>(this)->_grad;
  }

  template <bool Full = true> decltype(auto) hessian() {
    constexpr int N =
        decltype(static_cast<const Derived *>(this)->_grad)::num_vars();
    Eigen::Matrix<double, N, N> hess = Eigen::Matrix<double, N, N>::Zero();
    static_cast<const Derived *>(this)->add_to_hess(hess);
    if constexpr (Full) {
      return (decltype(hess))hess.template selfadjointView<Eigen::Lower>();
    } else {
      return hess;
    }
  }

  template <bool Full = false> decltype(auto) squeezed_hessian() {
    constexpr int N =
        decltype(static_cast<const Derived *>(this)->_grad)::num_vars();
    constexpr int M =
        decltype(static_cast<const Derived *>(this)->_grad)::first_var();
    Eigen::Matrix<double, N - M, N - M> hess =
        Eigen::Matrix<double, N - M, N - M>::Zero();
    static_cast<const Derived *>(this)->template add_to_hess<M>(hess);
    if constexpr (Full) {
      return (decltype(hess))hess.template selfadjointView<Eigen::Lower>();
    } else {
      return hess;
    }
  }

  std::string print() const {
    return static_cast<const Derived *>(this)->print();
  }

  friend std::ostream &operator<<(std::ostream &s, MetaVarBase const &expr) {
    return s << "MetaVar(" << expr.print() << ")";
  }
};

class MetaVarZero : public MetaVarBase<MetaVarZero> {
public:
  static constexpr bool is_leaf = false;

  MetaVarZero() {}

  static const MetaGrad<> _grad;

  double val() const { return 0.0; }

  template <int M = 0, typename Hess>
  void add_to_hess(Hess &hess, double mul = 1.0) const {}

  std::string print() const { return "0"; }
};

class MetaVarScalar : public MetaVarBase<MetaVarScalar> {
public:
  static constexpr bool is_leaf = false;
  double _value;
  static const MetaGrad<> _grad;

  MetaVarScalar(double value) : _value(value) {}

  double val() const { return _value; }

  template <int M = 0, typename Hess>
  void add_to_hess(Hess &hess, double mul = 1.0) const {}

  std::string print() const { return std::to_string(_value); }
};

template <int V> class MetaVar : public MetaVarBase<MetaVar<V>> {
public:
  static constexpr bool is_leaf = false;
  double _value;
  static const inline auto _grad = MetaGrad(MetaVec(GradEntry<V>(1)));

  MetaVar(double value) : _value(value) {}

  double val() const { return _value; }

  template <int M = 0, typename Hess>
  void add_to_hess(Hess &hess, double mul = 1.0) const {}

  std::string print() const {
    std::ostringstream out(std::ios::fixed);
    out << "x" << V;
    return out.str();
  }
};

template <typename Derived1, typename GradType = GRAD_TYPE(Derived1)>
class MetaVarScalarMul : public MetaVarBase<MetaVarScalarMul<Derived1>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  double _value;
  double scalar;

  const GradType _grad;

  MetaVarScalarMul(const Derived1 &derived1, double scalar)
      : _derived1(derived1), _value(derived1.val() * scalar), scalar(scalar),
        _grad(derived1._grad * scalar) {}

  double val() const { return _value; }

  template <int M = 0, typename Hess>
  void add_to_hess(Hess &hess, double mul = 1.0) const {
    _derived1.template add_to_hess<M>(hess, mul * scalar);
  }

  std::string print() const {
    return _derived1.print() + " * " + std::to_string(scalar);
  }
};

template <typename Derived1, typename GradType>
class MetaVarChain : public MetaVarBase<MetaVarChain<Derived1>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  double _value;
  double grad;
  double chain_hess;
  const GradType _grad;

  MetaVarChain(const Derived1 &derived1, double val, double grad, double hess)
      : _derived1(derived1), _value(val), _grad(derived1._grad * grad),
        grad(grad), chain_hess(hess) {}

  double val() const { return _value; }

  template <int M = 0, typename Hess>
  void add_to_hess(Hess &hess, double mul = 1.0) const {
    _derived1.template add_to_hess<M>(hess, mul * grad);
    _derived1._grad.template rank_update<M>(hess, mul * chain_hess);
  }

  std::string print() const {
    return "chain(" + _derived1.print() + ", " + grad + ", " + chain_hess + ")";
  }
};

template <typename Derived1, typename Derived2,
          typename GradType = decltype(std::declval<GRAD_TYPE(Derived1)>() +
                                       std::declval<GRAD_TYPE(Derived2)>())>
class MetaVarAdd : public MetaVarBase<MetaVarAdd<Derived1, Derived2>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  typename std::conditional<Derived2::is_leaf, const Derived2 &,
                            const Derived2>::type _derived2;
  double _value;
  const GradType _grad;

  MetaVarAdd(const Derived1 &derived1, const Derived2 &derived2)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() + derived2.val()),
        _grad(derived1._grad + derived2._grad) {}

  double val() const { return _value; }

  template <int M = 0, typename Hess>
  void add_to_hess(Hess &hess, double mul = 1.0) const {
    _derived1.template add_to_hess<M>(hess, mul);
    _derived2.template add_to_hess<M>(hess, mul);
  }

  std::string print() const {
    return "(" + _derived1.print() + " + " + _derived2.print() + ")";
  }
};

template <typename Derived1, typename Derived2,
          typename GradType = decltype(std::declval<GRAD_TYPE(Derived1)>() +
                                       std::declval<GRAD_TYPE(Derived2)>())>
class MetaVarSub : public MetaVarBase<MetaVarSub<Derived1, Derived2>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  typename std::conditional<Derived2::is_leaf, const Derived2 &,
                            const Derived2>::type _derived2;

  double _value;
  const GradType _grad;

  MetaVarSub(const Derived1 &derived1, const Derived2 &derived2)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() - derived2.val()),
        _grad(derived1._grad - derived2._grad) {}

  double val() const { return _value; }

  template <int M = 0, typename Hess>
  void add_to_hess(Hess &hess, double mul = 1.0) const {
    _derived1.template add_to_hess<M>(hess, mul);
    _derived2.template add_to_hess<M>(hess, -mul);
  }

  std::string print() const {
    return "(" + _derived1.print() + " - " + _derived2.print() + ")";
  }
};

template <typename Derived1, typename Derived2,
          typename GradType = decltype(std::declval<GRAD_TYPE(Derived1)>() +
                                       std::declval<GRAD_TYPE(Derived2)>())>
class MetaVarMul : public MetaVarBase<MetaVarMul<Derived1, Derived2>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  typename std::conditional<Derived2::is_leaf, const Derived2 &,
                            const Derived2>::type _derived2;
  double _value;
  const GradType _grad;

  MetaVarMul(const Derived1 &derived1, const Derived2 &derived2)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() * derived2.val()),
        _grad(derived1._grad * derived2.val() +
              derived2._grad * derived1.val()) {}

  double val() const { return _value; }

  template <int M = 0, typename Hess>
  void add_to_hess(Hess &hess, double mul = 1.0) const {
    _derived1.template add_to_hess<M>(hess, mul * _derived2.val());
    _derived2.template add_to_hess<M>(hess, mul * _derived1.val());
    _derived1._grad.template rank_update<M>(hess, _derived2._grad, mul);
  }

  std::string print() const {
    return _derived1.print() + " * " + _derived2.print();
  }
};

template <typename Derived1, typename Derived2,
          typename GradType = decltype(std::declval<GRAD_TYPE(Derived1)>() +
                                       std::declval<GRAD_TYPE(Derived2)>())>
class MetaVarDiv : public MetaVarBase<MetaVarDiv<Derived1, Derived2>> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  typename std::conditional<Derived2::is_leaf, const Derived2 &,
                            const Derived2>::type _derived2;
  double _value;
  const GradType _grad;

  MetaVarDiv(const Derived1 &derived1, const Derived2 &derived2)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() / derived2.val()),
        _grad(derived1._grad * (1.0 / derived2.val()) -
              derived2._grad * (_value / derived2.val())) {}

  double val() const { return _value; }

  template <int M = 0, typename Hess>
  void add_to_hess(Hess &hess, double mul = 1.0) const {
    _derived2.template add_to_hess<M>(hess, -mul * _value / _derived2.val());
    _derived1.template add_to_hess<M>(hess, mul / _derived2.val());
    _grad.template rank_update<M>(hess, _derived2._grad,
                                  -mul / _derived2.val());
  }

  std::string print() const {
    return _derived1.print() + " / " + _derived2.print();
  }
};

// Operators.

template <typename Derived1, typename Derived2>
MetaVarAdd<Derived1, Derived2>
operator+(MetaVarBase<Derived1> const &derived1,
          MetaVarBase<Derived2> const &derived2) {
  return MetaVarAdd<Derived1, Derived2>(
      *static_cast<const Derived1 *>(&derived1),
      *static_cast<const Derived2 *>(&derived2));
}

template <
    typename Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
decltype(auto) operator+(MetaVarBase<Derived1> const &derived1, G scalar) {
  return derived1 + MetaVarScalar(scalar);
}

template <
    typename Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
decltype(auto) operator+(G scalar, MetaVarBase<Derived1> const &derived1) {
  return MetaVarScalar(scalar) + derived1;
}

template <typename Derived1, typename Derived2>
MetaVarSub<Derived1, Derived2>
operator-(MetaVarBase<Derived1> const &derived1,
          MetaVarBase<Derived2> const &derived2) {
  return MetaVarSub<Derived1, Derived2>(
      *static_cast<const Derived1 *>(&derived1),
      *static_cast<const Derived2 *>(&derived2));
}

template <
    typename Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
decltype(auto) operator-(MetaVarBase<Derived1> const &derived1, G scalar) {
  return derived1 - MetaVarScalar(scalar);
}

template <
    typename Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
decltype(auto) operator-(G scalar, MetaVarBase<Derived1> const &derived1) {
  return MetaVarScalar(scalar) - derived1;
}

template <typename Derived1, typename Derived2>
MetaVarMul<Derived1, Derived2>
operator*(MetaVarBase<Derived1> const &derived1,
          MetaVarBase<Derived2> const &derived2) {
  return MetaVarMul<Derived1, Derived2>(
      *static_cast<const Derived1 *>(&derived1),
      *static_cast<const Derived2 *>(&derived2));
}

template <
    typename Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
MetaVarScalarMul<Derived1> operator*(MetaVarBase<Derived1> const &derived1,
                                     const G &scalar) {
  return MetaVarScalarMul<Derived1>(*static_cast<const Derived1 *>(&derived1),
                                    scalar);
}

template <
    typename Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
MetaVarScalarMul<Derived1> operator*(const G &scalar,
                                     MetaVarBase<Derived1> const &derived1) {
  return MetaVarScalarMul<Derived1>(*static_cast<const Derived1 *>(&derived1),
                                    scalar);
}

template <
    typename Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
MetaVarScalarMul<Derived1> operator/(MetaVarBase<Derived1> const &derived1,
                                     const G &scalar) {
  return MetaVarScalarMul<Derived1>(*static_cast<const Derived1 *>(&derived1),
                                    1.0 / scalar);
}

template <typename Derived1>
MetaVarChain<Derived1> inv(MetaVarBase<Derived1> const &derived1) {
  double valsqr = derived1.val() * derived1.val();
  double valcube = valsqr * derived1.val();
  return MetaVarChain<Derived1>(*static_cast<const Derived1 *>(&derived1),
                                1 / derived1.val(), -1 / valsqr, 2 / valcube);
}

template <
    typename Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
MetaVarDiv<MetaVarScalar, Derived1>
operator/(const G &scalar, MetaVarBase<Derived1> const &derived1) {
  return MetaVarDiv<MetaVarScalar, Derived1>(
      MetaVarScalar(scalar), *static_cast<const Derived1 *>(&derived1));
}

template <typename Derived1, typename Derived2>
MetaVarDiv<Derived1, Derived2>
operator/(MetaVarBase<Derived1> const &derived1,
          MetaVarBase<Derived2> const &derived2) {
  return MetaVarDiv<Derived1, Derived2>(
      *static_cast<const Derived1 *>(&derived1),
      *static_cast<const Derived2 *>(&derived2));
}

template <typename Derived1>
MetaVarChain<Derived1> sin(MetaVarBase<Derived1> const &derived1) {
  double sin_val = std::sin(derived1.val());
  return MetaVarChain<Derived1>(*static_cast<const Derived1 *>(&derived1),
                                sin_val, std::cos(derived1.val()), -sin_val);
}

template <typename Derived1>
MetaVarChain<Derived1> cos(MetaVarBase<Derived1> const &derived1) {
  double cos_val = std::cos(derived1.val());
  return MetaVarChain<Derived1>(*static_cast<const Derived1 *>(&derived1),
                                cos_val, -std::sin(derived1.val()), -cos_val);
}

template <typename Derived1>
MetaVarChain<Derived1> exp(MetaVarBase<Derived1> const &derived1) {
  double val = std::exp(derived1.val());
  return MetaVarChain<Derived1>(*static_cast<const Derived1 *>(&derived1), val,
                                val, val);
}

template <typename Derived1>
MetaVarChain<Derived1> log(MetaVarBase<Derived1> const &derived1) {
  return MetaVarChain<Derived1>(*static_cast<const Derived1 *>(&derived1),
                                std::log(derived1.val()), 1.0 / derived1.val(),
                                -1.0 / (derived1.val() * derived1.val()));
}

template <
    typename Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
MetaVarChain<Derived1, G> pow(MetaVarBase<Derived1> const &derived1,
                              const G &exponent) {
  double f2 = std::pow(derived1.val(), exponent - 2);
  double f1 = f2 * derived1.val();
  double f = f1 * derived1.val();
  return MetaVarChain<Derived1, G>(*static_cast<const Derived1 *>(&derived1), f,
                                   exponent * f1,
                                   exponent * (exponent - 1) * f2);
}

template <typename Derived1>
MetaVarChain<Derived1> sqr(MetaVarBase<Derived1> const &derived1) {
  return MetaVarChain<Derived1>(*static_cast<const Derived1 *>(&derived1),
                                derived1.val() * derived1.val(),
                                2 * derived1.val(), 2);
}

template <typename Derived1>
MetaVarChain<Derived1> operator-(MetaVarBase<Derived1> const &derived1) {
  return MetaVarChain<Derived1>(*static_cast<const Derived1 *>(&derived1),
                                -derived1.val(), -1, 0);
}

} // namespace Optiz