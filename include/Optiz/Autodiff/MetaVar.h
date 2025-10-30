#pragma once

#include "../Meta/MetaSparseMat.h"
#include "../Meta/MetaUtils.h"
#include "../Meta/MetaVec.h"
#include "MetaGrad.h"
#include <Eigen/Eigen>
#include <iostream>
#include <string>
#include <utility>

namespace Optiz {

#define GRAD_VAL(x) (std::declval<x>()._grad)
#define HESS_VAL(x) (std::declval<x>()._hess)
#define GRAD_TYPE(x) decltype(std::declval<x>()._grad)
#define HESS_TYPE(x) decltype(std::declval<x>()._hess)

template <typename> struct is_meta_var : std::false_type {};

template <typename Derived>
struct is_meta_var<MetaVarBase<Derived>> : std::true_type {};

template <typename Derived> struct inherits_from_meta_base {
private:
  template <typename MetaBaseDerived>
  static std::true_type test(const MetaVarBase<MetaBaseDerived> *);
  static std::false_type test(...);

public:
  static constexpr bool value =
      decltype(test(std::declval<Derived *>()))::value;
};
template <typename T>
concept InheritsFromMetaBase = inherits_from_meta_base<T>::value;

/**
 * @brief Calculates gradient in forward mode, hessian in reverse.
 * Quick & dirty implementation. Maybe it works, who knows.
 * The variable indices have to be known at compile time.
 *
 * @param Derived the derived expression
 */
template <typename Derived> class MetaVarBase {
public:
  static constexpr bool forward_grad = true;
  static constexpr bool forward_hessian = false;

  double val() const { return static_cast<const Derived *>(this)->val(); }

  decltype(auto) first_var() const {
    auto _grad = static_cast<const Derived *>(this)->_grad;
    constexpr int N = decltype(_grad)::first_var();
    return N;
  }

  decltype(auto) grad() const {
    auto _grad = meta_grad();
    constexpr int N = decltype(_grad)::num_vars();
    Eigen::Matrix<double, N, 1> res = Eigen::Matrix<double, N, 1>::Zero();
    _grad.for_each(
        [&](const auto &elem) { res(TYPE(elem)::Index) += elem.val; });
    return res;
  }

  decltype(auto) meta_grad() const
    requires(Derived::forward_grad)
  {
    return static_cast<const Derived *>(this)->_grad;
  }
  decltype(auto) meta_grad() const
    requires(!Derived::forward_grad)
  {
    return static_cast<const Derived *>(this)->reverse_grad();
  }

  template <bool Full = true>
  decltype(auto) hessian()
    requires(!Derived::forward_hessian)
  {
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

  template <bool Full = false>
  decltype(auto) squeezed_hessian()
    requires(!Derived::forward_hessian)
  {
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

class MetaVarScalar : public MetaVarBase<MetaVarScalar> {
public:
  double _value;
  static const inline auto _grad = MetaGrad(MetaVec<>());
  static const inline auto _hess = MetaSparseMat();

  MetaVarScalar(double value) : _value(value) {}

  double val() const { return _value; }

  auto reverse_grad(double mul = 1.0) const { return MetaGrad(MetaVec<>()); }
  template <int M = 0, typename Hess>
  void add_to_hess(Hess &hess, double mul = 1.0) const {}

  std::string print() const { return std::to_string(_value); }
};

template <int V, bool ForwardGrad = true, bool ForwardHessian = false>
class MetaVar : public MetaVarBase<MetaVar<V, ForwardGrad, ForwardHessian>> {
public:
  static constexpr bool forward_grad = ForwardGrad;
  static constexpr bool forward_hessian = ForwardHessian;
  double _value;
  static const inline auto _grad = MetaGrad(MetaVec(GradEntry<V>(1)));
  static const inline auto _hess = MetaSparseMat();

  MetaVar(double value) : _value(value) {}

  double val() const { return _value; }

  auto with_reverse_grad() const { return MetaVar<V, false, false>(_value); }
  auto with_forward_hessian() const {
    return MetaVar<V, ForwardGrad, true>(_value);
  }

  auto reverse_grad(double mul = 1.0) const {
    return MetaGrad(MetaVec(GradEntry<V>(mul)));
  }
  template <int M = 0, typename Hess>
  void add_to_hess(Hess &hess, double mul = 1.0) const {}

  std::string print() const {
    std::ostringstream out(std::ios::fixed);
    out << "x" << V;
    return out.str();
  }
};

template <typename Derived1, typename GradType = GRAD_TYPE(Derived1),
          typename HessType = HESS_TYPE(Derived1)>
class MetaVarScalarMul : public MetaVarBase<MetaVarScalarMul<Derived1>> {
public:
  static constexpr bool forward_grad = Derived1::forward_grad;
  static constexpr bool forward_hessian = Derived1::forward_hessian;

  const Derived1 _derived1;
  double _value;
  double scalar;

  typename std::conditional<forward_grad, const GradType,
                            MetaGrad<MetaVec<>>>::type _grad;
  typename std::conditional<forward_hessian, const HessType,
                            MetaSparseMat<>>::type _hess;

  // Forward grad, reverse hessian.
  MetaVarScalarMul(const Derived1 &derived1, double scalar)
    requires(forward_grad && !forward_hessian)
      : _derived1(derived1), _value(derived1.val() * scalar), scalar(scalar),
        _grad(derived1._grad * scalar) {}
  // Forward grad and hessian.
  MetaVarScalarMul(const Derived1 &derived1, double scalar)
    requires(forward_grad && forward_hessian)
      : _derived1(derived1), _value(derived1.val() * scalar), scalar(scalar),
        _grad(derived1._grad * scalar), _hess(derived1._hess * scalar) {}
  // Reverse grad.
  MetaVarScalarMul(const Derived1 &derived1, double scalar)
    requires(!forward_grad && !forward_hessian)
      : _derived1(derived1), _value(derived1.val() * scalar), scalar(scalar) {}

  double val() const { return _value; }

  auto reverse_grad(double mul = 1.0) const {
    return _derived1.reverse_grad(mul * scalar);
  }
  template <int M = 0, typename Hess>
  void add_to_hess(Hess &hess, double mul = 1.0) const {
    _derived1.template add_to_hess<M>(hess, mul * scalar);
  }

  std::string print() const {
    return _derived1.print() + " * " + std::to_string(scalar);
  }
};

template <typename Derived1, typename GradType = GRAD_TYPE(Derived1),
          typename HessType =
              decltype(std::declval<HESS_TYPE(Derived1)>().rank_update(
                  std::declval<GRAD_TYPE(Derived1)>()))>
class MetaVarChain : public MetaVarBase<MetaVarChain<Derived1>> {
public:
  static constexpr bool forward_grad = Derived1::forward_grad;
  static constexpr bool forward_hessian = Derived1::forward_hessian;

  const Derived1 _derived1;
  double _value;
  double chain_grad;
  double chain_hess;
  typename std::conditional<forward_grad, const GradType,
                            MetaGrad<MetaVec<>>>::type _grad;
  typename std::conditional<forward_hessian, const HessType,
                            MetaSparseMat<>>::type _hess;

  // Forward grad, reverse hessian.
  MetaVarChain(const Derived1 &derived1, double val, double grad, double hess)
    requires(forward_grad && !forward_hessian)
      : _derived1(derived1), _value(val), _grad(derived1._grad * grad),
        chain_grad(grad), chain_hess(hess) {}
  // Forward grad and hessian.
  MetaVarChain(const Derived1 &derived1, double val, double grad, double hess)
    requires(forward_grad && forward_hessian)
      : _derived1(derived1), _value(val), _grad(derived1._grad * grad),
        _hess((derived1._hess * grad).rank_update(derived1._grad, hess)),
        chain_grad(grad), chain_hess(hess) {}
  // Reverse grad.
  MetaVarChain(const Derived1 &derived1, double val, double grad, double hess)
    requires(!forward_grad && !forward_hessian)
      : _derived1(derived1), _value(val), chain_grad(grad), chain_hess(hess) {}

  double val() const { return _value; }

  auto reverse_grad(double mul = 1.0) const {
    return _derived1.reverse_grad(mul * chain_grad);
  }
  template <int M = 0, typename Hess>
  void add_to_hess(Hess &hess, double mul = 1.0) const {
    _derived1.template add_to_hess<M>(hess, mul * chain_grad);
    _derived1._grad.template rank_update<M>(hess, mul * chain_hess);
  }

  std::string print() const {
    return "chain(" + _derived1.print() + ", " + chain_grad + ", " +
           chain_hess + ")";
  }
};

template <typename Derived1, typename Derived2,
          typename GradType = decltype(std::declval<GRAD_TYPE(Derived1)>() +
                                       std::declval<GRAD_TYPE(Derived2)>()),
          typename HessType = decltype(std::declval<HESS_TYPE(Derived1)>() +
                                       std::declval<HESS_TYPE(Derived2)>())>
class MetaVarAdd : public MetaVarBase<MetaVarAdd<Derived1, Derived2>> {
public:
  static constexpr bool forward_grad =
      Derived1::forward_grad && Derived2::forward_grad;
  static constexpr bool forward_hessian =
      Derived1::forward_hessian && Derived2::forward_hessian;
  typename std::conditional<forward_grad, const GradType,
                            MetaGrad<MetaVec<>>>::type _grad;
  typename std::conditional<forward_hessian, const HessType,
                            MetaSparseMat<>>::type _hess;

  const Derived1 _derived1;
  const Derived2 _derived2;
  double _value;

  // Forward grad, reverse hessian.
  MetaVarAdd(const Derived1 &derived1, const Derived2 &derived2)
    requires(forward_grad && !forward_hessian)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() + derived2.val()),
        _grad(derived1._grad + derived2._grad) {}
  // Forward grad and hessian.
  MetaVarAdd(const Derived1 &derived1, const Derived2 &derived2)
    requires(forward_grad && forward_hessian)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() + derived2.val()),
        _grad(derived1._grad + derived2._grad),
        _hess(derived1._hess + derived2._hess) {}
  // Reverse grad.
  MetaVarAdd(const Derived1 &derived1, const Derived2 &derived2)
    requires(!forward_grad && !forward_hessian)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() + derived2.val()) {}

  double val() const { return _value; }

  auto reverse_grad(double mul = 1.0) const {
    return _derived1.reverse_grad(mul) + _derived2.reverse_grad(mul);
  }
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
                                       std::declval<GRAD_TYPE(Derived2)>()),
          typename HessType = decltype(std::declval<HESS_TYPE(Derived1)>() +
                                       std::declval<HESS_TYPE(Derived2)>())>
class MetaVarSub : public MetaVarBase<MetaVarSub<Derived1, Derived2>> {
public:
  static constexpr bool forward_grad =
      Derived1::forward_grad && Derived2::forward_grad;
  static constexpr bool forward_hessian =
      Derived1::forward_hessian && Derived2::forward_hessian;
  typename std::conditional<forward_grad, const GradType,
                            MetaGrad<MetaVec<>>>::type _grad;
  typename std::conditional<forward_hessian, const HessType,
                            MetaSparseMat<>>::type _hess;
  const Derived1 _derived1;
  const Derived2 _derived2;
  double _value;

  // Forward grad, reverse hessian.
  MetaVarSub(const Derived1 &derived1, const Derived2 &derived2)
    requires(forward_grad && !forward_hessian)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() - derived2.val()),
        _grad(derived1._grad - derived2._grad) {}
  // Forward grad and hessian.
  MetaVarSub(const Derived1 &derived1, const Derived2 &derived2)
    requires(forward_grad && forward_hessian)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() - derived2.val()),
        _grad(derived1._grad - derived2._grad),
        _hess(derived1._hess - derived2._hess) {}
  // Reverse grad.
  MetaVarSub(const Derived1 &derived1, const Derived2 &derived2)
    requires(!forward_grad && !forward_hessian)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() - derived2.val()) {}

  double val() const { return _value; }

  auto reverse_grad(double mul = 1.0) const {
    return _derived1.reverse_grad(mul) - _derived2.reverse_grad(mul);
  }
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
                                       std::declval<GRAD_TYPE(Derived2)>()),
          typename HessType =
              decltype((std::declval<HESS_TYPE(Derived1)>() +
                        std::declval<HESS_TYPE(Derived2)>())
                           .rank_update(std::declval<GRAD_TYPE(Derived1)>(),
                                        std::declval<GRAD_TYPE(Derived2)>()))>
class MetaVarMul : public MetaVarBase<MetaVarMul<Derived1, Derived2>> {
public:
  static constexpr bool forward_grad =
      Derived1::forward_grad && Derived2::forward_grad;
  static constexpr bool forward_hessian =
      Derived1::forward_hessian && Derived2::forward_hessian;
  typename std::conditional<forward_grad, const GradType,
                            MetaGrad<MetaVec<>>>::type _grad;
  typename std::conditional<forward_hessian, const HessType,
                            MetaSparseMat<>>::type _hess;
  const Derived1 _derived1;
  const Derived2 _derived2;
  double _value;

  // Forward grad, reverse hessian.
  MetaVarMul(const Derived1 &derived1, const Derived2 &derived2)
    requires(forward_grad && !forward_hessian)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() * derived2.val()),
        _grad(derived1._grad * derived2.val() +
              derived2._grad * derived1.val()) {}
  // Forward grad and hessian.
  MetaVarMul(const Derived1 &derived1, const Derived2 &derived2)
    requires(forward_grad && forward_hessian)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() * derived2.val()),
        _grad(derived1._grad * derived2.val() +
              derived2._grad * derived1.val()),
        _hess(
            (derived1._hess * derived2.val() + derived2._hess * derived1.val())
                .rank_update(derived1._grad, derived2._grad)) {}
  // Reverse grad.
  MetaVarMul(const Derived1 &derived1, const Derived2 &derived2)
    requires(!forward_grad && !forward_hessian)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() * derived2.val()) {}

  double val() const { return _value; }

  auto reverse_grad(double mul = 1.0) const {
    return _derived1.reverse_grad(mul * _derived2.val()) +
           _derived2.reverse_grad(mul * _derived1.val());
  }
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
          typename GradType = decltype(GRAD_VAL(Derived1) + GRAD_VAL(Derived2)),
          typename HessType =
              decltype((HESS_VAL(Derived1) - HESS_VAL(Derived2))
                           .rank_update(std::declval<GradType>(),
                                        GRAD_VAL(Derived2)))>
class MetaVarDiv : public MetaVarBase<MetaVarDiv<Derived1, Derived2>> {
public:
  const Derived1 _derived1;
  const Derived2 _derived2;
  double _value;
  static constexpr bool forward_grad =
      Derived1::forward_grad && Derived2::forward_grad;
  static constexpr bool forward_hessian =
      Derived1::forward_hessian && Derived2::forward_hessian;
  typename std::conditional<forward_grad, const GradType,
                            MetaGrad<MetaVec<>>>::type _grad;
  typename std::conditional<forward_hessian, const HessType,
                            MetaSparseMat<>>::type _hess;

  // Forward grad, reverse hessian.
  MetaVarDiv(const Derived1 &derived1, const Derived2 &derived2)
    requires(forward_grad && !forward_hessian)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() / derived2.val()),
        _grad(derived1._grad * (1.0 / derived2.val()) -
              derived2._grad * (_value / derived2.val())) {}
  // Forward grad and hessian.
  MetaVarDiv(const Derived1 &derived1, const Derived2 &derived2)
    requires(forward_grad && forward_hessian)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() / derived2.val()),
        _grad(derived1._grad * (1.0 / derived2.val()) -
              derived2._grad * (_value / derived2.val())),
        _hess((derived1._hess * (1.0 / derived2.val()) -
               derived2._hess * (_value / derived2.val()))
                  .rank_update(_grad, derived2._grad, -1.0 / derived2.val())) {}
  // Reverse grad.
  MetaVarDiv(const Derived1 &derived1, const Derived2 &derived2)
    requires(!forward_grad && !forward_hessian)
      : _derived1(derived1), _derived2(derived2),
        _value(derived1.val() / derived2.val()) {}

  double val() const { return _value; }

  auto reverse_grad(double mul = 1.0) const {
    return _derived1.reverse_grad(mul / _derived2.val()) -
           _derived2.reverse_grad(mul * _value / _derived2.val());
  }
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

template <InheritsFromMetaBase Derived1, InheritsFromMetaBase Derived2>
MetaVarAdd<Derived1, Derived2> operator+(Derived1 const &derived1,
                                         Derived2 const &derived2) {
  return MetaVarAdd<Derived1, Derived2>(derived1, derived2);
}

template <
    InheritsFromMetaBase Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
auto operator+(Derived1 const &derived1, G scalar) {
  return derived1 + MetaVarScalar(scalar);
}

template <
    InheritsFromMetaBase Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
auto operator+(G scalar, Derived1 const &derived1) {
  return MetaVarScalar(scalar) + derived1;
}

template <InheritsFromMetaBase Derived1, InheritsFromMetaBase Derived2>
auto operator-(Derived1 const &derived1, Derived2 const &derived2) {
  return MetaVarSub<Derived1, Derived2>(
      *static_cast<const Derived1 *>(&derived1),
      *static_cast<const Derived2 *>(&derived2));
}

template <
    InheritsFromMetaBase Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
decltype(auto) operator-(Derived1 const &derived1, G scalar) {
  return derived1 - MetaVarScalar(scalar);
}

template <
    InheritsFromMetaBase Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
decltype(auto) operator-(G scalar, Derived1 const &derived1) {
  return MetaVarScalar(scalar) - derived1;
}

template <InheritsFromMetaBase Derived1, InheritsFromMetaBase Derived2>
auto operator*(Derived1 const &derived1, Derived2 const &derived2) {
  return MetaVarMul<Derived1, Derived2>(derived1, derived2);
}

template <
    InheritsFromMetaBase Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
MetaVarScalarMul<Derived1> operator*(Derived1 const &derived1,
                                     const G &scalar) {
  return MetaVarScalarMul<Derived1>(derived1, scalar);
}

template <
    InheritsFromMetaBase Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
auto operator*(const G &scalar, Derived1 const &derived1) {
  return MetaVarScalarMul<Derived1>(derived1, scalar);
}

template <
    InheritsFromMetaBase Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
MetaVarScalarMul<Derived1> operator/(Derived1 const &derived1,
                                     const G &scalar) {
  return MetaVarScalarMul<Derived1>(derived1, 1.0 / scalar);
}

template <
    InheritsFromMetaBase Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
auto operator/(const G &scalar, Derived1 const &derived1) {
  return MetaVarDiv<MetaVarScalar, Derived1>(MetaVarScalar(scalar), derived1);
}

template <InheritsFromMetaBase Derived1, InheritsFromMetaBase Derived2>
auto operator/(Derived1 const &derived1, Derived2 const &derived2) {
  return MetaVarDiv<Derived1, Derived2>(derived1, derived2);
}

template <InheritsFromMetaBase Derived1> auto sin(Derived1 const &derived1) {
  double sin_val = std::sin(derived1.val());
  return MetaVarChain<Derived1>(derived1, sin_val, std::cos(derived1.val()),
                                -sin_val);
}

template <InheritsFromMetaBase Derived1> auto cos(Derived1 const &derived1) {
  double cos_val = std::cos(derived1.val());
  return MetaVarChain<Derived1>(derived1, cos_val, -std::sin(derived1.val()),
                                -cos_val);
}

template <InheritsFromMetaBase Derived1> auto exp(Derived1 const &derived1) {
  double val = std::exp(derived1.val());
  return MetaVarChain<Derived1>(derived1, val, val, val);
}

template <InheritsFromMetaBase Derived1> auto log(Derived1 const &derived1) {
  return MetaVarChain<Derived1>(derived1, std::log(derived1.val()),
                                1.0 / derived1.val(),
                                -1.0 / (derived1.val() * derived1.val()));
}

template <
    InheritsFromMetaBase Derived1, typename G,
    typename = typename std::enable_if<std::is_arithmetic<G>::value, G>::type>
auto pow(Derived1 const &derived1, const G &exponent) {
  double f2 = std::pow(derived1.val(), exponent - 2);
  double f1 = f2 * derived1.val();
  double f = f1 * derived1.val();
  return MetaVarChain<Derived1, G>(derived1, f, exponent * f1,
                                   exponent * (exponent - 1) * f2);
}

template <InheritsFromMetaBase Derived1> auto sqr(Derived1 const &derived1) {
  return MetaVarChain<Derived1>(derived1, derived1.val() * derived1.val(),
                                2 * derived1.val(), 2);
}

template <InheritsFromMetaBase Derived1>
auto operator-(Derived1 const &derived1) {
  return MetaVarChain<Derived1>(derived1, -derived1.val(), -1, 0);
}

} // namespace Optiz