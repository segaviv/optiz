#pragma once

#include <Eigen/Eigen>
#include <complex>
#include <iostream>
#include <string>

namespace Optiz {

template <typename Derived, typename T> class LinearExpressionBase {
public:
  static constexpr bool is_leaf = false;
  void append_to(int row, std::vector<Eigen::Triplet<T>> &vec) const {
    static_cast<const Derived *>(this)->impl(row, vec);
  }
  std::string print() const {
    return static_cast<const Derived *>(this)->print();
  }

  friend std::ostream &operator<<(std::ostream &s,
                                  LinearExpressionBase const &expr) {
    return s << "LinearExpression(" << expr.print() << ")";
  }
};

template <typename T>
class LinearExpressionVariable
    : public LinearExpressionBase<LinearExpressionVariable<T>, T> {
public:
  static constexpr bool is_leaf = true;
  int _index;

  LinearExpressionVariable(int index) : _index(index) {}
  void impl(int row, std::vector<Eigen::Triplet<T>> &vec) const {
    vec.emplace_back(row, _index, 1);
  }
  void impl(int row, const T &coef, std::vector<Eigen::Triplet<T>> &vec) const {
    vec.emplace_back(row, _index, coef);
  }
  std::string to_str(const T &val) const {
    std::ostringstream out(std::ios::fixed);
    out.precision(2);
    out << val;
    return out.str();
  }

  std::string print(const T &coef = T(1.0)) const {
    return (coef != 1.0 ? (to_str(coef) + " * ") : std::string()) + "x_" +
           std::to_string(_index);
  }
};

template <typename Derived1, typename Derived2, typename T>
class LinearExpressionAdd
    : public LinearExpressionBase<LinearExpressionAdd<Derived1, Derived2, T>,
                                  T> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  typename std::conditional<Derived2::is_leaf, const Derived2 &,
                            const Derived2>::type _derived2;

  LinearExpressionAdd(const Derived1 &derived1, const Derived2 &derived2)
      : _derived1(derived1), _derived2(derived2) {}
  void impl(int row, std::vector<Eigen::Triplet<T>> &vec) const {
    _derived1.impl(row, vec);
    _derived2.impl(row, vec);
  }
  void impl(int row, const T &coef, std::vector<Eigen::Triplet<T>> &vec) const {
    _derived1.impl(row, coef, vec);
    _derived2.impl(row, coef, vec);
  }
  std::string print(const T &coef = T(1.0)) const {
    return _derived1.print(coef) + " + " + _derived2.print(coef);
  }
};

template <typename Derived1, typename Derived2, typename T>
class LinearExpressionSub
    : public LinearExpressionBase<LinearExpressionSub<Derived1, Derived2, T>,
                                  T> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived1::is_leaf, const Derived1 &,
                            const Derived1>::type _derived1;
  typename std::conditional<Derived2::is_leaf, const Derived2 &,
                            const Derived2>::type _derived2;

  LinearExpressionSub(const Derived1 &derived1, const Derived2 &derived2)
      : _derived1(derived1), _derived2(derived2) {}
  void impl(int row, std::vector<Eigen::Triplet<T>> &vec) const {
    _derived1.impl(row, vec);
    _derived2.impl(row, -1, vec);
  }
  void impl(int row, const T &coef, std::vector<Eigen::Triplet<T>> &vec) const {
    _derived1.impl(row, coef, vec);
    _derived2.impl(row, -coef, vec);
  }
  std::string print(const T &coef = T(1.0)) const {
    return _derived1.print(coef) + " + " + _derived2.print(-coef);
  }
};

template <typename Derived, typename T>
class LinearExpressionScalarMul
    : public LinearExpressionBase<LinearExpressionScalarMul<Derived, T>, T> {
public:
  static constexpr bool is_leaf = false;
  typename std::conditional<Derived::is_leaf, const Derived &,
                            const Derived>::type _derived;
  T _scalar;
  LinearExpressionScalarMul(T scalar, const Derived &derived)
      : _scalar(scalar), _derived(derived) {}
  void impl(int row, std::vector<Eigen::Triplet<T>> &vec) const {
    _derived.impl(row, _scalar, vec);
  }
  void impl(int row, const T &coef, std::vector<Eigen::Triplet<T>> &vec) const {
    _derived.impl(row, coef * _scalar, vec);
  }
  std::string print(const T &coef = T(1.0)) const {
    return _derived.print(coef * _scalar);
  }
};

template <typename Derived1, typename Derived2, typename T>
LinearExpressionAdd<Derived1, Derived2, T>
operator+(LinearExpressionBase<Derived1, T> const &derived1,
          LinearExpressionBase<Derived2, T> const &derived2) {
  return LinearExpressionAdd<Derived1, Derived2, T>(
      *static_cast<const Derived1 *>(&derived1),
      *static_cast<const Derived2 *>(&derived2));
}

template <typename Derived1, typename Derived2, typename T>
LinearExpressionSub<Derived1, Derived2, T>
operator-(LinearExpressionBase<Derived1, T> const &derived1,
          LinearExpressionBase<Derived2, T> const &derived2) {
  return LinearExpressionSub<Derived1, Derived2, T>(
      *static_cast<const Derived1 *>(&derived1),
      *static_cast<const Derived2 *>(&derived2));
}

template <typename Derived, typename T, typename G>
LinearExpressionScalarMul<Derived, T>
operator*(const G &scalar, LinearExpressionBase<Derived, T> const &derived) {
  return LinearExpressionScalarMul<Derived, T>(
      scalar, *static_cast<const Derived *>(&derived));
}

template <typename Derived, typename T, typename G>
LinearExpressionScalarMul<Derived, T>
operator*(LinearExpressionBase<Derived, T> const &derived, const G &scalar) {
  return LinearExpressionScalarMul<Derived, T>(
      scalar, *static_cast<const Derived *>(&derived));
}

using LinearExpressionVariableD = LinearExpressionVariable<double>;
using LinearExpressionVariableC =
    LinearExpressionVariable<std::complex<double>>;

} // namespace Optiz
