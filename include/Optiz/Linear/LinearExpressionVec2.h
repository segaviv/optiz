#pragma once
#include "LinearExpressionBase.h"
#include <Eigen/Eigen>

namespace Optiz {

template <typename Derived1, typename Derived2> class LinearExpressionVec2 {
public:
  const Derived1 _derived1;
  const Derived2 _derived2;
  LinearExpressionVec2(const Derived1 &derived1, const Derived2 &derived2)
      : _derived1(derived1), _derived2(derived2) {}

  const Derived1 &x() const { return _derived1; }
  const Derived2 &y() const { return _derived2; }

  decltype(auto) dot(const Eigen::Vector2d &other) const {
    return _derived1 * other.x() + _derived2 * other.y();
  }
};

template <typename Derived1, typename Derived2, typename G>
decltype(auto) operator*(const G &scalar,
                         const LinearExpressionVec2<Derived1, Derived2> &vec) {
  return LinearExpressionVec2(scalar * vec.x(), scalar * vec.y());
}

template <typename Derived1, typename Derived2, typename G>
decltype(auto) operator*(const LinearExpressionVec2<Derived1, Derived2> &vec,
                         const G &scalar) {
  return LinearExpressionVec2(scalar * vec.x(), scalar * vec.y());
}

template <typename Derived1, typename Derived2, typename Derived3,
          typename Derived4>
decltype(auto) operator+(const LinearExpressionVec2<Derived1, Derived2> &vec1,
                         const LinearExpressionVec2<Derived3, Derived4> &vec2) {
  return LinearExpressionVec2(vec1.x() + vec2.x(), vec1.y() + vec2.y());
}

using VariablesVec2 =
    LinearExpressionVec2<LinearExpressionVariableD, LinearExpressionVariableD>;

} // namespace Optiz