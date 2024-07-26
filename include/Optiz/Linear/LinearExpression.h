#pragma once
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif
#include "../NewtonSolver/VectorMap.h"
#include <Eigen/Eigen>
#include <iostream>

namespace Optiz {

#define LINEAR_EXPRESSIONS_T(type) Eigen::VectorX<Optiz::LinearExpression<type>>

template <typename T> class LinearExpression {
public:
  using VecType = Eigen::VectorX<T>;
  LinearExpression() = default;
  LinearExpression(const LinearExpression &other) = default;
  LinearExpression(LinearExpression &&other) = default;

  LinearExpression &operator=(const LinearExpression &other) = default;

  LinearExpression(const T &val) : rhs(VecType::Constant(1, val)) {};
  LinearExpression(const VecType &rhs) : rhs(rhs) {}
  LinearExpression(int var_index, const T &value) {
    _values[var_index] = value;
  }

  VecType pad_add(const VecType &first, const VecType &second) {
    if (first.size() == 0)
      return second;
    if (second.size() == 0)
      return first;
    if (first.size() < second.size()) {
      VecType res = second;
      res.head(first.size()) += first;
      return res;
    } else if (first.size() > second.size()) {
      VecType res = first;
      res.head(second.size()) += second;
      return res;
    }
    return first + second;
  }

  LinearExpression &operator+=(const VecType &other) {
    rhs = pad_add(rhs, other);
    return *this;
  }
  LinearExpression &operator-=(const VecType &other) {
    rhs = pad_add(rhs, -other);
    return *this;
  }

  LinearExpression &operator+=(const LinearExpression &other) {
    for (const auto &val : other._values) {
      _values[val.first] += val.second;
    }
    *this += other.rhs;
    return *this;
  }
  LinearExpression &operator-=(const LinearExpression &other) {
    for (const auto &val : other._values) {
      _values[val.first] -= val.second;
    }
    *this -= other.rhs;
    return *this;
  }
  LinearExpression &operator*=(const T &scalar) {
    for (auto &val : _values) {
      val.second *= scalar;
    }
    rhs *= scalar;
    return *this;
  }
  // Bug in Eigen... Sparse-Dense product converts double scalars to
  // LinearExpression. LinearExpression operator*(double scalar) const;
  LinearExpression operator*(const LinearExpression &other) const {
    if (_values.size() == 0) {
      return LinearExpression(other) *= rhs(0);
    }
    LinearExpression res(*this);
    for (auto &val : res._values) {
      val.second *= other.rhs(0);
    }
    return res;
  }

  operator Eigen::VectorX<LinearExpression>() const {
    Eigen::Matrix<LinearExpression, 1, 1> res(*this);
    return res;
  }

  // Addition.
  friend LinearExpression operator+(const LinearExpression &lhs,
                                    const LinearExpression &rhs) {
    return LinearExpression(lhs) += rhs;
  }
  friend LinearExpression operator+(const LinearExpression &lhs,
                                    const VecType &rhs) {
    return LinearExpression(lhs) += rhs;
  }
  friend LinearExpression operator+(const VecType &lhs,
                                    const LinearExpression &rhs) {
    return LinearExpression(rhs) += lhs;
  }
  friend LinearExpression operator+(const LinearExpression &lhs, const T &rhs) {
    return LinearExpression(lhs) += VecType::Constant(1, rhs);
  }
  friend LinearExpression operator+(const T &lhs, const LinearExpression &rhs) {
    return LinearExpression(rhs) += VecType::Constant(1, lhs);
  }

  // Subtraction.
  friend LinearExpression operator-(const LinearExpression &lhs,
                                    const LinearExpression &rhs) {
    return LinearExpression(lhs) -= rhs;
  }
  friend LinearExpression operator-(const LinearExpression &lhs,
                                    const VecType &rhs) {
    return LinearExpression(lhs) -= rhs;
  }
  friend LinearExpression operator-(const VecType &lhs,
                                    const LinearExpression &rhs) {
    return LinearExpression(lhs) -= rhs;
  }
  friend LinearExpression operator-(const LinearExpression &lhs, const T &rhs) {
    return LinearExpression(lhs) -= VecType::Constant(1, rhs);
  }
  friend LinearExpression operator-(const T &lhs, const LinearExpression &rhs) {
    return LinearExpression(lhs) -= rhs;
  }

  friend LinearExpression operator-(const LinearExpression &lce) {
    return LinearExpression(lce) *= -1;
  }

  friend LinearExpression operator*(const T &scalar,
                                    const LinearExpression &rhs) {
    return LinearExpression(rhs) *= scalar;
  }
  friend LinearExpression operator*(const LinearExpression &lhs,
                                    const T &scalar) {
    return LinearExpression(lhs) *= scalar;
  }

  friend LinearExpression operator/(const LinearExpression &lhs,
                                    const T &scalar) {
    return LinearExpression(lhs) *= 1.0 / scalar;
  }

  friend Eigen::VectorX<LinearExpression>
  operator-(const Eigen::VectorX<LinearExpression> &lhs,
            const Eigen::MatrixX<T> &rhs) {
    Eigen::VectorX<LinearExpression> res(lhs);
    for (int i = 0; i < lhs.rows(); i++) {
      res(i) -= rhs.row(i).reshaped();
    }
    return res;
  }

  friend std::ostream &operator<<(std::ostream &s,
                                  const LinearExpression &expr) {
    s << "LinearExpression(";
    bool first = true;
    for (const auto &val : expr._values) {
      if (first) {
        first = false;
      } else {
        s << " + ";
      }
      s << val.second << " * x_" << val.first;
    }
    if (expr.rhs.size() > 0) {
      if (!first) {
        s << " + ";
      }
      s << expr.rhs.transpose();
    }
    s << ")";
    return s;
  }

  inline const VectorMap<int, T> &values() const { return _values; }

  inline const VecType &rhs_vector() const { return rhs; }

private:
  VectorMap<int, T> _values;
  VecType rhs;
};
} // namespace Optiz

namespace Eigen {

/**
 * See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
 * and https://eigen.tuxfamily.org/dox/structEigen_1_1NumTraits.html
 */
template <typename T>
struct NumTraits<Optiz::LinearExpression<T>> : NumTraits<double> {
  typedef Optiz::LinearExpression<T> Real;
  typedef Optiz::LinearExpression<T> NonInteger;
  typedef Optiz::LinearExpression<T> Nested;

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

template <typename BinaryOp, typename T>
struct ScalarBinaryOpTraits<Optiz::LinearExpression<T>, double, BinaryOp> {
  typedef Optiz::LinearExpression<T> ReturnType;
};

template <typename BinaryOp, typename T>
struct ScalarBinaryOpTraits<double, Optiz::LinearExpression<T>, BinaryOp> {
  typedef Optiz::LinearExpression<T> ReturnType;
};

} // namespace Eigen