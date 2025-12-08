#pragma once
#include <Eigen/Eigen>
#include <iostream>
#include <tuple>

#include "SparseVector.h"

namespace Optiz {

class VarGrad {
public:
  VarGrad();
  explicit VarGrad(double val);
  VarGrad(const double val, int index);
  VarGrad(const VarGrad &);
  VarGrad(VarGrad &&) noexcept;
  VarGrad(double var, const SparseVector &grad)
      : _val(var), _grad(grad) {}

  VarGrad &operator=(const VarGrad &other) = default;
  VarGrad &operator=(VarGrad &&);
  VarGrad &operator=(double val);

  template <int Rows, int Cols>
  static Eigen::Matrix<VarGrad, Rows, Cols>
  from_eigen(const Eigen::Matrix<double, Rows, Cols> &mat, int start_index = 0) {
    return Eigen::Matrix<VarGrad, Rows, Cols>::NullaryExpr(
        mat.rows(), mat.cols(), [&mat, start_index](Eigen::Index i, Eigen::Index j) {
          return VarGrad(mat(i, j), static_cast<int>(start_index + i + j * mat.rows()));
        });
  }

  // Getters.
  double val() const;
  inline double &val() { return _val; }
  inline SparseVector &grad() { return _grad; }
  const SparseVector &grad() const;
  Eigen::VectorXd dense_grad() const;
  using Tup =
      std::tuple<double, Eigen::VectorXd, std::vector<Eigen::Triplet<double>>>;
  operator Tup() const;

  int n_vargrads() const;

  friend std::ostream &operator<<(std::ostream &s, const VarGrad &vargrad);
  std::string referenced_str() const;

  // Operators.
  VarGrad &operator*=(const VarGrad &b);
  VarGrad &operator*=(double b);
  VarGrad &operator/=(const VarGrad &b);
  VarGrad &operator/=(double b);
  VarGrad &operator+=(const VarGrad &b);
  VarGrad &operator+=(double b);
  VarGrad &operator-=(const VarGrad &b);
  VarGrad &operator-=(double b);
  VarGrad &chain_this(double val, double grad);
  VarGrad chain(double val, double grad) const;

  VarGrad inv() const;
  VarGrad &inv_self();
  VarGrad &neg();

  // Mul operator between two vargradiables.
  friend VarGrad operator*(const VarGrad &a, const VarGrad &b);
  friend VarGrad operator*(VarGrad &&a, const VarGrad &b);
  friend VarGrad operator*(const VarGrad &a, VarGrad &&b);
  friend VarGrad operator*(VarGrad &&a, VarGrad &&b);
  friend VarGrad operator*(double b, const VarGrad &a);
  friend VarGrad operator*(const VarGrad &a, double b);
  friend VarGrad operator*(double b, VarGrad &&a);
  friend VarGrad operator*(VarGrad &&a, double b);

  // Div operator between two vargradiables.
  friend VarGrad operator/(const VarGrad &a, const VarGrad &b);
  friend VarGrad operator/(VarGrad &&a, const VarGrad &b);
  friend VarGrad operator/(const VarGrad &a, VarGrad &&b);
  friend VarGrad operator/(VarGrad &&a, VarGrad &&b);
  friend VarGrad operator/(double b, const VarGrad &a);
  friend VarGrad operator/(const VarGrad &a, double b);
  friend VarGrad operator/(double b, VarGrad &&a);
  friend VarGrad operator/(VarGrad &&a, double b);

  // Add operator between two vargradiables.
  friend VarGrad operator+(const VarGrad &a, const VarGrad &b);
  friend VarGrad operator+(VarGrad &&a, const VarGrad &b);
  friend VarGrad operator+(const VarGrad &a, VarGrad &&b);
  friend VarGrad operator+(VarGrad &&a, VarGrad &&b);
  // Add operator between VarGrad and double
  friend VarGrad operator+(double b, const VarGrad &a);
  friend VarGrad operator+(double b, VarGrad &&a);
  friend VarGrad operator+(const VarGrad &a, double b);
  friend VarGrad operator+(VarGrad &&a, double b);

  // Sub operator between two vargradiables.
  friend VarGrad operator-(const VarGrad &a, const VarGrad &b);
  friend VarGrad operator-(VarGrad &&a, const VarGrad &b);
  friend VarGrad operator-(const VarGrad &a, VarGrad &&b);
  friend VarGrad operator-(VarGrad &&a, VarGrad &&b);
  // Sub operator between VarGrad and double
  friend VarGrad operator-(double b, const VarGrad &a);
  friend VarGrad operator-(const VarGrad &a, double b);
  friend VarGrad operator-(double b, VarGrad &&a);
  friend VarGrad operator-(VarGrad &&a, double b);

  VarGrad operator^(const VarGrad &p) { return pow(*this, p.val()); }
  friend VarGrad operator-(const VarGrad &a);
  friend VarGrad operator-(VarGrad &&a);
  friend VarGrad sqrt(const VarGrad &a);
  friend VarGrad sqrt(VarGrad &&a);
  friend VarGrad abs(const VarGrad &a);
  friend VarGrad abs(VarGrad &&a);
  friend VarGrad pow(const VarGrad &a, const double exponent);
  friend VarGrad pow(const VarGrad &a, const int exponent);
  friend VarGrad pow(VarGrad &&a, const int exponent);
  friend VarGrad pow(VarGrad &&a, const double exponent);
  friend VarGrad exp(const VarGrad &a);
  friend VarGrad exp(VarGrad &&a);
  friend VarGrad log(const VarGrad &a);
  friend VarGrad log(VarGrad &&a);
  friend VarGrad cos(const VarGrad &a);
  friend VarGrad cos(VarGrad &&a);
  friend VarGrad sin(const VarGrad &a);
  friend VarGrad sin(VarGrad &&a);
  friend VarGrad atan(const VarGrad &x);
  friend VarGrad atan2(const VarGrad &y, const VarGrad &x);
  friend VarGrad chain2(const VarGrad &x, const VarGrad &y, double val, double dx,
                    double dy, double dxdx, double dxdy, double dydy);
  friend bool isfinite(const VarGrad &x);
  friend bool isinf(const VarGrad &x);

  // ----------------------- Comparisons -----------------------
  friend bool operator<(const VarGrad &a, const VarGrad &b);
  friend bool operator<=(const VarGrad &a, const VarGrad &b);
  friend bool operator>(const VarGrad &a, const VarGrad &b);
  friend bool operator>=(const VarGrad &a, const VarGrad &b);
  friend bool operator==(const VarGrad &a, const VarGrad &b);
  friend bool operator!=(const VarGrad &a, const VarGrad &b);

public:
  double _val;
  SparseVector _grad;
};

VarGrad sqr(const VarGrad &a);
VarGrad sqr(VarGrad &&a);

} // namespace Optiz

#pragma omp declare reduction(+ : Optiz::VarGrad : omp_out += omp_in)

namespace Eigen {

/**
 * See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
 * and https://eigen.tuxfamily.org/dox/structEigen_1_1NumTraits.html
 */
template <> struct NumTraits<Optiz::VarGrad> : NumTraits<double> {
  typedef Optiz::VarGrad Real;
  typedef Optiz::VarGrad NonInteger;
  typedef Optiz::VarGrad Nested;

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
struct ScalarBinaryOpTraits<Optiz::VarGrad, double, BinaryOp> {
  typedef Optiz::VarGrad ReturnType;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, Optiz::VarGrad, BinaryOp> {
  typedef Optiz::VarGrad ReturnType;
};

} // namespace Eigen