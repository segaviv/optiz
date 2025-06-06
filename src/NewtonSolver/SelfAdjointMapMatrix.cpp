#include "SelfAdjointMapMatrix.h"

#include <Common/Utils.h>
#include <vector>

namespace Optiz {

Optiz::SelfAdjointMapMatrix::SelfAdjointMapMatrix(long n) : _n(n) {}

SelfAdjointMapMatrix::SelfAdjointMapMatrix(SelfAdjointMapMatrix &&) noexcept =
    default;
double &SelfAdjointMapMatrix::operator()(long i, long j) {
  auto [col, row] = minmax(i, j);
  return values[CellIndex{row, col}];
}

double SelfAdjointMapMatrix::operator()(long i, long j) const {
  auto [col, row] = minmax(i, j);
  return values[CellIndex{row, col}];
}

SelfAdjointMapMatrix &
SelfAdjointMapMatrix::operator=(SelfAdjointMapMatrix &&other) = default;

SelfAdjointMapMatrix &
SelfAdjointMapMatrix::operator+=(const SelfAdjointMapMatrix &other) {
  other.for_each([&](auto &val) { values[val.first] += val.second; });
  return *this;
}
SelfAdjointMapMatrix &
SelfAdjointMapMatrix::operator-=(const SelfAdjointMapMatrix &other) {
  other.for_each([&](auto &val) { values[val.first] -= val.second; });
  return *this;
}
SelfAdjointMapMatrix &Optiz::SelfAdjointMapMatrix::operator*=(double scalar) {
  for_each([&](auto &val) { val.second *= scalar; });
  return *this;
}

SelfAdjointMapMatrix &Optiz::SelfAdjointMapMatrix::operator/=(double scalar) {
  for_each([&](auto &val) { val.second /= scalar; });
  return *this;
}

SelfAdjointMapMatrix &
SelfAdjointMapMatrix::add(const SelfAdjointMapMatrix &other, double alpha) {
  other.for_each([&](auto &val) { values[val.first] += alpha * val.second; });
  return *this;
}

SelfAdjointMapMatrix &SelfAdjointMapMatrix::rank_update(const SparseVector &u,
                                                        const SparseVector &v) {
  for (const auto &u_val : u.get_values()) {
    for (const auto &v_val : v.get_values()) {
      double val = u_val.second * v_val.second;
      if (u_val.first == v_val.first)
        val *= 2;
      operator()(u_val.first, v_val.first) += val;
    }
  }
  return *this;
}
SelfAdjointMapMatrix &SelfAdjointMapMatrix::rank_update(const SparseVector &u,
                                                        const SparseVector &v,
                                                        double alpha) {
  for (const auto &u_val : u.get_values()) {
    for (const auto &v_val : v.get_values()) {
      double val = alpha * u_val.second * v_val.second;
      if (u_val.first == v_val.first)
        val *= 2;
      operator()(u_val.first, v_val.first) += val;
    }
  }
  return *this;
}
SelfAdjointMapMatrix &SelfAdjointMapMatrix::rank_update(const SparseVector &u,
                                                        double alpha) {
  for (const auto &u_val : u.get_values()) {
    for (const auto &v_val : u.get_values()) {
      if (v_val.first > u_val.first) {
        continue;
      }
      values[{u_val.first, v_val.first}] += alpha * u_val.second * v_val.second;
    }
  }
  return *this;
}

SelfAdjointMapMatrix::operator std::vector<Eigen::Triplet<double>>() const {
  std::vector<Eigen::Triplet<double>> triplets;
  for (const auto &val : values) {
    triplets.push_back(
        Eigen::Triplet<double>(val.first.row, val.first.col, val.second));
  }
  return triplets;
}

Eigen::MatrixXd SelfAdjointMapMatrix::to_dense() const {
  int s = n();
  Eigen::MatrixXd res(s, s);
  for (const auto &val : values) {
    res(val.first.row, val.first.col) = val.second;
    res(val.first.col, val.first.row) = val.second;
  }
  return res;
}

Optiz::SelfAdjointMapMatrix::operator Eigen::SparseMatrix<double>() const {
  std::vector<Eigen::Triplet<double>> triplets;
  for (const auto &val : values) {
    triplets.push_back(
        Eigen::Triplet<double>(val.first.row, val.first.col, val.second));
  }
  int s = n();
  Eigen::SparseMatrix<double> result(s, s);
  result.setFromTriplets(triplets.begin(), triplets.end());
  return result;
}
std::ostream &operator<<(std::ostream &s, const SelfAdjointMapMatrix &mat) {
  int size = mat.n();
  Eigen::MatrixXd res(size, size);
  for (const auto &val : mat.values) {
    long row = val.first.row, col = val.first.col;
    res(row, col) = val.second;
    res(col, row) = val.second;
  }
  s << res;
  return s;
}

} // namespace Optiz
