#pragma once
#include "SparseVector.h"
#include "VectorMap.h"
#include <Eigen/Eigen>
#include <vector>

namespace Optiz {

class SelfAdjointMapMatrix {
public:
  struct CellIndex {
    long row;
    long col;

    bool operator==(const CellIndex &other) const = default;
  };

  SelfAdjointMapMatrix(long n = -1);
  SelfAdjointMapMatrix(SelfAdjointMapMatrix &&) noexcept;
  SelfAdjointMapMatrix(const SelfAdjointMapMatrix &) = default;

  friend std::ostream &operator<<(std::ostream &s,
                                  const SelfAdjointMapMatrix &var);

  double &operator()(long i, long j);
  double &insert(long i, long j);

  SelfAdjointMapMatrix &operator=(const SelfAdjointMapMatrix &) = default;
  SelfAdjointMapMatrix &operator=(SelfAdjointMapMatrix &&);

  SelfAdjointMapMatrix &operator+=(const SelfAdjointMapMatrix &other);
  SelfAdjointMapMatrix &operator-=(const SelfAdjointMapMatrix &other);
  SelfAdjointMapMatrix &operator*=(double scalar);
  SelfAdjointMapMatrix &operator/=(double scalar);

  SelfAdjointMapMatrix &add(const SelfAdjointMapMatrix &u, double alpha = 1.0);
  SelfAdjointMapMatrix &rank_update(const SparseVector &u,
                                    const SparseVector &v);
  SelfAdjointMapMatrix &rank_update(const SparseVector &u,
                                    const SparseVector &v, double alpha);
  SelfAdjointMapMatrix &rank_update(const SparseVector &u, double alpha = 1.0);

  operator std::vector<Eigen::Triplet<double>>() const;
  operator Eigen::SparseMatrix<double>() const;
  Eigen::MatrixXd to_dense() const;

  const inline VectorMap<CellIndex, double> &get_values() const {
    return values;
  }

  inline VectorMap<CellIndex, double>::iterator begin() {
    return values.begin();
  }
  inline VectorMap<CellIndex, double>::iterator end() { return values.end(); }
  inline VectorMap<CellIndex, double>::const_iterator begin() const {
    return values.begin();
  }
  inline VectorMap<CellIndex, double>::const_iterator end() const {
    return values.end();
  }

  inline long n() const {
    if (_n >= 0)
      return _n;
    if (values.size() == 0)
      return 0;
    
    return 1 + values.max(
                   [](auto &p) { return std::max(p.first.row, p.first.col); });
  }

  inline long rows() const { return n(); }
  inline long cols() const { return n(); }

private:
  long _n;
  VectorMap<CellIndex, double> values;
};
} // namespace Optiz