#pragma once
#include "MetaIntMap.h"

namespace Optiz {

template <typename MapType = MetaIntMap<>> struct MetaSparseMat {
  MapType data;
  MetaSparseMat(const MapType &data = MapType()) : data(data) {}

  template <int I> auto row() const {
    return data.template get<I, MetaIntMap<>>();
  }
  template <int I, int J, typename T> auto get() const {
    return data.template get<I, MetaIntMap<>>().template get<J, T>();
  }

  template <int I, int J, typename V> auto set(const V &value) const {
    auto new_row = row<I>().template set<J>(value);
    auto new_data = data.template set<I>(new_row);
    return MetaSparseMat<TYPE(new_data)>(new_data);
  }

  template <int I = 0, int J = 0, typename Grad>
  auto rank_update(const Grad &grad, double mul = 1.0) const {
    if constexpr (I == decltype(grad.entries)::Size) {
      return *this;
    } else if constexpr (J == decltype(grad.entries)::Size) {
      return rank_update<I + 1, I + 1, Grad>(grad, mul);
    } else {
      auto entry_i = grad.template get<I>();
      auto entry_j = grad.template get<J>();
      constexpr int i = TYPE(entry_i)::Index;
      constexpr int j = TYPE(entry_j)::Index;
      double val = (entry_i.val * entry_j.val) * mul;
      if constexpr (i > j) {
        auto new_mat = set<i, j>(get<i, j, double>() + val);
        return new_mat.template rank_update<I, J + 1, Grad>(grad, mul);
      } else {
        auto new_mat = set<j, i>(get<j, i, double>() + val);
        return new_mat.template rank_update<I, J + 1, Grad>(grad, mul);
      }
    }
  }

  template <int I = 0, int J = 0, typename Grad, typename Grad2>
  auto rank_update(const Grad &grad, const Grad2 &grad2,
                   double mul = 1.0) const {
    if constexpr (I == decltype(grad.entries)::Size) {
      return *this;
    } else if constexpr (J == decltype(grad2.entries)::Size) {
      return rank_update<I + 1, 0>(grad, grad2, mul);
    } else {
      auto entry_i = grad.template get<I>();
      auto entry_j = grad2.template get<J>();
      constexpr int i = TYPE(entry_i)::Index;
      constexpr int j = TYPE(entry_j)::Index;
      double val = (entry_i.val * entry_j.val) * mul;
      if constexpr (i == j)
        val *= 2.0;
      if constexpr (i > j) {
        auto new_mat = set<i, j>(get<i, j, double>() + val);
        return new_mat.template rank_update<I, J + 1>(grad, grad2, mul);
      } else {
        auto new_mat = set<j, i>(get<j, i, double>() + val);
        return new_mat.template rank_update<I, J + 1>(grad, grad2, mul);
      }
    }
  }

  template <typename Func, int I = 0> auto map(const Func &f) const {
    auto new_data = data.map([&](const auto &entry) {
      return entry.val.map([&](const auto &entry2) { return f(entry2); });
    });
    return MetaSparseMat<TYPE(new_data)>(new_data);
  }

  auto operator*(double scalar) const {
    return map([&](const auto &entry) { return entry.val * scalar; });
  }

  template <typename OtherMat, int I = 0>
  auto operator+(const OtherMat &otherMat) const {
    if constexpr (I == TYPE(otherMat.data.entries)::Size) {
      return *this;
    } else {
      constexpr int row_index =
          TYPE(otherMat.data.entries.template get<I>())::Index;
      auto entry = otherMat.data.entries.template get<I>();
      auto new_row = this->template row<row_index>() + entry.val;
      auto new_data = data.template set<row_index>(new_row);
      auto new_mat = MetaSparseMat<TYPE(new_data)>(new_data);
      return new_mat.template operator+ <OtherMat, I + 1>(otherMat);
    }
  }
  template <typename OtherMat, int I = 0>
  auto operator-(const OtherMat &otherMat) const {
    if constexpr (I == TYPE(otherMat.data.entries)::Size) {
      return *this;
    } else {
      constexpr int row_index =
          TYPE(otherMat.data.entries.template get<I>())::Index;
      auto entry = otherMat.data.entries.template get<I>();
      auto new_row = this->template row<row_index>() - entry.val;
      auto new_data = data.template set<row_index>(new_row);
      auto new_mat = MetaSparseMat<TYPE(new_data)>(new_data);
      return new_mat.template operator- <OtherMat, I + 1>(otherMat);
    }
  }

  template <int N, int M> Eigen::Matrix<double, N, M> to_eigen() const {
    Eigen::Matrix<double, N, M> mat = Eigen::Matrix<double, N, M>::Zero();
    For<0, N>([&]<int I>() {
      For<0, M>([&]<int J>() { mat(I, J) = get<I, J, double>(); });
    });
    return mat;
  }

  template <int I = 0>
  friend std::ostream &operator<<(std::ostream &s, MetaSparseMat const &expr) {
    if constexpr (I == TYPE(expr.data.entries)::Size) {
      return s;
    } else {
      constexpr int index = TYPE(expr.data.entries.template get<I>())::Index;
      s << "{" << index << ": " << expr.data.entries.template get<I>().val << "}"
        << std::endl;
      return operator<<<I + 1>(s, expr);
    }
  }
};

} // namespace Optiz