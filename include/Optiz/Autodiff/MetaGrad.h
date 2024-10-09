#pragma once
#include "MetaUtils.h"
#include "MetaVec.h"
#include <iostream>

namespace Optiz {

template <int Var> struct GradEntry {
  enum CompileTimeStuff { Index = Var };
  double val;
  GradEntry(double val) : val(val) {}
  friend std::ostream &operator<<(std::ostream &s, GradEntry const &expr) {
    return s << Var << ": " << expr.val;
  }
  GradEntry operator*(double b) const { return GradEntry(val * b); }
};

template <typename Vec = MetaVec<>> struct MetaGrad {
  Vec entries;

  MetaGrad(const Vec &vec = MetaVec()) : entries(vec) {}

  template <int I = 0, int N = 0> static constexpr int num_vars() {
    if constexpr (I == TYPE(entries)::Size) {
      return N;
    } else {
      return num_vars<I + 1, std::max(N, TYPE(std::declval<MetaGrad<Vec>>()
                                                  .template get<I>())::Index +
                                             1)>();
    }
  }

  template <int I = 0, int N = 1000> static constexpr int first_var() {
    if constexpr (I == TYPE(entries)::Size) {
      return N;
    } else if constexpr (I == 0) {
      return first_var<I + 1, TYPE(std::declval<MetaGrad<Vec>>()
                                       .template get<I>())::Index>();
    } else {
      return first_var<I + 1, std::min(N, TYPE(std::declval<MetaGrad<Vec>>()
                                                   .template get<I>())::Index +
                                              0)>();
    }
  }

  template <int I> decltype(auto) get() const {
    return entries.template get<I>();
  }

  template <int I = 0, typename OtherVec>
  decltype(auto) operator+(const MetaGrad<OtherVec> &other) const {
    if constexpr (I == TYPE(other.entries)::Size) {
      auto res = *this;
      return res;
    } else {
      return add_to_grad(other.template get<I>())
          .template operator+ <I + 1>(other);
    }
  }

  template <int I = 0, typename OtherVec>
  decltype(auto) operator-(const MetaGrad<OtherVec> &other) const {
    if constexpr (I == TYPE(other.entries)::Size) {
      auto res = *this;
      return res;
    } else {
      auto entry = other.template get<I>();
      entry.val = -entry.val;
      return add_to_grad(entry).template operator- <I + 1>(other);
    }
  }

  template <typename G, typename = typename std::enable_if<
                            std::is_arithmetic<G>::value, G>::type>
  decltype(auto) operator*(G other) const {
    auto new_entries = entries * other;
    return MetaGrad<TYPE(new_entries)>(new_entries);
  }

  template <int Var, int I = 0>
  decltype(auto) add_to_grad(const GradEntry<Var> &entry) const {
    if constexpr (I == TYPE(entries)::Size) {
      auto new_entries = entries.push(entry);
      return MetaGrad<TYPE(new_entries)>(new_entries);
    } else if constexpr (TYPE(entries.template get<I>())::Index == Var) {
      auto entry_i = entries.template get<I>();
      entry_i.val += entry.val;
      auto new_entries = entries.template set<I>(entry_i);
      return MetaGrad<TYPE(new_entries)>(new_entries);
    } else {
      return add_to_grad<Var, I + 1>(entry);
    }
  }

  template <int M = 0, typename Hess, int I = 0, int J = 0>
  void rank_update(Hess &hess, double mul = 1.0) const {
    if constexpr (I == TYPE(entries)::Size) {
      return;
    } else if constexpr (J == I + 1) {
      rank_update<M, Hess, I + 1>(hess, mul);
    } else {
      auto entry_i = entries.template get<I>();
      auto entry_j = entries.template get<J>();
      constexpr int i = TYPE(entry_i)::Index - M;
      constexpr int j = TYPE(entry_j)::Index - M;
      double val = entry_i.val * entry_j.val;
      if constexpr (i > j) {
        hess(i, j) += val * mul;
      } else {
        hess(j, i) += val * mul;
      }
      rank_update<M, Hess, I, J + 1>(hess, mul);
    }
  }

  template <int M = 0, typename OtherVec, typename Hess, int I = 0, int J = 0>
  void rank_update(Hess &hess, const MetaGrad<OtherVec> &other,
                   double mul = 1.0) const {
    if constexpr (I == TYPE(entries)::Size) {
      return;
    } else if constexpr (J == TYPE(other.entries)::Size) {
      rank_update<M, OtherVec, Hess, I + 1>(hess, other, mul);
    } else {
      auto entry_i = entries.template get<I>();
      auto entry_j = other.entries.template get<J>();
      constexpr int i = TYPE(entry_i)::Index - M;
      constexpr int j = TYPE(entry_j)::Index - M;
      double val = entry_i.val * entry_j.val;
      if (i == j)
        val *= 2.0;
      if constexpr (i > j) {
        hess(i, j) += val * mul;
      } else {
        hess(j, i) += val * mul;
      }
      rank_update<M, OtherVec, Hess, I, J + 1>(hess, other, mul);
    }
  }

  template <int I = 0, typename Func> constexpr void for_each(Func &&f) const {
    if constexpr (I == TYPE(entries)::Size) {
      return;
    } else {
      f(get<I>());
      for_each<I + 1>(std::forward<Func>(f));
    }
  }

  friend std::ostream &operator<<(std::ostream &s, MetaGrad const &expr) {
    return s << expr.entries;
  }
};

} // namespace Optiz