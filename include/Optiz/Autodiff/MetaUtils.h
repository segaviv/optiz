#pragma once

#include <type_traits>

#define TYPE(x) std::decay_t<decltype(x)>
#define INDEX(for_index) TYPE(for_index)::index

template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

namespace Optiz {
template <int I> struct ForIndex {
  constexpr static int index = I;
};

template <int I, int N, typename Func> constexpr void For(Func &&f) {
  if constexpr (I == N) {
    return;
  } else {
    f.template operator()<I>();
    For<I + 1, N>(std::forward<Func>(f));
  }
}

template <typename Derived> class MetaVarBase;
template <typename Derived1,
          typename GradType = decltype(std::declval<Derived1>()._grad)>
class MetaVarChain;

template <typename Derived1>
MetaVarChain<Derived1> sqr(MetaVarBase<Derived1> const &derived1);

} // namespace Optiz