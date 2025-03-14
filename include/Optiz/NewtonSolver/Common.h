#pragma once
#include "../Autodiff/TDenseVar.h"
#include "../Autodiff/Var.h"
#include "Complex.h"
#include "../NewtonSolver/ElementFunc.h"

namespace Optiz {

// Project psd.
template <typename T> inline T project_psd(const T &x) { return x; }

inline Var project_psd(const Var &x) {
  Var res = Var(x).projectHessian();
  return res;
}
inline Var project_psd(Var &&x) {
  x.projectHessian();
  return x;
}
template <int k> inline TDenseVar<k> project_psd(const TDenseVar<k> &x) {
  TDenseVar<k> res = TDenseVar<k>(x).projectHessian();
  return res;
}
template <int k> inline TDenseVar<k> project_psd(TDenseVar<k> &&x) {
  x.projectHessian();
  return x;
}

// Val
template <typename T> inline T val(const T &x) { return x; }

inline double val(const Var &x) { return x.val(); }

template <int k> inline double val(const TDenseVar<k> &x) { return x.val(); }

template <typename T> inline Complex<double> val(const Complex<T> &x) {
  return Complex(val(x.real()), val(x.imag()));
}

template <int k>
LocalVarFactory<k>
get_local_factory(const TGenericVariableFactory<Var> &other) {
  return LocalVarFactory<k>(other);
}

template <int k>
const ValFactory<double> &get_local_factory(const ValFactory<double> &fac) {
  return fac;
}

} // namespace Optiz