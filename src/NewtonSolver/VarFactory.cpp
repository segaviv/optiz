#include "VarFactory.h"

namespace Optiz {
  template class TVarFactory<Var>;
  template class VecValFactory<double>;


VecVarFactory::VecVarFactory(const Eigen::VectorXd& init, 
const std::vector<int>& block_start_indices): 
TGenericVariableFactory<Var>(init, {init.size(), 1}), block_start_indices(block_start_indices) {}

Var VecVarFactory::operator()(int i) const {
  return Var(this->_current(i), i);
}

Var VecVarFactory::operator()(int i, int j) const {
  return operator()(block_start_indices[i] + j);
}

}
