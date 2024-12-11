#include "VarFactory.h"

namespace Optiz {
template class TVarFactory<Var>;

Var VarFactoryWithOffset::operator()(int i, int j) const {
  int index = offset + i + j * this->shape.first;
  return Var(this->_current(index), index);
}

Var VarFactoryWithOffset::operator()(int i) const {
  return Var(this->_current(i + offset), i + offset);
}

VarFactoryWithOffset::VarFactoryWithOffset(const Eigen::VectorXd &init,
                                           const std::pair<int, int> &shape,
                                           int offset,
                                           const std::shared_ptr<void> &state)
    : TGenericVariableFactory<Var>(init, shape, state), offset(offset) {}

VecVarFactory::VecVarFactory(
    const Eigen::VectorXd &init, const std::vector<int> &block_start_indices,
    const std::vector<std::pair<int, int>> &block_shapes)
    : TGenericVariableFactory<Var>(init, {init.size(), 1}) {
  for (int i = 0; i < block_start_indices.size(); i++) {
    var_factories.push_back(
        VarFactoryWithOffset(init, block_shapes[i], block_start_indices[i]));
  }
}

Var VecVarFactory::operator()(int i) const { return Var(this->_current(i), i); }

Var VecVarFactory::operator()(int i, int j) const {
  return var_factories[i](j);
}

const TGenericVariableFactory<Var> &VecVarFactory::var_block(int index) const {
  return var_factories[index];
}

ValFactoryWithOffset::ValFactoryWithOffset(const Eigen::VectorXd &init,
                                           const std::pair<int, int> &shape,
                                           int offset,
                                           const std::shared_ptr<void> &state)
    : TGenericVariableFactory<double>(init, shape, state), offset(offset) {}
double ValFactoryWithOffset::operator()(int i) const {
  return this->_current(i + offset);
}
double ValFactoryWithOffset::operator()(int i, int j) const {
  int index = offset + i + j * this->shape.first;
  return this->_current(index);
}

VecValFactory::VecValFactory(
    const Eigen::VectorXd &init, const std::vector<int> &block_start_indices,
    const std::vector<std::pair<int, int>> &block_shapes)
    : ValFactory<double>(init, {init.size(), 1}) {
  for (int i = 0; i < block_start_indices.size(); i++) {
    val_factories.push_back(
        ValFactoryWithOffset(init, block_shapes[i], block_start_indices[i]));
  }
}

double VecValFactory::operator()(int i) const { return this->_current(i); }

double VecValFactory::operator()(int i, int j) const {
  return val_factories[i](j);
}

const TGenericVariableFactory<double> &VecValFactory::var_block(int index) const {
  return val_factories[index];
}

} // namespace Optiz
