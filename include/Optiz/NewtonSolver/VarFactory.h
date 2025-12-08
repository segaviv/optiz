#pragma once
#include <Eigen/Eigen>
#include <memory>
#include <vector>

#include "../Autodiff/Var.h"

namespace Optiz {

template <typename T> class TGenericVariableFactory {
public:
  TGenericVariableFactory(
      const Eigen::VectorXd &current, const std::pair<int, int> &shape,
      const std::shared_ptr<void> &state = nullptr,
      const std::vector<std::pair<int, int>> &block_shapes = {})
      : _current(current), shape(shape), state(state),
        block_shapes(block_shapes) {
    offsets.push_back(0);
    for (const auto &bs : block_shapes) {
      offsets.push_back(offsets.back() + bs.first * bs.second);
    }
  }

  using Scalar = T;

  virtual T operator()(int i) const = 0;
  virtual T operator()(int i, int j) const = 0;
  Eigen::Map<const Eigen::MatrixXd> current_mat() const {
    return Eigen::Map<const Eigen::MatrixXd>(_current.data(), shape.first,
                                             shape.second);
  }
  std::shared_ptr<void> get_state() const { return state; }
  int num_vars() const { return _current.size(); }

  Eigen::RowVectorX<T> row(int i) const {
    Eigen::RowVectorX<T> result(shape.second);
    for (int j = 0; j < shape.second; j++) {
      result(j) = operator()(i, j);
    }
    return result;
  }
  template <int k> Eigen::RowVector<T, k> row(int i) const {
    Eigen::RowVector<T, k> result;
    for (int j = 0; j < k; j++) {
      result(j) = operator()(i, j);
    }
    return result;
  }

  const T &get(const T &v) { return v; }

  const Eigen::VectorXd &current() const { return _current; }

  template <typename State> const State &get_state() const {
    return *static_cast<const State *>(state.get());
  }

  // Overridden in case of more than one variables block.
  virtual const TGenericVariableFactory &var_block(int index) const {
    return *this;
  }

public:
  const Eigen::VectorXd &_current;
  std::pair<int, int> shape;
  std::shared_ptr<void> state;
  std::vector<std::pair<int, int>> block_shapes;
  std::vector<int> offsets;
};

class VarFactoryWithOffset : public TGenericVariableFactory<Var> {
public:
  VarFactoryWithOffset(const Eigen::VectorXd &init,
                       const std::pair<int, int> &shape, int offset,
                       const std::shared_ptr<void> &state = nullptr);
  Var operator()(int i) const;
  Var operator()(int i, int j) const;

protected:
  int offset;
};

class VecVarFactory : public TGenericVariableFactory<Var> {
public:
  VecVarFactory(const Eigen::VectorXd &init,
                const std::vector<std::pair<int, int>> &block_shapes);

  Var operator()(int i) const;
  Var operator()(int i, int j) const;
  const TGenericVariableFactory &var_block(int index) const;

private:
  std::vector<VarFactoryWithOffset> var_factories;
};

template <typename T>
class TVarFactoryWithOffset : public TGenericVariableFactory<T> {
public:
  TVarFactoryWithOffset(const Eigen::VectorXd &init,
                        const std::pair<int, int> &shape, int offset,
                        const std::shared_ptr<void> &state = nullptr)
      : TGenericVariableFactory<T>(init, shape, state), offset(offset) {}
  T operator()(int i) const {
    return T(this->_current(i + offset), i + offset);
  }
  T operator()(int i, int j) const {
    int index = offset + i + j * this->shape.first;
    return T(this->_current(index), index);
  }

private:
  int offset;
};

template <typename T> class TVarFactory : public TGenericVariableFactory<T> {
public:
  TVarFactory(const Eigen::VectorXd &init, const std::pair<int, int> &shape,
              const std::shared_ptr<void> &state = nullptr)
      : TGenericVariableFactory<T>(init, shape, state) {}
  // Initialize from another generic factory.
  template <typename G>
  TVarFactory(const TGenericVariableFactory<G> &other)
      : TGenericVariableFactory<T>(other.current(), other.shape,
                                   other.get_state(), other.block_shapes) {
    for (int i = 0; i < this->block_shapes.size(); i++) {
      var_factories.push_back(
          TVarFactoryWithOffset<T>(this->_current, this->block_shapes[i],
                                   this->offsets[i], this->state));
    }
  }
  T operator()(int i) const { return T(this->_current(i), i); }
  T operator()(int i, int j) const {
    int index = i + j * this->shape.first;
    return T(this->_current(index), index);
  }
  const TGenericVariableFactory<T> &var_block(int index) const {
    return var_factories[index];
  }

private:
  std::vector<TVarFactoryWithOffset<T>> var_factories;
};

template <typename T> class ValFactory : public TGenericVariableFactory<T> {
public:
  ValFactory(const Eigen::VectorXd &init, const std::pair<int, int> &shape,
             const std::shared_ptr<void> &state = nullptr,
             const std::vector<std::pair<int, int>> &block_shapes = {})
      : TGenericVariableFactory<T>(init, shape, state, block_shapes) {}

  T operator()(int i) const { return this->_current(i); }

  T operator()(int i, int j) const {
    int index = i + j * this->shape.first;
    return this->_current(index);
  }
};

class ValFactoryWithOffset : public TGenericVariableFactory<double> {
public:
  ValFactoryWithOffset(const Eigen::VectorXd &init,
                       const std::pair<int, int> &shape, int offset,
                       const std::shared_ptr<void> &state = nullptr);

  double operator()(int i) const;
  double operator()(int i, int j) const;

private:
  int offset;
};

class VecValFactory : public ValFactory<double> {
public:
  VecValFactory(const Eigen::VectorXd &init,
                const std::vector<std::pair<int, int>> &block_shapes);

  double operator()(int i) const;
  double operator()(int i, int j) const;
  const TGenericVariableFactory &var_block(int index) const;

private:
  std::vector<ValFactoryWithOffset> val_factories;
};

using VarFactory = TVarFactory<Var>;
extern template class TVarFactory<Var>;

} // namespace Optiz