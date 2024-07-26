#pragma once

namespace Optiz {

template<typename>
struct is_optiz_complex : std::false_type {};

template <typename T>
class Complex;
template<typename T>
struct is_optiz_complex<Optiz::Complex<T>> : std::true_type {};


template <typename T>
class Complex {
 public:
  Complex() : _real(T(0.0)), _imag(T(0.0)) {}
  Complex(const T& real, const T& imag) : _real(real), _imag(imag) {}

  template <typename G>
  Complex& operator+=(const G& other) {
    if constexpr (is_optiz_complex<G>::value) {
      _real += other.real();
      _imag += other.imag();
    } else {
      _real += other;
    }
    return *this;
  }
  template <typename G>
  Complex operator+(const G& other) const {
    return Complex(_real, _imag) += other;
  }

  template <typename G>
  Complex& operator-=(const G& other) {
    if constexpr (is_optiz_complex<G>::value) {
      _real -= other.real();
      _imag -= other.imag();
    } else {
      _real -= other;
    }
    return *this;
  }
  template <typename G>
  Complex operator-(const G& other) const {
    return Complex(_real, _imag) -= other;
  }

  template <typename G>
  Complex& operator*=(const G& other) {
    if constexpr (is_optiz_complex<G>::value) {
      T new_real = _real * other.real() - _imag * other.imag();
      _imag = _real * other.imag() + other.real() * _imag;
      _real = new_real;
    } else if constexpr (std::is_same_v<std::decay_t<decltype(other)>, std::complex<double>>) {
      T new_real = _real * other.real() - _imag * other.imag();
      _imag = _real * other.imag() + other.real() * _imag;
      _real = new_real;
    } else {
      _real *= other;
      _imag *= other;
    }
    return *this;
  }
  template <typename G>
  Complex operator*(const G& other) const {
    return Complex(_real, _imag) *= other;
  }

  template <typename G>
  Complex& operator/=(const G& other) {
    if constexpr (std::is_same_v<decltype(other), T>) {
      return operator*=(other.inv());
    } else {
      _real /= other;
      _imag /= other;
    }
    return *this;
  }
  template <typename G>
  Complex operator/(const G& other) const {
    return Complex(_real, _imag) /= other;
  }

  T ang() const {
    return atan2(_imag, _real);
  }

  Complex log() const {
    return Complex(std::log(mag()), ang());
  }

  Complex pow(double n) const {

  }

  Complex conj() const { return Complex(_real, -_imag); }

  T mag() const { return sqrt(_real * _real + _imag * _imag); }
  T magsqr() const { return _real * _real + _imag * _imag; }

  Complex inv() const {
    T lensqr = magsqr();
    return Complex(_real / lensqr, -_imag / lensqr);
  }

  T& real() { return _real; }
  T& imag() { return _imag; }
  T real() const { return _real; }
  T imag() const { return _imag; }

  friend std::ostream& operator<<(std::ostream& s, const Complex& comp) {
    s << "real: " << comp.real() << ", "
      << "imag: " << comp.imag();
    return s;
  }

  Complex operator-() const {
    return Complex(-_real, -_imag);
  }

  template<typename G,
  std::enable_if_t<std::is_same_v<G, Complex&>, bool> = false>
  friend Complex operator*(const G& a, const Complex& b) {
    return Complex(a) *= b;
  }
  template<typename G,
  std::enable_if_t<std::is_same_v<G, Complex&>, bool> = false>
  friend Complex operator+(const G& a, const Complex& b) {
    return Complex(a) += b;
  }
  template<typename G,
  std::enable_if_t<std::is_same_v<G, Complex&>, bool> = false>
  friend Complex operator-(const G& a, const Complex& b) {
    return Complex(a) -= a;
  }
  template<typename G,
  std::enable_if_t<std::is_same_v<G, Complex&>, bool> = false>
  friend Complex operator/(const G& a, const Complex& b) {
    return Complex(a) /= b;
  }

  operator Eigen::VectorX<T>() const {
    Eigen::VectorX<T> res(2);
    res << _real, _imag;
    return res;
  }

 private:
  T _real;
  T _imag;
};
}  // namespace Optiz
