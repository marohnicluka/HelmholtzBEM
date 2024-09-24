/**
 * \file parametrized_polynomial.cpp
 * \brief This file defines the class for representing a polynomial
 *        parametrization
 * @see parametrized_polynomial.cpp
 *
 *  This File is a part of the 2D-Parametric BEM package
 */

#include "parametrized_polynomial.hpp"
#include <assert.h>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <Eigen/Dense>

using CoefficientsList = typename ParametrizedPolynomial::CoefficientsList;

ParametrizedPolynomial::ParametrizedPolynomial(const ComplexSpline &spline,
                                               double tmin, double tmax, double len)
    : spline_(spline), length_(len), tmin_(tmin), tmax_(tmax) { }

Eigen::Vector2d ParametrizedPolynomial::operator()(double t) const {
  assert(IsWithinParameterRange(t));
  t = t * (tmax_ - tmin_) / 2 + (tmax_ + tmin_) / 2; // converting to the range [tmin,tmax]
  Eigen::Vector2d point;
  auto z = spline_.eval(t);
  point << z.real(), z.imag();
  return point;
}

Eigen::Vector2d ParametrizedPolynomial::operator[](double t) const {
  assert(IsWithinParameterRange(t));
  t = t * (tmax_ - tmin_) + tmin_; // converting to the range [tmin,tmax]
  Eigen::Vector2d point;
  auto z = spline_.eval(t);
  point << z.real(), z.imag();
  return point;
}

Eigen::Vector2d ParametrizedPolynomial::swapped_op(double t) const {
  assert(IsWithinParameterRange(t));
  t = tmax_ - t * (tmax_ - tmin_); // converting to the range [tmin,tmax]
  Eigen::Vector2d point;
  auto z = spline_.eval(t);
  point << z.real(), z.imag();
  return point;
}

Eigen::ArrayXXcd ParametrizedPolynomial::operator()(const Eigen::ArrayXXd &t) const {
  unsigned nr = t.rows(), nc = t.cols(), i, j;
  Eigen::ArrayXXcd ret(nr, nc);
  for (j = 0; j < nc; ++j) for (i = 0; i < nr; ++i) {
    auto v = this->operator()(t(i, j));
    ret(i, j) = std::complex<double>(v(0), v(1));
  }
  return ret;
}

Eigen::ArrayXXcd ParametrizedPolynomial::operator[](const Eigen::ArrayXXd &t) const {
  unsigned nr = t.rows(), nc = t.cols(), i, j;
  Eigen::ArrayXXcd ret(nr, nc);
  for (j = 0; j < nc; ++j) for (i = 0; i < nr; ++i) {
    auto v = this->operator[](t(i, j));
    ret(i, j) = std::complex<double>(v(0), v(1));
  }
  return ret;
}

Eigen::ArrayXXcd ParametrizedPolynomial::swapped_op(const Eigen::ArrayXXd &t) const {
  unsigned nr = t.rows(), nc = t.cols(), i, j;
  Eigen::ArrayXXcd ret(nr, nc);
  for (j = 0; j < nc; ++j) for (i = 0; i < nr; ++i) {
    auto v = this->swapped_op(t(i, j));
    ret(i, j) = std::complex<double>(v(0), v(1));
  }
  return ret;
}

Eigen::Vector2d ParametrizedPolynomial::Derivative(double t) const {
  assert(IsWithinParameterRange(t));
  t = t * (tmax_ - tmin_) / 2 + (tmax_ + tmin_) / 2; // converting to the range [tmin,tmax]
  Eigen::Vector2d point;
  auto z = spline_.eval_der(t);
  point << z.real(), z.imag();
  return point;
}

Eigen::Vector2d ParametrizedPolynomial::Derivative_01(double t) const {
  assert(IsWithinParameterRange(t));
  t = t * (tmax_ - tmin_) + tmin_; // converting to the range [tmin,tmax]
  Eigen::Vector2d point;
  auto z = spline_.eval_der(t);
  point << z.real(), z.imag();
  return point;
}

Eigen::Vector2d ParametrizedPolynomial::Derivative_01_swapped(double t) const {
  assert(IsWithinParameterRange(t));
  t = tmax_ - t * (tmax_ - tmin_); // converting to the range [tmin,tmax]
  Eigen::Vector2d point;
  auto z = spline_.eval_der(t);
  point << z.real(), z.imag();
  return point;
}

void ParametrizedPolynomial::Derivative(const Eigen::ArrayXXd &t, Eigen::ArrayXXcd &res, Eigen::ArrayXXd &norm) const {
  unsigned nr = t.rows(), nc = t.cols(), i, j;
  res.resize(nr, nc);
  for (j = 0; j < nc; ++j) for (i = 0; i < nr; ++i) {
    auto v = this->Derivative(t(i, j));
    res(i, j) = std::complex<double>(v(0), v(1));
  }
  norm = res.cwiseAbs();
}

void ParametrizedPolynomial::Derivative_01(const Eigen::ArrayXXd &t, Eigen::ArrayXXcd &res, Eigen::ArrayXXd &norm) const {
  unsigned nr = t.rows(), nc = t.cols(), i, j;
  res.resize(nr, nc);
  for (j = 0; j < nc; ++j) for (i = 0; i < nr; ++i) {
    auto v = this->Derivative_01(t(i, j));
    res(i, j) = std::complex<double>(v(0), v(1));
  }
  norm = res.cwiseAbs();
}

void ParametrizedPolynomial::Derivative_01_swapped(const Eigen::ArrayXXd &t, Eigen::ArrayXXcd &res, Eigen::ArrayXXd &norm, bool neg) const {
  unsigned nr = t.rows(), nc = t.cols(), i, j;
  res.resize(nr, nc);
  for (j = 0; j < nc; ++j) for (i = 0; i < nr; ++i) {
    auto v = this->swapped_op(t(i, j));
    res(i, j) = std::complex<double>(v(0), v(1));
  }
  norm = res.cwiseAbs();
  if (neg)
    res *= -1.;
}

Eigen::Vector2d ParametrizedPolynomial::DoubleDerivative(double t) const {
  assert(IsWithinParameterRange(t));
  t = t * (tmax_ - tmin_) / 2 + (tmax_ + tmin_) / 2; // converting to the range [tmin,tmax]
  Eigen::Vector2d point;
  auto z = spline_.eval_der(t, 2);
  point << z.real(), z.imag();
  return point;
}

PanelVector ParametrizedPolynomial::split(unsigned int N) const {
  // PanelVector for storing the part parametrizations
  PanelVector parametrization_parts;
  // Generating the parts
  for (int i = 0; i < N; ++i) {
    // Partitioning by splitting the parameter range [tmin,tmax]
    double tmin = tmin_ + i * (tmax_ - tmin_) / N;
    double tmax = tmin_ + (i + 1) * (tmax_ - tmin_) / N;
    if (i==N-1)
      tmax = tmax_;
    std::function<double(double)> f = [&](double t) {
      return std::abs(spline_.eval_der(t));
    };
    double len = boost::math::quadrature::gauss_kronrod<double, 15>::integrate(f, tmin, tmax, 5, 1e-8);
    // Adding the part parametrization to the vector with a shared pointer
    parametrization_parts.push_back(std::make_shared<ParametrizedPolynomial>(spline_, tmin, tmax, len));
  }
  return parametrization_parts;
}

double ParametrizedPolynomial::length() {
  return length_;
}

