/**
 * \file parametrized_line.cpp
 * \brief This file defines the class representing parametrization of
 *        a line segment in 2D.
 * @see parametrized_line.hpp
 *
 *  This File is a part of the 2D-Parametric BEM package
 */

#include "parametrized_line.hpp"
#include <assert.h>
#include <iostream>
#include <math.h>
#include <utility>
#include <Eigen/Dense>

using Point = typename ParametrizedLine::Point;
using namespace std::complex_literals;

ParametrizedLine::ParametrizedLine(Point first, Point second)
  : start_(first), end_(second) { }

Point ParametrizedLine::operator()(double t) const {
  assert(IsWithinParameterRange(t));
  double x1(start_(0)), y1(start_(1)), x2(end_(0)), y2(end_(1));
  // Linear interpolation of x & y coordinates based on parameter t
  Eigen::Vector2d point(t * (x2 - x1) + (x2 + x1), t * (y2 - y1) + (y2 + y1));
  return point * 0.5;
}
Eigen::ArrayXXcd ParametrizedLine::operator()(const Eigen::ArrayXXd &t) const {
  assert(IsWithinParameterRange(t));
  double x1(start_(0)), y1(start_(1)), x2(end_(0)), y2(end_(1));
  // Linear interpolation of x & y coordinates based on parameter t
  return (t * ((x2 - x1) + 1i * (y2 - y1)) + (x2 + x1) + 1i * (y2 + y1)) * .5;
}
Point ParametrizedLine::operator[](double t) const {
  assert(IsWithinParameterRange(t));
  double x1(start_(0)), y1(start_(1)), x2(end_(0)), y2(end_(1));
  // Linear interpolation of x & y coordinates based on parameter t
  Eigen::Vector2d point(t * (x2 - x1) + x1, t * (y2 - y1) + y1);
  return point;
}
Eigen::ArrayXXcd ParametrizedLine::operator[](const Eigen::ArrayXXd &t) const {
  assert(IsWithinParameterRange(t));
  double x1(start_(0)), y1(start_(1)), x2(end_(0)), y2(end_(1));
  // Linear interpolation of x & y coordinates based on parameter t
  return ((x2 - x1) + 1i * (y2 - y1)) * t + (x1 + 1i * y1);
}
Point ParametrizedLine::swapped_op(double t) const {
  assert(IsWithinParameterRange(t));
  double x1(start_(0)), y1(start_(1)), x2(end_(0)), y2(end_(1));
  // Linear interpolation of x & y coordinates based on parameter t
  Eigen::Vector2d point(-(x2 - x1) * t + x2, -(y2 - y1) * t + y2);
  return point;
}
Eigen::ArrayXXcd ParametrizedLine::swapped_op(const Eigen::ArrayXXd &t) const {
  assert(IsWithinParameterRange(t));
  double x1(start_(0)), y1(start_(1)), x2(end_(0)), y2(end_(1));
  // Linear interpolation of x & y coordinates based on parameter t
  return ((x1 - x2) + 1i * (y1 - y2)) * t + (x2 + 1i * y2);
}

Eigen::Vector2d ParametrizedLine::Derivative(double t) const {
  assert(IsWithinParameterRange(t));
  double x1(start_(0)), y1(start_(1)), x2(end_(0)), y2(end_(1));
  // Derivative of the linear interpolation used in the function operator()
  Eigen::Vector2d derivative(x2 - x1, y2 - y1);
  return 0.5 * derivative;
}
void ParametrizedLine::Derivative(const Eigen::ArrayXXd &t, Eigen::ArrayXXcd &res, Eigen::ArrayXXd &norm) const {
  assert(IsWithinParameterRange(t));
  double x1(start_(0)), y1(start_(1)), x2(end_(0)), y2(end_(1));
  // Derivative of the linear interpolation used in the function operator()
  std::complex<double> z = ((x2 - x1) + 1i * (y2 - y1)) * .5;
  res.setConstant(t.rows(), t.cols(), z);
  norm.setConstant(t.rows(), t.cols(), std::abs(z));
}
Eigen::Vector2d ParametrizedLine::Derivative_01(double t) const {
  assert(IsWithinParameterRange(t));
  double x1(start_(0)), y1(start_(1)), x2(end_(0)), y2(end_(1));
  // Derivative of the linear interpolation used in the function operator()
  Eigen::Vector2d derivative(x2 - x1, y2 - y1);
  return derivative;
}
void ParametrizedLine::Derivative_01(const Eigen::ArrayXXd &t, Eigen::ArrayXXcd &res, Eigen::ArrayXXd &norm) const {
  assert(IsWithinParameterRange(t));
  double x1(start_(0)), y1(start_(1)), x2(end_(0)), y2(end_(1));
  // Derivative of the linear interpolation used in the function operator()
  std::complex<double> z = (x2 - x1) + 1i * (y2 - y1);
  res.setConstant(t.rows(), t.cols(), z);
  norm.setConstant(t.rows(), t.cols(), std::abs(z));
}
Eigen::Vector2d ParametrizedLine::Derivative_01_swapped(double t) const {
  assert(IsWithinParameterRange(t));
  double x1(start_(0)), y1(start_(1)), x2(end_(0)), y2(end_(1));
  // Derivative of the linear interpolation used in the function operator()
  Eigen::Vector2d derivative(x1 - x2, y1 - y2);
  return derivative;
}
void ParametrizedLine::Derivative_01_swapped(const Eigen::ArrayXXd &t, Eigen::ArrayXXcd &res, Eigen::ArrayXXd &norm, bool neg) const {
  assert(IsWithinParameterRange(t));
  double x1(start_(0)), y1(start_(1)), x2(end_(0)), y2(end_(1));
  // Derivative of the linear interpolation used in the function operator()
  std::complex<double> z = (x1 - x2) + 1i * (y1 - y2);
  res.setConstant(t.rows(), t.cols(), (neg ? -1. : 1.) * z);
  norm.setConstant(t.rows(), t.cols(), std::abs(z));
}

Eigen::Vector2d ParametrizedLine::DoubleDerivative(double t) const {
  assert(IsWithinParameterRange(t));
  // Double Derivative of the linear interpolation used in function operator()
  Eigen::Vector2d double_derivative(0, 0);
  return double_derivative;
}

PanelVector ParametrizedLine::split(unsigned int N) const {
  // PanelVector for storing the part parametrizations
  PanelVector parametrization_parts;
  double tmin, tmax;
  std::tie(tmin, tmax) = ParameterRange();
  // Generating the parts
  for (int i = 0; i < N; ++i) {
    // Partitioning by splitting the parameter line segment
    // Splitting the parameter range to split the line segment
    double t1 = tmin + i * (tmax - tmin) / N;
    double t2 = tmin + (i + 1) * (tmax - tmin) / N;
    if (i==N-1)
      t2 = tmax;
    // Using evaluation on the split parameter range to get line segment split
    // points
    Point first = this->operator()(t1);
    Point second = this->operator()(t2);
    // Adding the part parametrization to the vector with a shared pointer
    parametrization_parts.push_back(std::make_shared<ParametrizedLine>(first, second));
  }
  return parametrization_parts;
}

double ParametrizedLine::length() const {
  double tmin, tmax;
  // Getting the parameter range
  std::tie(tmin, tmax) = ParameterRange();
  return (this->operator()(tmax) - this->operator()(tmin)).norm();
}

