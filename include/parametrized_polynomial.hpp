/**
 * \file parametrized_polynomial.hpp
 * \brief This file declares the class for representing a polynomial
 *        parametrization
 *
 *  This File is a part of the 2D-Parametric BEM package
 */

#ifndef PARAMETRIZEDPOLYNOMIALHPP
#define PARAMETRIZEDPOLYNOMIALHPP

#include "abstract_parametrized_curve.hpp"
#include "cspline.hpp"

/**
 * \class ParametrizedPolynomial
 * \brief This class represents a polynomial parametrization
 *        of the form \f$\gamma\f$(t) = \f$\sum_{n=0}^{N} a_n t^{n}\f$ and
 *        inherits from the Abstract base class representing parametrized curves
 * @see abstract_parametrized_curve.hpp
 */
class ParametrizedPolynomial : public AbstractParametrizedCurve {
public:
  /**
   * Defining the type for coefficient list for the parametrization
   */
  using CoefficientsList = typename Eigen::Matrix<double, 2, Eigen::Dynamic>;

  /**
   * Constructor.
   *
   * @param spline complex spline representing a 2D curve
   * @param tmin Lower end of the actual parameter interval used which is
   *             linearly mapped to the standard interval
   * @param tmax Upper end of the actual parameter interval used which is
   *             linearly mapped to the standard interval
   */
  ParametrizedPolynomial(const ComplexSpline &spline, double tmin, double tmax, double len);

  /**
   * See documentation in AbstractParametrizedCurve
   */
  Eigen::Vector2d operator()(double) const;
  Eigen::Vector2d operator[](double) const;
  Eigen::Vector2d swapped_op(double) const;
  Eigen::ArrayXXcd operator()(const Eigen::ArrayXXd &t) const;
  Eigen::ArrayXXcd operator[](const Eigen::ArrayXXd &t) const;
  Eigen::ArrayXXcd swapped_op(const Eigen::ArrayXXd &t) const;

  /**
   * See documentation in AbstractParametrizedCurve
   */
  Eigen::Vector2d Derivative(double) const;
  Eigen::Vector2d Derivative_01(double) const;
  Eigen::Vector2d Derivative_01_swapped(double) const;
  void Derivative(const Eigen::ArrayXXd &t, Eigen::ArrayXXcd &res, Eigen::ArrayXXd &norm) const;
  void Derivative_01(const Eigen::ArrayXXd &t, Eigen::ArrayXXcd &res, Eigen::ArrayXXd &norm) const;
  void Derivative_01_swapped(const Eigen::ArrayXXd &t, Eigen::ArrayXXcd &res, Eigen::ArrayXXd &norm, bool neg) const;

  /**
   * See documentation in AbstractParametrizedCurve
   */
  Eigen::Vector2d DoubleDerivative(double) const;

  /**
   * See documentation in AbstractParametrizedCurve
   */
  PanelVector split(unsigned int) const;

  /**
   * See documentation in AbstractParametrizedCurve
   */
  double length();

  /**
   * See documentation in AbstractParametrizedCurve
   */
  bool isLineSegment() const { return false; }

private:
  const ComplexSpline &spline_;
  double length_;
  /**
   * Storing the actual range of parameter within the parameter range
   * By default, it is exactly equal to the parameter range. It is used
   * for the split functionality to make part Fourier Sum parameterizations.
   */
  const double tmin_, tmax_;
}; // class ParametrizedPolynomial

#endif // PARAMETRIZEDPOLYNOMIALHPP
