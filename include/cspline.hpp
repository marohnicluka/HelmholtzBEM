/**
 * \file cspline.hpp
 *
 * \brief This is a C++ library for (periodic) spline
 * interpolation of complex values using GSL routines
 * for real splines.
 *
 * (c) 2024 Luka Marohnić
 */

#ifndef CSPLINE_HPP
#define CSPLINE_HPP

#include <complex>
#include <Eigen/Dense>
#include "gsl/gsl_spline.h"
#include "parametrized_mesh.hpp"

using namespace std;

/**
 * This class represents a (periodic) Akima spline interpolation
 * of a complex univariate function.
 */
class ComplexSpline {
    unsigned int n;
    double *t;
    double *re;
    double *im;
    gsl_spline *s_re;
    gsl_spline *s_im;
    gsl_interp_accel *acc;

public:
    ComplexSpline(const Eigen::VectorXd &x, const Eigen::VectorXcd &y, bool periodic = false);
    ComplexSpline(const ParametrizedMesh &mesh, const Eigen::VectorXcd &y);
    ~ComplexSpline();
    unsigned int num_points() const { return n; }
    double x_val(unsigned i) const { assert(i < n); return t[i]; }
    std::complex<double> y_val(unsigned i) const { assert(i < n); return re[i] + 1i * im[i]; }
    std::complex<double> eval(double x) const;
    std::complex<double> eval_der(double x, int order = 1) const;
};

#endif // CSPLINE_HPP
