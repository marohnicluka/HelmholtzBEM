/**
 * \file cspline.cpp
 *
 * \brief This file contains the implementation of
 * complex spline interpolation using GSL.
 *
 * (c) 2024 Luka Marohnić
 */

#include "cspline.hpp"

ComplexSpline::ComplexSpline(const Eigen::VectorXd &x, const Eigen::VectorXcd &y, bool periodic) {
    assert(x.size() == y.size());
    n = x.size();
    t = new double[n];
    re = new double[n];
    im = new double[n];
    for (unsigned i = 0; i < n; ++i) {
        t[i] = x(i);
        re[i] = y(i).real();
        im[i] = y(i).imag();
    }
    acc = gsl_interp_accel_alloc();
    s_re = gsl_spline_alloc(periodic ? gsl_interp_cspline_periodic : gsl_interp_cspline, n);
    s_im = gsl_spline_alloc(periodic ? gsl_interp_cspline_periodic : gsl_interp_cspline, n);
    gsl_spline_init(s_re, t, re, n);
    gsl_spline_init(s_im, t, im, n);
}

ComplexSpline::ComplexSpline(const ParametrizedMesh &mesh, const Eigen::VectorXcd &y) {
    const PanelVector &panels = mesh.getPanels();
    assert(panels.size() == y.size());
    n = panels.size();
    t = new double[n+1];
    re = new double[n+1];
    im = new double[n+1];
    double t0 = 0.;
    for (unsigned i = 0; i < n; ++i) {
        t[i] = t0;
        re[i] = y(i).real();
        im[i] = y(i).imag();
        t0 += panels[i]->length();
    }
    t[n] = t0;
    re[n] = re[0];
    im[n] = im[0];
    acc = gsl_interp_accel_alloc();
    s_re = gsl_spline_alloc(gsl_interp_cspline_periodic, n+1);
    s_im = gsl_spline_alloc(gsl_interp_cspline_periodic, n+1);
    gsl_spline_init(s_re, t, re, n+1);
    gsl_spline_init(s_im, t, im, n+1);
}

ComplexSpline::~ComplexSpline() {
    gsl_spline_free(s_re);
    gsl_spline_free(s_im);
    gsl_interp_accel_free(acc);
    delete[] t;
    delete[] re;
    delete[] im;
}

std::complex<double> ComplexSpline::eval(double x) const {
    return gsl_spline_eval(s_re, x, acc) + 1i * gsl_spline_eval(s_im, x, acc);
}
