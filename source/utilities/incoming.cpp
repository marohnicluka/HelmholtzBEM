/**
 * \file incoming.cpp
 *
 * \brief This file contains the implementation of incoming wave
 * functions.
 *
 * (c) 2024 Luka Marohnić
 */

#include "incoming.hpp"
#include "cbessel.hpp"
#include <gsl/gsl_integration.h>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iostream>

namespace incoming {

    const complex_t ii = complex_t(0.,1.);

    bool read_double(istringstream &iss, double &res) {
        string num;
        if (!getline(iss, num, ','))
            return false;
        try {
            res = stod(num);
        } catch (exception &e) {
            return false;
        }
        return true;
    }

    bool read_int(istringstream &iss, int &res) {
        string num;
        if (!getline(iss, num, ','))
            return false;
        try {
            res = stoi(num);
        } catch (exception &e) {
            return false;
        }
        return true;
    }

    bool load(const std::string& fname, incoming::wave& w) {
        ifstream input(fname);
        string s, param;
        while (getline(input, s)) {
            for (size_t i = s.size(); i-->0;) {
                if (isspace(s[i]))
                    s.erase(s.begin() + i);
            }
            if (s.empty() || s[0] == '#')
                continue;
            wave_params p;
            istringstream iss(s);
            getline(iss, param, ',');
            if (param == "Plane")
                p.type = Plane;
            else if (param == "CircularJ")
                p.type = CircularJ;
            else if (param == "CircularY")
                p.type = CircularY;
            else if (param == "FourierHankel1")
                p.type = FourierHankel1;
            else if (param == "FourierHankel2")
                p.type = FourierHankel2;
            else if (param == "Herglotz")
                p.type = Herglotz;
            else return false;
            if (!read_double(iss, p.amplitude))
                return false;
            if (!read_double(iss, p.x0(0)))
                return false;
            if (!read_double(iss, p.x0(1)))
                return false;
            if (p.type == Plane || p.type == Herglotz) {
                if (!read_double(iss, p.angle))
                    return false;
                if (p.type == Herglotz && !read_double(iss, p.eps))
                    return false;
            } else if (!read_int(iss, p.order))
                return false;
            if (getline(iss, param, ','))
                return false;
            w.push_back(p);
        }
        return true;
    }


    complex_t compute(const wave &w, const Eigen::Vector2d &x, const complex_t &k) {
        complex_t res = 0.;
        for_each (w.cbegin(), w.cend(), [&](const wave_params &p) {
            switch (p.type) {
            case Plane:
                res += p.amplitude * plane(x, p.x0, p.angle, k);
                break;
            case CircularJ:
                res += p.amplitude * circular_J(x, p.x0, p.order, k);
                break;
            case CircularY:
                res += p.amplitude * circular_Y(x, p.x0, p.order, k);
                break;
            case FourierHankel1:
                res += p.amplitude * fourier_hankel(x, p.x0, 1, p.order, k);
                break;
            case FourierHankel2:
                res += p.amplitude * fourier_hankel(x, p.x0, 2, p.order, k);
                break;
            case Herglotz:
                res += p.amplitude * herglotz(x, p.x0, p.angle, p.eps, k);
                break;
            }
        });
        return res;
    }

    Eigen::Vector2cd compute_del(const wave &w, const Eigen::Vector2d &x, const complex_t &k) {
        Eigen::Vector2cd res(0., 0.);
        for_each (w.cbegin(), w.cend(), [&](const wave_params &p) {
            switch (p.type) {
            case Plane:
                res += p.amplitude * plane_del(x, p.x0, p.angle, k);
                break;
            case CircularJ:
                res += p.amplitude * circular_J_del(x, p.x0, p.order, k);
                break;
            case CircularY:
                res += p.amplitude * circular_Y_del(x, p.x0, p.order, k);
                break;
            case FourierHankel1:
                res += p.amplitude * fourier_hankel_del(x, p.x0, 1, p.order, k);
                break;
            case FourierHankel2:
                res += p.amplitude * fourier_hankel_del(x, p.x0, 2, p.order, k);
                break;
            case Herglotz:
                res += p.amplitude * herglotz_del(x, p.x0, p.angle, p.eps, k);
                break;
            }
        });
        return res;
    }

    complex_t plane(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, double angle, const complex_t &k) {
        auto p = x - x0;
        return exp(ii * k * (p(0) * cos(angle) + p(1) * sin(angle)));
    }

    Eigen::Vector2cd plane_del(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, double angle, const complex_t &k) {
        Eigen::Vector2d d(cos(angle), sin(angle));
        return ii * k * plane(x, x0, angle, k) * d;
    }

    complex_t circular_J(const Eigen::Vector2d& x, const Eigen::Vector2d &x0, int l, const complex_t &k) {
        auto p = x - x0;
        double r = p.norm();
        double theta = atan2(p(1) / r, p(0) / r);
        return complex_bessel::J(l, k*r) * exp(ii * double(l) * theta);
    }

    Eigen::Vector2cd circular_J_del(const Eigen::Vector2d& x, const Eigen::Vector2d &x0, int l, const complex_t &k, complex_t *ddr) {
        auto p = x - x0;
        double r = p.norm();
        complex_t a = k * r;
        double theta = atan2(p(1) / r, p(0) / r), s = sin(theta), c = cos(theta);
        complex_t il = ii * double(l);
        complex_t jl = complex_bessel::J(l, a), jlp = complex_bessel::J(l-1, a), jln = complex_bessel::J(l+1, a);
        complex_t dfdr = k * (jlp - jln) / 2., dfdt = jl * il;
        Eigen::Vector2cd res;
        res << dfdr * c - dfdt * s / r, dfdr * s + dfdt * c / r;
        res *= exp(-il * theta);
        if (ddr != NULL)
            *ddr = p.normalized().dot(res);
        return res;
    }

    complex_t circular_Y(const Eigen::Vector2d& x, const Eigen::Vector2d &x0, int l, const complex_t &k) {
        auto p = x - x0;
        double r = p.norm();
        double theta = atan2(p(1) / r, p(0) / r);
        return complex_bessel::Y(l, k*r) * exp(ii * double(l) * theta);
    }

    Eigen::Vector2cd circular_Y_del(const Eigen::Vector2d& x, const Eigen::Vector2d &x0, int l, const complex_t &k, complex_t *ddr) {
        auto p = x - x0;
        double r = p.norm();
        complex_t a = k * r;
        double theta = atan2(p(1) / r, p(0) / r), s = sin(theta), c = cos(theta);
        complex_t il = ii * double(l);
        complex_t yl = complex_bessel::Y(l, a), ylp = complex_bessel::Y(l-1, a), yln = complex_bessel::Y(l+1, a);
        complex_t dfdr = k * (ylp - yln) / 2., dfdt = yl * il;
        Eigen::Vector2cd res;
        res << dfdr * c - dfdt / r * s, dfdr * s + dfdt / r * c;
        res *= exp(-il * theta);
        if (ddr != NULL)
            *ddr = p.normalized().dot(res);
        return res;
    }

    complex_t fourier_hankel(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, int kind, int l, const complex_t &k) {
        return circular_J(x, x0, l, k) + (kind == 1 ? ii : -ii) * circular_Y(x, x0, l, k);
    }

    Eigen::Vector2cd fourier_hankel_del(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, int kind, int l, const complex_t &k, complex_t *ddr) {
        complex_t ddr_1, ddr_2, a = (kind == 1 ? ii : -ii);
        auto v1 = circular_J_del(x, x0, l, k, &ddr_1);
        auto v2 = circular_Y_del(x, x0, l, k, &ddr_2);
        if (ddr != NULL)
            *ddr = ddr_1 + a * ddr_2;
        return v1 + a * v2;
    }

    double integrand_real(double theta, void *p) {
        double *params = (double*) p;
        complex_t k(params[0], params[1]);
        double x1 = params[2], x2 = params[3];
        return exp(ii * k * (x1 * cos(theta) + x2 * sin(theta))).real();
    }

    double integrand_imag(double theta, void *p) {
        double *params = (double*) p;
        complex_t k(params[0], params[1]);
        double x1 = params[2], x2 = params[3];
        return exp(ii * k * (x1 * cos(theta) + x2 * sin(theta))).imag();
    }

    double integrand_x1_real(double theta, void *p) {
        double *params = (double*) p;
        complex_t k(params[0], params[1]);
        double x1 = params[2], x2 = params[3];
        return (ii * k * cos(theta) * exp(ii * k * (x1 * cos(theta) + x2 * sin(theta)))).real();
    }

    double integrand_x1_imag(double theta, void *p) {
        double *params = (double*) p;
        complex_t k(params[0], params[1]);
        double x1 = params[2], x2 = params[3];
        return (ii * k * cos(theta) * exp(ii * k * (x1 * cos(theta) + x2 * sin(theta)))).imag();
    }

    double integrand_x2_real(double theta, void *p) {
        double *params = (double*) p;
        complex_t k(params[0], params[1]);
        double x1 = params[2], x2 = params[3];
        return (ii * k * sin(theta) * exp(ii * k * (x1 * cos(theta) + x2 * sin(theta)))).real();
    }

    double integrand_x2_imag(double theta, void *p) {
        double *params = (double*) p;
        complex_t k(params[0], params[1]);
        double x1 = params[2], x2 = params[3];
        return (ii * k * sin(theta) * exp(ii * k * (x1 * cos(theta) + x2 * sin(theta)))).imag();
    }

    complex_t herglotz(const Eigen::Vector2d& x, const Eigen::Vector2d &x0, double angle, double eps, const complex_t &k) {
        size_t N = 1000;
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(N);
        double result_real, result_imag, error, p[4], lb = angle - eps, ub = angle + eps;
        p[0] = k.real();
        p[1] = k.imag();
        p[2] = x(0) - x0(0);
        p[3] = x(1) - x0(1);
        gsl_function F;
        F.params = &p;
        F.function = &integrand_real;
        gsl_integration_qag(&F, lb, ub, 0., 1e-7, N, 3, w, &result_real, &error);
        F.function = &integrand_imag;
        gsl_integration_qag(&F, lb, ub, 0., 1e-7, N, 3, w, &result_imag, &error);
        gsl_integration_workspace_free(w);
        return result_real + ii * result_imag;
    }

    Eigen::Vector2cd herglotz_del(const Eigen::Vector2d& x, const Eigen::Vector2d &x0, double angle, double eps, const complex_t &k) {
        size_t N = 1000;
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(N);
        double result_real, result_imag, error, p[4], lb = angle - eps, ub = angle + eps;
        p[0] = k.real();
        p[1] = k.imag();
        p[2] = x(0) - x0(0);
        p[3] = x(1) - x0(1);
        Eigen::Vector2cd res;
        gsl_function F;
        F.params = &p;
        F.function = &integrand_x1_real;
        gsl_integration_qag(&F, lb, ub, 0., 1e-7, N, 3, w, &result_real, &error);
        F.function = &integrand_x1_imag;
        gsl_integration_qag(&F, lb, ub, 0., 1e-7, N, 3, w, &result_imag, &error);
        res(0) = result_real + ii * result_imag;
        F.function = &integrand_x2_real;
        gsl_integration_qag(&F, lb, ub, 0., 1e-7, N, 3, w, &result_real, &error);
        F.function = &integrand_x2_imag;
        gsl_integration_qag(&F, lb, ub, 0., 1e-7, N, 3, w, &result_imag, &error);
        gsl_integration_workspace_free(w);
        res(1) = result_real + ii * result_imag;
        return res;
    }

}

