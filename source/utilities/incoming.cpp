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
#include "gauleg.hpp"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iostream>

namespace incoming {

    const complex_t ii = complex_t(0.,1.);
    const auto gaussQR = getCGaussQR(21);

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
        return complex_bessel::BesselJ(l, k*r) * exp(ii * double(l) * theta);
    }

    Eigen::Vector2cd circular_J_del(const Eigen::Vector2d& x, const Eigen::Vector2d &x0, int l, const complex_t &k) {
        auto p = x - x0;
        double r = p.norm();
        complex_t a = k * r;
        double theta = atan2(p(1) / r, p(0) / r), s = sin(theta), c = cos(theta);
        complex_t il = ii * double(l);
        complex_t jl = complex_bessel::BesselJ(l, a), jlp = complex_bessel::BesselJ(l-1, a), jln = complex_bessel::BesselJ(l+1, a);
        complex_t dfdr = k * (jlp - jln) / 2., dfdt = jl * il;
        Eigen::Vector2cd res;
        res << dfdr * c - dfdt * s / r, dfdr * s + dfdt * c / r;
        return res * exp(il * theta);
    }

    complex_t circular_Y(const Eigen::Vector2d& x, const Eigen::Vector2d &x0, int l, const complex_t &k) {
        auto p = x - x0;
        double r = p.norm();
        double theta = atan2(p(1) / r, p(0) / r);
        return complex_bessel::BesselY(l, k*r) * exp(ii * double(l) * theta);
    }

    Eigen::Vector2cd circular_Y_del(const Eigen::Vector2d& x, const Eigen::Vector2d &x0, int l, const complex_t &k) {
        auto p = x - x0;
        double r = p.norm();
        complex_t a = k * r;
        double theta = atan2(p(1) / r, p(0) / r), s = sin(theta), c = cos(theta);
        complex_t il = ii * double(l);
        complex_t yl = complex_bessel::BesselY(l, a), ylp = complex_bessel::BesselY(l-1, a), yln = complex_bessel::BesselY(l+1, a);
        complex_t dfdr = k * (ylp - yln) / 2., dfdt = yl * il;
        Eigen::Vector2cd res;
        res << dfdr * c - dfdt / r * s, dfdr * s + dfdt / r * c;
        return res * exp(il * theta);
    }

    complex_t fourier_hankel(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, int kind, int l, const complex_t &k) {
        return circular_J(x, x0, l, k) + (kind == 1 ? ii : -ii) * circular_Y(x, x0, l, k);
    }

    Eigen::Vector2cd fourier_hankel_del(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, int kind, int l, const complex_t &k) {
        return circular_J_del(x, x0, l, k) + (kind == 1 ? ii : -ii) * circular_Y_del(x, x0, l, k);
    }

    complex_t herglotz(const Eigen::Vector2d& x, const Eigen::Vector2d &x0, double angle, double eps, const complex_t &k) {
        double lb = angle - eps, ub = angle + eps, h = ub - lb;
        auto y = x - x0;
        auto theta = lb + h * gaussQR.x.array();
        return ((ii * k * (y(0) * theta.cos() + y(1) * theta.sin())).exp() * gaussQR.w.array()).sum();
    }

    Eigen::Vector2cd herglotz_del(const Eigen::Vector2d& x, const Eigen::Vector2d &x0, double angle, double eps, const complex_t &k) {
        double lb = angle - eps, ub = angle + eps, h = ub - lb;
        auto y = x - x0;
        auto theta = lb + h * gaussQR.x.array();
        auto cos_theta = theta.cos();
        auto sin_theta = theta.sin();
        auto e = (ii * k * (y(0) * cos_theta + y(1) * sin_theta)).exp();
        Eigen::Vector2cd res;
        res << (cos_theta * e * gaussQR.w.array()).sum(), (sin_theta * e * gaussQR.w.array()).sum();
        return ii * k * res;
    }

}

