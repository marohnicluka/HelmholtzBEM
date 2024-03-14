/**
 * \file incoming.hpp
 *
 * \brief This is a C++ library of incoming waves that satisfy Delta u + k^2 u = 0.
 *
 * (c) 2024 Luka Marohnić
 */

#ifndef INCOMING_HPP
#define INCOMING_HPP

#include <complex>
#include <Eigen/Dense>

using namespace std;

typedef complex<double> complex_t;

namespace incoming {

    enum WaveType { Plane, CircularJ, CircularY, FourierHankel1, FourierHankel2, Herglotz };

    typedef struct {
        int type;
        double amplitude;
        double eps;
        double angle;
        int order;
        Eigen::Vector2d x0;
    } wave_params;

    typedef vector<wave_params> wave;

    /**
     * This function loads incoming wave from a file.
     * If an error occurs, then false is returned.
     *
     * @param fname file name
     * @param w wave specification (the result)
     */
    bool load(const string &fname, wave &w);

    /**
     * This function computes incoming wave as a
     * linear combination of waves specified in spec
     * and wavenumber k at x
     *
     * @param w wave specification
     * @param x real 2d vector
     * @param k wavenumber
     */
    complex_t compute(const wave &w, const Eigen::Vector2d &x, double k);
    // gradient
    Eigen::Vector2cd compute_del(const wave &w, const Eigen::Vector2d &x, double k);

    /**
     * This function computes the plane wave with
     * direction d and wavenumber k at x
     *
     * u(x) = exp(i*k*dot(x,d))
     *
     * @param x real 2d vector
     * @param d real 2d unit vector
     * @param k wavenumber
     */
    complex_t plane(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, double angle, double k);
    // gradient
    Eigen::Vector2cd plane_del(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, double angle, double k);

    /**
     * This function computes the circular besselJ
     * wave with order l and wavenumber k at x
     *
     * u(x) = J(l,k*r)*exp(i*l*theta),
     * x = (r*cos(theta), r*sin(theta))
     *
     * @param x real 2d vector
     * @param l integral order
     * @param k wavenumber
     */
    complex_t circular_J(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, int l, double k);
    // gradient
    Eigen::Vector2cd circular_J_del(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, int l, double k);

    /**
     * This function computes the circular besselY
     * wave with order l and wavenumber k at x
     *
     * u(x) = Y(l,k*r)*exp(i*l*theta),
     * x = (r*cos(theta), r*sin(theta))
     *
     * @param x real 2d vector
     * @param l integral order
     * @param k wavenumber
     */
    complex_t circular_Y(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, int l, double k);
    // gradient
    Eigen::Vector2cd circular_Y_del(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, int l, double k);

    /**
     * This function computes the Fourier-Hankel
     * wave with order l and wavenumber k at x
     *
     * u(x) = (J(l,k*r) +- i*Y(l,k*r)) * exp(i*l*theta),
     * x = (r*cos(theta), r*sin(theta))
     *
     * @param x real 2d vector
     * @param kind integer = 1, 2
     * @param l integral order
     * @param k wavenumber
     */
    complex_t fourier_hankel(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, int kind, int l, double k);
    // gradient
    Eigen::Vector2cd fourier_hankel_del(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, int kind, int l, double k);

    /**
     * This function computes the Herglotz ray wave
     * with direction d, diameter eps and wavenumber k at x
     *
     * u(x) = integral(exp(i*k*(x1*cos(t)+x2*sin(t))), t=theta-eps..theta+eps),
     * theta = atan2(x2, x1)
     *
     * @param x real 2d vector
     * @param d real 2d vector
     * @param eps ray width
     * @param k wavenumber
     */
    complex_t herglotz(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, double angle, double eps, double k);
    // gradient
    Eigen::Vector2cd herglotz_del(const Eigen::Vector2d &x, const Eigen::Vector2d &x0, double angle, double eps, double k);

}

#endif // INCOMING_HPP
