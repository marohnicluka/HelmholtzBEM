#include <Eigen/Dense>
#include <cmath>
#include "gen_sol.hpp"
#include "cbessel.hpp"

using namespace std::complex_literals;
using namespace complex_bessel;

namespace sol {
    typedef std::complex<double> complex_t;

    complex_t fund_sol_dir(const complex_t &k, double x1, double x2, double ipt1, double ipt2) {
        // compute distance from interior evaluation point to evaluation point
        double r = std::sqrt((x1 - ipt1) * (x1 - ipt1) + (x2 - ipt2) * (x2 - ipt2));
        // evaluate fundamental solution using computed distance
        return 1i * HankelH1(0, k * r) * 0.25;
    }

    complex_t fund_sol_neu(const complex_t &k, double x1, double x2, double ipt1, double ipt2) {
        // compute distance from interior evaluation point to evaluation point
        Eigen::Vector2d x(x1, x2);
        Eigen::Vector2d normal = x.normalized();
        Eigen::Vector2d ipt(ipt1, ipt2);
        double r = (x - ipt).norm();
        // evaluate Neumann data of fundamental solution using computed distance
        return k * 1i * HankelH1p(0, k * r) * (x - ipt).normalized().dot(normal) * 0.25;
    }

    complex_t t_coeff(int n,
                      double eps,
                      const complex_t &k,
                      double n_i) {
        // simplify parameter
        complex_t k_eps = k * eps;
        double lambda = sqrt(n_i);
        // compute transmission coefficient that satisfies transmission conditions
        complex_t result = (2. * 1i / (M_PI * k_eps)) / (HankelH1p(n, k_eps) * BesselJ(n, lambda * k_eps) -
                                                         lambda * BesselJp(n, lambda * k_eps) * HankelH1(n, k_eps));
        return result;
    };

    complex_t r_coeff(int n,
                      double eps,
                      const complex_t &k,
                      double n_i) {
        // simplify parameter
        complex_t k_eps = k * eps;
        double lambda = sqrt(n_i);
        // compute reflection coefficient that satisfies transmission conditions
        return -(HankelH1p(n, k_eps) * BesselJ(n, lambda * k_eps) - lambda * BesselJp(n, lambda * k_eps) * HankelH1(n, k_eps)).real() /
               (HankelH1p(n, k_eps) * BesselJ(n, lambda * k_eps) - lambda * BesselJp(n, lambda * k_eps) * HankelH1(n, k_eps));
    }

    complex_t u_i(double x1,
                  double x2,
                  int l,
                  double *a_n,
                  const complex_t &k) {
        // simplify parameters
        complex_t result = complex_t(0.0, 0.0);
        double r = sqrt(x1 * x1 + x2 * x2);
        double eta = atan2(x2 / r, x1 / r);
        if (eta < 0) eta += 2*M_PI;
        // evaluate incoming wave as series expansion
        for (int i = 0; i < 2 * l + 1; i++) {
            result += a_n[i] * BesselJ(i - l, k * r) * complex_t(cos((i - l) * eta), sin((i - l) * eta));
        }
        return result;
    }

    complex_t u_s(double x1,
                  double x2,
                  int l,
                  double eps,
                  double *a_n,
                  const complex_t &k,
                  double n_i) {
        // simplify parameters
        complex_t result = complex_t(0.0, 0.0);
        double r = sqrt(x1 * x1 + x2 * x2);
        double eta = atan2(x2 / r, x1 / r);
        if (eta < 0) eta += 2*M_PI;
        // evaluate scattered wave as series expansion
        for (int i = 0; i < 2 * l + 1; i++) {
            result += a_n[i] * r_coeff(i - l, eps, k, n_i) * HankelH1(i - l, k * r) *
                      complex_t(cos((i - l) * eta), sin((i - l) * eta));
        }
        return result;
    }

    complex_t u_t(double x1,
                  double x2,
                  int l,
                  double eps,
                  double *a_n,
                  const complex_t &k,
                  double n_i) {
        // simplify parameters
        complex_t result(0.0, 0.0);
        double lambda = sqrt(n_i);
        double r = sqrt(x1 * x1 + x2 * x2);
        double eta = atan2(x2 / r, x1 / r);
        if (eta < 0) eta += 2*M_PI;
        // evaluate transmitted wave as series expansion
        for (int i = 0; i < 2 * l + 1; i++) {
            result += a_n[i] * t_coeff(i - l, eps, k, n_i) * BesselJ(i - l, lambda * k * r) *
                      complex_t(cos((i - l) * eta), sin((i - l) * eta));
        }
        return result;
    }

    Eigen::Vector2cd u_i_del(double x1,
                             double x2,
                             int l,
                             double *a_n,
                             const complex_t &k) {
        // simplify parameters
        Eigen::Vector2cd result(0, 0);
        double r = sqrt(x1 * x1 + x2 * x2);
        if (r == 0) return result;
        double eta = atan2(x2 / r, x1 / r);
        Eigen::Vector2d e_r;
        e_r << cos(eta), sin(eta);
        Eigen::Vector2d e_eta;
        e_eta << -sin(eta), cos(eta);
        // evaluate Neumann data of incoming wave as series expansion
        for (int i = 0; i < 2 * l + 1; i++) {
            result += a_n[i] * complex_t(cos((i - l) * eta), sin((i - l) * eta)) *
                      (k * BesselJp(i - l, k * r) * e_r + 1i * double(i - l) / r * e_eta);
        }
        return result;
    }

    complex_t u_i_neu(double x1,
                      double x2,
                      int l,
                      double *a_n,
                      const complex_t &k) {
        // simplify parameters
        complex_t result;
        double r = sqrt(x1 * x1 + x2 * x2);
        double eta = atan2(x2 / r, x1 / r);
        Eigen::Vector2d e_r;
        e_r << cos(eta), sin(eta);
        Eigen::Vector2d e_eta;
        e_eta << -sin(eta), cos(eta);
        // evaluate Neumann data of incoming wave as series expansion
        for (int i = 0; i < 2 * l + 1; i++) {
            result += a_n[i] * complex_t(cos((i - l) * eta), sin((i - l) * eta)) *
                      k * BesselJp(i - l, k * r);
        }
        return result;
    }

    complex_t u_s_neu(double x1,
                      double x2,
                      int l,
                      double eps,
                      double *a_n,
                      const complex_t &k,
                      double n_i) {
        // simplify parameters
        complex_t result;
        double r = sqrt(x1 * x1 + x2 * x2);
        double eta = atan2(x2 / r, x1 / r);
        if (eta < 0) eta += 2*M_PI;
        Eigen::Vector2d e_r;
        e_r << cos(eta), sin(eta);
        Eigen::Vector2d e_eta;
        e_eta << -sin(eta), cos(eta);
        // evaluate Neumann data of scattered wave as series expansion
        for (int i = 0; i < 2 * l + 1; i++) {
            result += a_n[i] * r_coeff(i - l, eps, k, n_i) * complex_t(cos((i - l) * eta), sin((i - l) * eta)) *
                      k * HankelH1p(i - l, k * r);
        }
        return result;
    }

    complex_t u_t_neu(double x1,
                      double x2,
                      int l,
                      double eps,
                      double *a_n,
                      const complex_t &k,
                      double n_i) {
        // simplify parameters
        complex_t result;
        double lambda = sqrt(n_i);
        double r = sqrt(x1 * x1 + x2 * x2);
        double eta = atan2(x2 / r, x1 / r);
        Eigen::Vector2d e_r;
        e_r << cos(eta), sin(eta);
        Eigen::Vector2d e_eta;
        e_eta << -sin(eta), cos(eta);
        // evaluate Neumann data of transmitted wave as series expansion
        for (int i = 0; i < 2 * l + 1; i++) {
            result += (a_n[i] * t_coeff(i - l, eps, k, n_i) *
                       complex_t(cos((i - l) * eta), sin((i - l) * eta)) *
                       lambda * k * BesselJp(i - l, lambda * k * r));
        }
        return result;
    }
}
