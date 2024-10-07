/**
 * \file scatterer.cpp
 *
 * \brief This file contains the implementation of
 * utility functions for obstacle objects.
 *
 * (c) 2024 Luka Marohnić
 */

#include "scatterer.hpp"
#include "parametrized_line.hpp"
#include "find_roots.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <execution>
#include <chrono>
#include <tuple>
#include <boost/math/quadrature/gauss_kronrod.hpp>

#define MAX_ITER_ALPHA 5000

using namespace std;
using namespace std::chrono;

namespace scatterer {

    bool read_array(const std::string &fname, Eigen::ArrayXXd &res) {
        vector<vector<double> > result;
        ifstream input(fname);
        string s, num;
        size_t n = 0, m = 0;
        double d;
        while (getline(input, s)) {
            for (size_t i = s.size(); i-->0;) {
                if (isspace(s[i]))
                    s.erase(s.begin() + i);
            }
            if (s.empty())
                continue;
            vector<double> row;
            istringstream iss(s);
            while (getline(iss, num, ',')) {
                try {
                    d = stod(num);
                } catch (exception &e) {
                    return false;
                }
                row.push_back(d);
            }
            result.push_back(row);
            if (m == 0)
                m = row.size();
            else if (row.size() < m)
                row.resize(m, 0.);
            else {
                m = row.size();
                for (size_t i = 0; i < n; ++i)
                    result[i].resize(m, 0.);
            }
            ++n;
        }
        if (n == 0)
            return false;
        res.resize(n, m);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j)
                res(i, j) = result[i][j];
        }
        return true;
    }

    bool read(const string &fname, Eigen::VectorXd &x, Eigen::VectorXd &y, double scale) {
        Eigen::ArrayXXd ar;
        if (!read_array(fname, ar) || ar.cols() != 2)
            return false;
        x = scale * ar.col(0);
        y = scale * ar.col(1);
        return true;
    }

    double length(const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
        int n = x.size();
        double L = 0.;
        Eigen::Vector2d v;
        for (int i = 0; i < n; ++i) {
            v << x((i + 1) % n) - x(i), y((i + 1) % n) - y(i);
            L += v.norm();
        }
        return L;
    }

    unsigned int auto_num_panels(const Eigen::VectorXd& x, const Eigen::VectorXd& y, double f) {
        size_t n = x.size(), i;
        if (n == 0)
            return 0;
        double min_s = __DBL_MAX__, d = 0.0;
        Eigen::Vector2d p1, p2;
        for (i = 0; i < n; ++i) {
            p1(0) = x(i);
            p1(1) = y(i);
            p2(0) = x(i < n - 1 ? i + 1 : 0);
            p2(1) = y(i < n - 1 ? i + 1 : 0);
            double l = (p2 - p1).norm();
            d += l;
            if (l < min_s)
                min_s = l;
        }
        return (unsigned int)std::round(d / (min_s * f));
    }

    PanelVector make_N_polygonal_panels(const Eigen::VectorXd &x, const Eigen::VectorXd &y, unsigned N) {
        assert(x.size() > 0 && x.size() == y.size());
        PanelVector res;
        res.reserve(N);
        unsigned n = x.size(), i, cs = 0;
        Eigen::Vector2d p, q;
        Eigen::VectorXd m(n);
        Eigen::VectorXd d(n);
        double len = length(x, y), alpha = double(N) / len;
        for (i = 0; i < n; ++i) {
            p << x(i), y(i);
            q << x(i == n - 1 ? 0 : i + 1), y(i == n - 1 ? 0 : i + 1);
            cs += (m(i) = std::round(alpha * (d(i) = (p - q).norm())));
        }
        while (cs != N) {
            int k = cs > N ? -1 : 1;
            double min_var = std::numeric_limits<double>::max(), var, mean;
            size_t i_min;
            for (i = 0; i < n; ++i) {
                m(i) += k;
                auto a = d.array() / m.array();
                mean = a.mean();
                var = (a - mean).square().mean();
                if (var < min_var) {
                    min_var = var;
                    i_min = i;
                }
                m(i) -= k;
            }
            m(i_min) += k;
            cs += k;
        }
        assert(cs == N);
        for (i = 0; i < n; ++i) {
            p << x(i), y(i);
            q << x(i == n - 1 ? 0 : i + 1), y(i == n - 1 ? 0 : i + 1);
            auto panel = ParametrizedLine(p, q);
            PanelVector panels = panel.split(m[i]);
            res.insert(res.end(), panels.begin(), panels.end());
        }
        return res;
    }

    PanelVector make_polygonal_panels(const Eigen::VectorXd &x, const Eigen::VectorXd &y, unsigned N, double refinement_factor) {
        PanelVector res;
        unsigned n = x.size();
        int i, j;
        double eps = numeric_limits<double>::epsilon(), eps2 = 1e-3;
        Eigen::Vector2d p0, p1, p2, p3;
        Eigen::VectorXd d, k, b;
        Eigen::MatrixXd E;
        E.setZero(n, n);
        d.resize(n);
        b.resize(n);
        for (i = 0; i < n; ++i) {
            p1(0) = x(i);
            p1(1) = y(i);
            p2(0) = x(i < n - 1 ? i + 1 : 0);
            p2(1) = y(i < n - 1 ? i + 1 : 0);
            d(i) = (p2 - p1).norm();
    #ifdef CMDL
            if (d(i) < eps)
                std::cerr << "Warning: two nearly identical vertices detected! Distance: " << d(i) << std::endl;
    #endif
            if (i < n - 1)
                E(i, i) = 1.0 / d(i);
            if (i > 0)
                E(i-1, i) = -1.0 / d(i);
            E(n-1, i) = 1.0;
            b(i) = i < n - 1 ? 0.0 : double(N);
        }
        k = E.fullPivLu().solve(b);
        std::vector<std::pair<double, size_t> > dd(n);
        for (i = 0; i < n; ++i)
            dd[i] = std::make_pair(k(i), i);
        std::sort(dd.begin(), dd.end());
        std::reverse(dd.begin(), dd.end());
        auto obj = [&](double send) {
            double mu = (d.array() / k.array()).mean();
            double sigma = std::sqrt((d.array() / k.array() - mu).square().sum() / double(n));
            return sigma / mu + send * std::abs(double(N) / k.sum() - 1.0);
        };
        for (i = 0; i < n; ++i) {
            j = dd[i].second;
            k(j) = std::floor(k(j));
            double f1 = obj(1.0);
            k(j) += 1.0;
            double f2 = obj(1.0);
            if (f1 == f1 && f1 < f2)
                k(j) -= 1.0;
        }
        unsigned M = (unsigned int)k.sum();
        res.reserve(M);
        for (i = 0; i < n; ++i) {
            p0(0) = i == 0 ? x(n - 1) : x(i - 1);
            p0(1) = i == 0 ? y(n - 1) : y(i - 1);
            p1(0) = x(i);
            p1(1) = y(i);
            p2(0) = x(i < n - 1 ? i + 1 : 0);
            p2(1) = y(i < n - 1 ? i + 1 : 0);
            p3(0) = x(i < n - 2 ? i + 2 : i - n + 2);
            p3(1) = y(i < n - 2 ? i + 2 : i - n + 2);
            Eigen::Vector2d D = (p2 - p1) / k(i);
            int n_i = (int)k(i);
            double f1 = (1. - (p0 - p1).dot(p2 - p1) / ((p0 - p1).norm() * (p2 - p1).norm())) / 2.;
            double f2 = (1. - (p1 - p2).dot(p3 - p2) / ((p1 - p2).norm() * (p3 - p2).norm())) / 2.;
            f1 = refinement_factor + (1. - refinement_factor) * f1;
            f2 = refinement_factor + (1. - refinement_factor) * f2;
            if (refinement_factor >= 1. || n_i < 3) {
                d = D;
                for (j = 0; j < n_i; ++j) {
                    res.push_back(std::make_shared<ParametrizedLine>(p1 + double(j) * d, j + 1 < n_i ? p1 + double(j+1) * d : p2));
                }
            } else {
                int j_best = 0;
                double alpha_best, d1_best, d2_best;
                for (j = 1; j < n_i; ++j) {
                    double alpha, alpha_next = refinement_factor;
                    auto f = [&](double x) {
                        return f1 * std::pow(x, 1-j) + f2 * std::pow(x, j+1-n_i) - x * (f1 + f2) - double(n_i) * (1. - x);
                    };
                    auto f_der = [&](double x) {
                        return double(n_i) - f1 - f2 - f1 * (j - 1.) * std::pow(x, -j) - f2 * (n_i - j - 1.) * std::pow(x, j-n_i);
                    };
                    unsigned iter_count = 0;
                    do {
                        alpha = alpha_next;
                        alpha_next = alpha - f(alpha) / f_der(alpha);
                        ++iter_count;
                    } while (std::abs(alpha_next - alpha) > eps && iter_count <= MAX_ITER_ALPHA);
                    double d1 = f1 / std::pow(alpha, j-1), d2 = f2 / std::pow(alpha, n_i-j-1);
                    if (!j_best || std::abs(d1 - d2) < std::abs(d1_best - d2_best)) {
                        d1_best = d1;
                        d2_best = d2;
                        j_best = j;
                        alpha_best = alpha;
                    }
                }
                Eigen::Vector2d pos = p1;
                for (j = 0; j < n_i; ++j) {
                    if (j < j_best)
                        d = D * d1_best * std::pow(alpha_best, j_best - 1 - j);
                    else
                        d = D * d2_best * std::pow(alpha_best, j - j_best);
                    res.push_back(std::make_shared<ParametrizedLine>(pos, j + 1 < n_i ? pos + d : p2));
                    //std::cout << pos(0) << " " << pos(1) << ";\n";
                    pos += d;
                }
                if ((pos - p2).norm() / D.norm() > eps2) {
#ifdef CMDL
                    std::cerr << "Warning: panel subdivision failed, using uniform subdivision instead (rel. error: "
                              << (pos - p2).norm() / D.norm() << ")" << std::endl;
#endif
                    for (j = 0; j < n_i; ++j) res.pop_back();
                    pos = p1;
                    for (j = 0; j < n_i; ++j) {
                        res.push_back(std::make_shared<ParametrizedLine>(pos, j + 1 < n_i ? pos + D : p2));
                        pos += D;
                    }
                }
            }
        }
        return res;
    }

    double panel_line_distance(const AbstractParametrizedCurve &panel, const Eigen::Vector2d &p, double *t_min) {
        Eigen::Vector2d p1 = panel[0], p2 = panel[1], h = p2 - p1;
        double t = std::max(0., std::min(1., ((p - p1).transpose() * h)(0) / h.squaredNorm()));
        Eigen::Vector2d proj = p1 + t * h;
        if (t_min) *t_min = t;
        return (p - proj).norm();
    }

    bool inside_mesh(const ParametrizedMesh &mesh, const Eigen::Vector2d &p) {
        bool c = false;
        size_t npanels = mesh.getNumPanels(), n = 0;
        std::vector<size_t> ind(npanels);
        std::iota(ind.begin(), ind.end(), 0);
        std::for_each (ind.cbegin(), ind.cend(), [&](size_t i) {
            const auto &panel = *mesh.getPanels()[i];
            auto p1 = panel[0], p2 = panel[1];
            double x = p(0), y = p(1), x1 = p1(0), x2 = p2(0), y1 = p1(1), y2 = p2(1), d, tc;
            if ((((y1 <= y) && (y < y2)) || ((y2 <= y) && (y < y1))) && (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1))
                c = !c;
            if (panel.isLineSegment() || (p2 - p1).dot(p - p1) <= 0 || (p1 - p2).dot(p - p2) <= 0 || panel_line_distance(panel, p, &tc) > 2.0 * panel.length())
                return;
            /** check whether p is between straight line and panel curve */
            double s = (x2 - x1) / (y2 - y1);
            Eigen::Vector2d cp = p1 + tc * (p2 - p1);
            std::function<double(double)> f = [&](double t) {
                auto pt = panel[t];
                return std::pow(pt(1) + s * (pt(0) - x) - y, 2);
            };
            unsigned ic;
            double t_min = brent_gsl_with_values(0, std::pow(y1 + s * (x1 - x) - y, 2), 1, std::pow(y2 + s * (x2 - x) - y, 2), tc, f(tc), f, 1e-6, ic);
            auto p_min = panel[t_min];
            d = (cp - p_min).squaredNorm();
            if ((p - cp).squaredNorm() < d && (p - p_min).squaredNorm() < d)
                ++n;
        });
        return n % 2 ? !c : c;
    }

    double panel_distance(const AbstractParametrizedCurve &panel, const Eigen::Vector2d &p, double *t0) {
        double t_min, d = panel_line_distance(panel, p, &t_min);
        if (panel.isLineSegment()) {
            if (t0) *t0 = t_min;
            return d;
        }
        /** for a non-linear panel, find the nearest point using Brent minimizer */
        std::function<double(double)> f = [&](double t) { return (p - panel[t]).squaredNorm(); };
        d = f(t_min);
        double d0 = (p - panel[0]).squaredNorm(), d1 = (p - panel[1]).squaredNorm(), t_res;
        double eps_mach = std::numeric_limits<double>::epsilon();
        unsigned ic;
        if (d0 <= d + eps_mach || d1 <= d + eps_mach) {
            d = d0 <= d1 ? d0 : d1;
            t_res = d0 <= d1 ? 0 : 1;
        } else t_res = brent_gsl_with_values(0, d0, 1, d1, t_min, d, f, 1e-6, ic);
        if (std::isnan(t_res)) { /** Brent failed */
#ifdef CMDL
            std::cerr << "Warning: failed to compute panel distance using Brent method, returning initial approximation" << std::endl;
#endif
            t_res = t_min;
        } else d = (p - panel[t_res]).squaredNorm();
        if (t0) *t0 = std::max(0., std::min(1., t_res));
        return std::sqrt(d);
    }

    double distance(const ParametrizedMesh &mesh, const Eigen::Vector2d &p, unsigned *i, double *t0) {
        unsigned N = mesh.getNumPanels();
        std::vector<std::pair<double,double> > dt(N);
        std::vector<size_t> ind(N);
        std::iota(ind.begin(), ind.end(), 0);
        std::transform (std::execution::par, ind.cbegin(), ind.cend(), dt.begin(), [&](size_t j) {
            double d, t;
            d = panel_distance(*mesh.getPanels()[j], p, &t);
            return std::make_pair(d, t);
        });
        double min_d = std::numeric_limits<double>::infinity();
        for (auto j : ind) {
            double d = dt[j].first, t = dt[j].second;
            if (d < min_d) {
                min_d = d;
                if (i != NULL)
                    *i = j;
                if (t0 != NULL)
                    *t0 = t;
            }
        }
        return min_d;
    }

    void print_panels_info(const PanelVector& panels) {
        unsigned n = panels.size();
        std::cout << "Working with " << n << " panels" << std::endl;
        if (panels.empty())
            return;
        double min_length = panels[0]->length(), max_length = min_length, avg_length = 0., s = 0.;
        for (const auto &panel : panels) {
            double len = panel->length();
            if (len < min_length)
                min_length = len;
            if (len > max_length)
                max_length = len;
            avg_length += len;
        }
        avg_length /= double(n);
        for (const auto &panel : panels) {
            s += std::pow(avg_length - panel->length(), 2);
        }
        s = std::sqrt(s);
        std::cout << "Panel length (min, max, mean, stddev): "
                  << min_length << ", " << max_length << ", " << avg_length << ", " << s / avg_length << " %" << std::endl;
    }

    SmoothScatterer::SmoothScatterer(const std::string& fname, double scale)
        : GaussQR(getGaussQR(3, 0., 1.)), tol(1e-8), total_length_(-1.)
    {
        if (!read(fname, x, y, scale))
            throw std::runtime_error("Failed to read data for smooth scatterer");
        n = x.size();
        assert(n > 1 && n == y.size());
        x.conservativeResize(n + 1);
        y.conservativeResize(n + 1);
        x(n) = x(0);
        y(n) = y(0);
        double len = length(x, y); // approx circumference of the scatterer
        t_vert.resize(2 * n + 1);
        t_vert(0) = 0.;
        vert.resize(2 * n);
        Eigen::VectorXd t_vert_read(n + 1);
        t_vert_read(0) = 0.;
        // make spline from input vertices
        for (unsigned i = 0; i < n; ++i) {
            Eigen::Vector2d &p = vert[2 * i], q;
            p << x(i), y(i);
            q << x(i + 1), y(i + 1);
            t_vert_read(i + 1) = t_vert(2 * (i + 1)) = (p - q).norm() / len + t_vert_read(i);
        }
        spline = new ComplexSpline(t_vert_read, x + 1i * y, true);
        // interleave another n vertices
        for (unsigned i = 0; i < n; ++i) {
            double t = t_vert(2 * i + 1) = (t_vert(2 * i) + t_vert(2 * i + 2)) * .5;
            vert[2 * i + 1] = curve_point(t);
        }
        dist.resize(2 * n);
        ind.resize(2 * n);
        std::iota(ind.begin(), ind.end(), 0);
        // define functions for optimization and length computation
        wf = [&](double t) {
            auto z = spline->eval(t), dz = spline->eval_der(t);
            double dx = z.real() - user_point(0), dy = z.imag() - user_point(1);
            return (dx * dz.imag() - dy * dz.real()) / (dx * dx + dy * dy);
        };
        distf = [&](double t) {
            return (user_point - curve_point(t)).norm();
        };
        lf = [&](double t) {
            return std::abs(spline->eval_der(t));
        };
    }

    Eigen::Vector2d SmoothScatterer::curve_point(double t) {
        auto z = spline->eval(t);
        Eigen::Vector2d q;
        q << z.real(), z.imag();
        return q;
    }

    bool SmoothScatterer::is_inside(const Eigen::Vector2d& p) {
        user_point = p;
        double res = 0.;
        std::for_each(ind.cbegin(), ind.cend(), [&](unsigned i) {
            double t0 = t_vert(i), t1 = t_vert(i + 1), d = t1 - t0;
            res += boost::math::quadrature::gauss_kronrod<double, 15>::integrate(wf, t0, t1, 5, 1e-8);
        });
        res *= .5 * M_1_PI;
        return std::abs(res) > 1e-15;
    }

    double SmoothScatterer::distance(const Eigen::Vector2d& p, double* t) {
        std::transform(std::execution::par, ind.cbegin(), ind.cend(), dist.begin(), [&](unsigned i) {
            return (p - vert[i]).norm();
        });
        double t_min, d, dl, dr;
        unsigned iter_count, il, ir;
        user_point = p;
        std::vector<std::pair<double, double> > minima(2 * n, {0, -1});
        std::transform(std::execution::par, ind.cbegin(), ind.cend(), minima.begin(), [&](unsigned i) {
            il = i - 1, ir = (i + 1) % (2 * n + 1);
            d = dist[i], dl = dist[il], dr = dist[ir];
            if (d < dl && d < dr) {
                t_min = brent_gsl_with_values(t_vert(il), dl, t_vert(ir), dr, t_vert(i), d, distf, tol, iter_count);
                d = (p - curve_point(t_min)).norm();
                return std::make_pair(d, t_min);
            }
            return std::make_pair(0., -1.);
        });
        minima.erase(std::remove_if(minima.begin(), minima.end(), [](const std::pair<double, double> &dt) { return dt.second < 0.; }), minima.end());
        if (minima.empty()) // the scatterer is a regular polygon and p is its center
            return (p - vert.front()).norm();
        std::sort(minima.begin(), minima.end());
        if (t != NULL)
            *t = minima.front().second;
        return minima.front().first;
    }

    double SmoothScatterer::total_length() {
        if (total_length_ < 0.)
            total_length_ = std::accumulate(ind.cbegin(), ind.cend(), 0., [&](double sum, unsigned i) {
                return sum + boost::math::quadrature::gauss_kronrod<double, 15>::integrate(lf, t_vert(i), t_vert(i + 1), 5, tol);
            });
        return total_length_;
    }

    PanelVector SmoothScatterer::panelize(unsigned int npanels) {
        auto tic = high_resolution_clock::now();
        // compute the length of the curve
        unsigned max_iter = 1000, ic;
        double len = total_length(), panel_length = len / double(npanels), L, t0 = 0., t1, t2;
        PanelVector panels;
        panels.reserve(npanels);
        for (unsigned i = 1; i < npanels; ++i) {
            t1 = t0 + (1. - t0) / double(npanels - i + 1);
            ic = 0;
            while(ic < max_iter) {
                L = boost::math::quadrature::gauss_kronrod<double, 15>::integrate(lf, t0, t1, 5, tol);
                t2 = t1 + (panel_length - L) / lf(t1);
                if (std::abs(t2 - t1) / t1 < tol)
                    break;
                t1 = t2;
                ++ic;
            }
            panels.push_back(std::make_shared<ParametrizedPolynomial>(*spline, t0, t1, L));
            t0 = t1;
            len -= L;
        }
        panels.push_back(std::make_shared<ParametrizedPolynomial>(*spline, t0, 1., len));
        auto toc = high_resolution_clock::now();
#ifdef CMDL
        double et = 1e-3 * duration_cast<milliseconds>(toc - tic).count();
        std::cout << "Smooth scatterer panelized in " << et << " seconds" << std::endl;
#endif
        return panels;
    }

    void SmoothScatterer::sample_vertices(double refsize, unsigned int resolution, Eigen::VectorXd& vx, Eigen::VectorXd& vy) {
        int N = (int)std::ceil((total_length() * resolution) / refsize);
        vx.resize(N);
        vy.resize(N);
        double dt = 1./double(N);
        for (int i = 0; i < N; ++i) {
            auto p = curve_point(i * dt);
            vx(i) = p(0);
            vy(i) = p(1);
        }
    }

}
