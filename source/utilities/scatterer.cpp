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
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <numeric>

#define MAX_ITER_ALPHA 5000

using namespace std;

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

    bool read_polygon(const string &fname, Eigen::VectorXd &x, Eigen::VectorXd &y, double scale) {
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

    ParametrizedMesh panelize(const Eigen::VectorXd &x, const Eigen::VectorXd &y, unsigned N, double refinement_factor, bool print_info) {
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
        if (print_info) {
#ifdef CMDL
            std::cout << "Working with " << M << " panels" << std::endl;
            double min_length = res[0]->length(), max_length = min_length, avg_length = 0., s = 0.;
            for (i = 0; i < M; ++i) {
                double len = res[i]->length();
                if (len < min_length)
                    min_length = len;
                if (len > max_length)
                    max_length = len;
                avg_length += len;
            }
            avg_length /= double(M);
            for (i = 0; i < M; ++i) {
                s += std::pow(avg_length - res[i]->length(), 2);
            }
            s = std::sqrt(s);
            std::cout << "Panel length (min, max, mean, variability): "
                      << min_length << ", " << max_length << ", " << avg_length << ", " << s / avg_length << " %" << std::endl;
#endif
        }
        return ParametrizedMesh(res);
    }

    bool inside_poly(const ParametrizedMesh &mesh, const Eigen::Vector2d &p) {
        bool c = false;
        unsigned i, n = mesh.getNumPanels();
        for (i = 0; i < n; ++i) {
            const auto &panel = *mesh.getPanels()[i];
            double x1 = panel[0](0), x2 = panel[1](0), y1 = panel[0](1), y2 = panel[1](1);
            if ((((y1 <= p(1)) && (p(1) < y2)) || ((y2 <= p(1)) && (p(1) < y1))) &&
                (p(0) < (x2 - x1) * (p(1) - y1) / (y2 - y1) + x1))
                c = !c;
        }
        return c;
    }

    double panel_distance(const ParametrizedMesh &mesh, unsigned i, const Eigen::Vector2d &p, double *t0) {
        assert(i < mesh.getNumPanels());
        const auto &panel = *mesh.getPanels()[i];
        Eigen::Vector2d p1 = panel[0], p2 = panel[1], h = p2 - p1;
        double t = fmax(0, fmin(1, ((p - p1).transpose() * h)(0) / h.squaredNorm()));
        Eigen::Vector2d proj = p1 + t * h;
        double d = (p - proj).norm();
        if (t0 != NULL)
            *t0 = t;
        return d;
    }

    double distance(const ParametrizedMesh &mesh, const Eigen::Vector2d &p, unsigned *i, double *t0) {
        unsigned N = mesh.getNumPanels();
        double min_d = -1., d, t;
        for (unsigned j = 0; j < N; ++j) {
            d = panel_distance(mesh, j, p, &t);
            if (min_d < 0. || d < min_d) {
                min_d = d;
                if (t0 != NULL)
                    *t0 = t;
                if (i != NULL)
                    *i = j;
            }
        }
        return min_d;
    }
}
