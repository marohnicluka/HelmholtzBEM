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

using namespace std;

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

bool read_polygon(const string &fname, Eigen::VectorXd &x, Eigen::VectorXd &y) {
    Eigen::ArrayXXd ar;
    if (!read_array(fname, ar) || ar.cols() != 2)
        return false;
    x = ar.col(0);
    y = ar.col(1);
    return true;
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


PanelVector make_scatterer(const Eigen::VectorXd &x, const Eigen::VectorXd &y, unsigned N) {
    PanelVector res;
    size_t n = x.size(), i, j;
    Eigen::Vector2d p1, p2;
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
        if (d(i) < 1e-9)
            std::cout << "Warning: two nearly identical vertices detected! Distance: " << d(i) << std::endl;
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
#ifdef CMDL
    std::cout << "Working with " << M << " panels" << std::endl
              << "Mean panel size: " << (d.array() / k.array()).mean() << ", variability: " << obj(0.0) * 100.0 << " %" << std::endl;
#endif
    res.reserve(M);
    for (i = 0; i < n; ++i) {
        p1(0) = x(i);
        p1(1) = y(i);
        p2(0) = x(i < n - 1 ? i + 1 : 0);
        p2(1) = y(i < n - 1 ? i + 1 : 0);
        d = (p2 - p1) / k(i);
        for (j = 0; j < (size_t)k(i); ++j) {
            res.push_back(std::make_shared<ParametrizedLine>(p1 + double(j) * d, p1 + double(j+1) * d));
        }
    }
    return res;
}

bool inside_poly(const PanelVector &panels, const Eigen::Vector2d &p) {
    unsigned i, j;
    bool c = false;
    size_t n = panels.size();
    for (i = 0, j = n-1; i < n; j = i++) {
        const auto &panel_i = *panels[i], &panel_j = *panels[j];
        double xi = panel_i[0](0), xj = panel_j[0](0);
        double yi = panel_i[0](1), yj = panel_j[0](1);
        if ((((yi <= p(1)) && (p(1) < yj)) || ((yj <= p(1)) && (p(1) < yi))) &&
            (p(0) < (xj - xi) * (p(1) - yi) / (yj - yi) + xi))
            c = !c;
    }
    return c;
}

bool on_the_boundary(const PanelVector &panels, const Eigen::Vector2d &p, unsigned &ind, double &t0, double boundary_threshold) {
    size_t n = panels.size();
    double min_d = -1.;
    for (unsigned i = 0; i < n; ++i) {
        const auto &panel = *panels[i];
        Eigen::Vector2d p1 = panel[0], p2 = panel[1];
        double l2 = (p2-p1).squaredNorm();
        double t = fmax(0, fmin(1, ((p-p1).transpose() * (p2-p1))(0) / l2));
        Eigen::Vector2d proj = p1 + t * (p2 - p1);
        double d = (p-proj).norm();
        if (min_d < 0. || min_d > d) {
            min_d = d;
            ind = i;
            t0 = t;
        }
    }
    return min_d < boundary_threshold;
}

int ppoly(const PanelVector &panels, const Eigen::Vector2d &p, unsigned &ind, double &t, double boundary_threshold) {
    if (on_the_boundary(panels, p, ind, t, boundary_threshold))
        return 0;
    return inside_poly(panels, p) ? 1 : -1;
}
