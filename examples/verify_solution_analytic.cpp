/**
 * \file verify_solution_analytic.cpp
 * \brief This target builds a script that solves the
 * Helmholtz transmission problem in a circle with BesselJ
 * incoming wave and compares the result with the analytic
 * solution.
 * The script can be run as follows:
 *
 * <tt>
 *  /path/to/verify_solution_analytic \<circle radius\> \<bessel order\>
 *     \<refraction inside\> \<refraction outside\> \<wavenumber\> \<#panels\>
 *     \<order of quadrature rule\> \<grid size\>
 * </tt>
 *
 * The user will be updated through the command line about the
 * progress of the algorithm if <tt>CMDL</tt> is set.
 *
 * This file is a part of the HelmholtzTransmissionProblemBEM library.
 *
 * (c) 2024 Luka Marohnić
 */

#include <complex>
#include <numeric>
#include <iostream>
#include <chrono>
#include <execution>
#include <string>
#include <fstream>
#include <gsl/gsl_spline.h>
#include "parametrized_circular_arc.hpp"
#include "gen_sol.hpp"
#include "gen_sol_op.hpp"
#include "continuous_space.hpp"
#include "discontinuous_space.hpp"
#include "mass_matrix.hpp"
#include "cbessel.hpp"
#include "incoming.hpp"

using namespace std::chrono;

typedef std::complex<double> complex_t;
complex_t ii = complex_t(0,1.);

int main(int argc, char** argv) {

    // check whether we have the correct number of input arguments
    if (argc < 9) {
        std::cerr << "Error: too few input arguments" << std::endl;
        return 1;
    }
    if (argc > 9) {
        std::cerr << "Error: too many input arguments" << std::endl;
        return 1;
    }

    // define circle radius, Bessel order, refraction indices and wavenumber
    double radius = atof(argv[1]);
    int bessel_order = atoi(argv[2]);
    double c_i = atof(argv[3]);
    double c_o = atof(argv[4]);
    double k = atof(argv[5]);

    // define number of panels, grid size and order of quadrature rule
    unsigned numpanels = atoi(argv[6]);
    unsigned order = atoi(argv[7]);
    unsigned grid_size = atoi(argv[8]);

    // construction of a ParametrizedMesh object from the vector of panels
    using PanelVector = PanelVector;
    auto origin = Eigen::Vector2d(0,0);
    ParametrizedCircularArc circle(origin, radius, 0, 2*M_PI);
    PanelVector panels = circle.split(numpanels);
    ParametrizedMesh mesh(panels);

    // generate output filename with set parameters
    std::string base_name = "file_plot_solution_analytic_";
    std::string suffix = ".dat";
    std::string sep = "_";
    std::string fname = base_name;
    fname.append(argv[3]).append(sep).append(argv[5]).append(sep).append(argv[6]);

    // open script file
    std::ofstream file_script;
    string script_fname = "../data/" + fname + ".gnuplot";
    file_script.open(script_fname, std::ofstream::out | std::ofstream::trunc);
    file_script.close();
    file_script.open(script_fname, std::ios_base::app);
    if (!file_script.is_open()) {
        std::cerr << "Error: failed to open gnuplot script file for writing" << std::endl;
        return 1;
    }
    file_script << "set view map" << std::endl
                << "load \'parula.pal\'" << std::endl
                //<< "set dgrid3d " << grid_size << ", " << grid_size << ", 1" << std::endl
                << "set pm3d at b" << std::endl
                << "unset key" << std::endl
                << "unset surface" << std::endl
                << "set size square" << std::endl;

    // enable parallelization
    complex_bessel::parallelize(true);
    parallelize_builder(true);

    auto tic = high_resolution_clock::now();

    // create Galerkin matrix builder
    unsigned galerkin_quadrature_order = 11;
    ContinuousSpace<1> cont_space;
    DiscontinuousSpace<0> discont_space;
    BuilderData builder_data(mesh, cont_space, cont_space, galerkin_quadrature_order);
    GalerkinMatrixBuilder builder(builder_data);
    SolutionsOperator sol_op(builder_data, false, false);

    // assemble solutions operator matrix
#ifdef CMDL
    std::cout << "Assembling solutions operator matrix..." << std::endl;
#endif
    Eigen::MatrixXcd A, B, K_i, K_o, V_i, V_o, W_i, W_o;
    Eigen::MatrixXd M;
    A.resize(2 * numpanels, 2 * numpanels);
    B.resize(2 * numpanels, 2 * numpanels);
    builder.assembleAll(k, c_i);
    K_i = builder.getDoubleLayer();
    W_i = builder.getHypersingular();
    V_i = builder.getSingleLayer();
    builder.assembleAll(k, c_o);
    K_o = builder.getDoubleLayer();
    W_o = builder.getHypersingular();
    V_o = builder.getSingleLayer();
    M = mass_matrix::GalerkinMatrix(builder_data.mesh, builder_data.trial_space, builder_data.test_space, builder_data.GaussQR);
    A.block(0, 0, numpanels, numpanels) = M - K_o + K_i;
    A.block(numpanels, 0, numpanels, numpanels) = W_o - W_i;
    A.block(0, numpanels, numpanels, numpanels) = V_o - V_i;
    A.block(numpanels, numpanels, numpanels, numpanels) = (M + K_o - K_i).transpose();
    B.block(0, 0, numpanels, numpanels) = 0.5 * M - K_o;
    B.block(numpanels, 0, numpanels, numpanels) = W_o;
    B.block(0, numpanels, numpanels, numpanels) = V_o;
    B.block(numpanels, numpanels, numpanels, numpanels) = (0.5 * M + K_o).transpose();

    auto toc = high_resolution_clock::now();
#ifdef CMDL
    std::cout << "Assembly time: " << 1e-3 * duration_cast<milliseconds>(toc - tic).count() << " sec" << std::endl;
#endif

    // analytic solution data
    double *a_n = new double[2 * bessel_order + 1];
    for (unsigned i = 0; i < 2 * bessel_order + 1; ++i) {
        a_n[i] = 1.;
    }
    double k_sqrt_ci = k * std::sqrt(c_i), k_sqrt_co = k * std::sqrt(c_o), kappa;

    // compute traces
#ifdef CMDL
    std::cout << "Computing boundary traces..." << std::endl;
#endif
    Eigen::VectorXcd gamma_o_uinc, gamma_u, gamma_u_analytic;
    gamma_o_uinc.resize(2 * numpanels);
    gamma_u_analytic.resize(2 * numpanels);
    double *t = new double[numpanels + 1];
    double t0 = 0., t1;
    for (unsigned i = 0; i < numpanels; ++i) {
        const auto &p = *panels[i];
        Eigen::Vector2d x = p(0);
        double plen_2 = p.length() / 2.;
        t0 += plen_2;
        t[i] = t0;
        t0 += plen_2;
        gamma_o_uinc(i) = sol::u_i(x(0), x(1), bessel_order, a_n, k);
        gamma_o_uinc(numpanels + i) = sol::u_i_neu(x(0), x(1), bessel_order, a_n, k);
        gamma_u_analytic(i) = sol::u_t(x(0), x(1), bessel_order, radius, a_n, k, c_i);
        gamma_u_analytic(numpanels + i) = sol::u_t_neu(x(0), x(1), bessel_order, radius, a_n, k, c_i);
    }
    double boundary_length = t0;
    t[numpanels] = boundary_length + panels[0]->length() / 2.;
    gamma_u = A.colPivHouseholderQr().solve(B * gamma_o_uinc);
    //std::cout << "Relative trace error: " << (gamma_u-gamma_u_analytic).norm() / gamma_u_analytic.norm() << std::endl;
    double *trace_i_D_real = new double[numpanels + 1];
    double *trace_i_D_imag = new double[numpanels + 1];
    double *trace_i_N_real = new double[numpanels + 1];
    double *trace_i_N_imag = new double[numpanels + 1];
    for (unsigned i = 0; i < numpanels; ++i) {
        trace_i_D_real[i] = gamma_u(i).real();
        trace_i_D_imag[i] = gamma_u(i).imag();
        trace_i_N_real[i] = gamma_u(numpanels + i).real();
        trace_i_N_imag[i] = gamma_u(numpanels + i).imag();
    }
    trace_i_D_real[numpanels] = trace_i_D_real[0];
    trace_i_D_imag[numpanels] = trace_i_D_imag[0];
    trace_i_N_real[numpanels] = trace_i_N_real[0];
    trace_i_N_imag[numpanels] = trace_i_N_imag[0];
    // interpolate traces
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline_i_D_real = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
    gsl_spline *spline_i_D_imag = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
    gsl_spline_init(spline_i_D_real, t, trace_i_D_real, numpanels + 1);
    gsl_spline_init(spline_i_D_imag, t, trace_i_D_imag, numpanels + 1);
    gsl_spline *spline_i_N_real = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
    gsl_spline *spline_i_N_imag = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
    gsl_spline_init(spline_i_N_real, t, trace_i_N_real, numpanels + 1);
    gsl_spline_init(spline_i_N_imag, t, trace_i_N_imag, numpanels + 1);

    // prepare for lifting the solution from traces
#ifdef CMDL
    std::cout << "Lifting solution from traces..." << std::endl;
#endif
    Eigen::MatrixXcd S, S_a;
    S.resize(grid_size, grid_size);
    S.setZero();
    S_a.resize(grid_size, grid_size);
    S_a.setZero();
    QuadRule qr = getGaussQR(order, 0., 1.);
    size_t n = qr.n;
    Eigen::ArrayXXcd Y, Z, H0, H1, Trace_i_D, Trace_i_N, Trace_o_D, Trace_o_N, T_i_D, T_i_N, T_o_D, T_o_N, N, G;
    Eigen::ArrayXXd X, D;
    X.resize(numpanels, n);
    Y.resize(numpanels, n);
    Z.resize(numpanels, n);
    H0.resize(numpanels, n);
    H1.resize(numpanels, n);
    Trace_i_D.resize(numpanels, n);
    Trace_i_N.resize(numpanels, n);
    Trace_o_D.resize(numpanels, n);
    Trace_o_N.resize(numpanels, n);
    T_i_D.resize(numpanels, n);
    T_i_N.resize(numpanels, n);
    T_o_D.resize(numpanels, n);
    T_o_N.resize(numpanels, n);
    D.resize(numpanels, n);
    N.resize(numpanels, n);
    G.resize(numpanels, n);
    t0 = 0.;
    for (unsigned i = 0; i < numpanels; ++i) {
        const auto &p = *panels[i];
        double plen = p.length();
        for (unsigned j = 0; j < n; ++j) {
            Eigen::Vector2d y = p[qr.x(j)], tangent = p.Derivative_01(qr.x(j)), normal_o;
            T_i_D(i, j) = sol::u_t(y(0), y(1), bessel_order, radius, a_n, k, c_i);
            T_i_N(i, j) = sol::u_t_neu(y(0), y(1), bessel_order, radius, a_n, k, c_i);
            T_o_D(i, j) = sol::u_s(y(0), y(1), bessel_order, radius, a_n, k, c_i);
            T_o_N(i, j) = sol::u_s_neu(y(0), y(1), bessel_order, radius, a_n, k, c_i);
            Y(i, j) = -y(0) - ii * y(1);
            D(i, j) = tangent.norm() * qr.w(j);
            // outward normal vector
            normal_o << tangent(1), -tangent(0);
            normal_o.normalize();
            N(i, j) = normal_o(0) + ii * normal_o(1);
            t1 = t0 + qr.x(j) * plen;
            if (t1 < t[0])
                t1 += boundary_length;
            Trace_i_D(i, j) = gsl_spline_eval(spline_i_D_real, t1, acc) + ii * gsl_spline_eval(spline_i_D_imag, t1, acc);
            Trace_i_N(i, j) = gsl_spline_eval(spline_i_N_real, t1, acc) + ii * gsl_spline_eval(spline_i_N_imag, t1, acc);
            Trace_o_D(i, j) = Trace_i_D(i, j) - sol::u_i(y(0), y(1), bessel_order, a_n, k);
            Trace_o_N(i, j) = Trace_i_N(i, j) - sol::u_i_neu(y(0), y(1), bessel_order, a_n, k);
        }
        t0 += plen;
    }
    std::cout << "Traces error: " << ((Trace_i_D - T_i_D).matrix().norm() / T_i_D.matrix().norm() +
                                      (Trace_o_D - T_o_D).matrix().norm() / T_o_D.matrix().norm() +
                                      (Trace_i_N - T_i_N).matrix().norm() / T_i_N.matrix().norm() +
                                      (Trace_o_N - T_o_N).matrix().norm() / T_o_N.matrix().norm()) * 0.25 << std::endl;

    // compute solution in [-1,1]^2
    double step = 2. / (grid_size - 1.);
    for (unsigned I = 0; I < grid_size; ++I) {
        for (unsigned J = 0; J < grid_size; ++J) {
            Eigen::Vector2d x;
            x << -1. + I * step, -1. + J * step;
            double r = x.norm();
            int pos = r < radius ? 1 : -1;
            kappa = (pos == 1 ? k_sqrt_ci : k_sqrt_co);
            Z = Y + x(0) + ii * x(1);
            X = Z.cwiseAbs();
            complex_bessel::H1_01(kappa * X, H0, H1);
            G = (H0 * (pos == 1 ? Trace_i_N : Trace_o_N)
                - kappa * H1 * (pos == 1 ? Trace_i_D : Trace_o_D) * (Z.real() * N.real() + Z.imag() * N.imag()) / X) * D;
            S(I, J) = double(pos) * ii * 0.25 * G.sum();
            S_a(I, J) = pos == 1 ? sol::u_t(x(0), x(1), bessel_order, radius, a_n, k, c_i)
                                 : sol::u_s(x(0), x(1), bessel_order, radius, a_n, k, c_i);
        }
    }

    double sol_err = (S - S_a).norm() / S_a.norm();
    std::cout << "Solution error: " << sol_err << std::endl;
    std::cout << "Max relative error: " << ((S - S_a).array().cwiseAbs() / S_a.array().cwiseAbs()).matrix().maxCoeff() << std::endl;

    //S = (S - S_a).cwiseAbs() / S_a.norm();

    toc = high_resolution_clock::now();

#ifdef CMDL
    std::cout << "Total time: " << 1e-3 * duration_cast<milliseconds>(toc - tic).count() << " sec" << std::endl;
#endif

    // free resources
    delete[] a_n;
    delete[] t;
    delete[] trace_i_D_real;
    delete[] trace_i_D_imag;
    delete[] trace_i_N_real;
    delete[] trace_i_N_imag;
    gsl_spline_free(spline_i_D_real);
    gsl_spline_free(spline_i_D_imag);
    gsl_spline_free(spline_i_N_real);
    gsl_spline_free(spline_i_N_imag);
    gsl_interp_accel_free(acc);

    // output results to file
#ifdef CMDL
    std::cout << "Writing results to file..." << std::endl;
#endif
    std::ofstream file_out;
    file_out.open("../data/img/" + fname + suffix, std::ofstream::out | std::ofstream::trunc);
    file_out.close();
    file_out.open("../data/img/" + fname + suffix, std::ios_base::app);
    if (!file_out.is_open()) {
        std::cerr << "Error: failed to open plotted data file for writing" << std::endl;
        return 1;
    }
    for (unsigned I = 0; I < grid_size; ++I) {
        for (unsigned J = 0; J < grid_size; ++J)
            file_out << -1. + I * step << '\t' << -1. + J * step << '\t' << S(I, J).real() << std::endl;
        file_out << std::endl;
    }
    file_out.close();
    //file_script << "set cbrange [" << 0 << ":" << S.real().maxCoeff() << "]" << std::endl;
    file_script << "splot \'img/" << fname << suffix << "\'" << std::endl;
    file_script.close();
    return 0;
}
