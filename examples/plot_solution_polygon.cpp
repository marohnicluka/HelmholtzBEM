/**
 * \file plot_solution_polygon.cpp
 * \brief This target builds a script that solves the
 * Helmholtz transmission problem and outputs the gnuplot file.
 * The scatterer is polygonal (loaded from file). The results are
 * written to the <tt>data</tt> directory.
 * The script can be run as follows:
 *
 * <tt>
 *  /path/to/plot_solution_polygon \<scatterer file\> \<incoming wave file\>
 *     \<refraction inside\> \<refraction outside\> \<wavenumber\> \<#panels\>
 *     \<quadrature order\> \<grid size\> \<lower left x\> \<lower left y\>
 *     \<upper right x\> \<upper right x\> \<mode\> \<intensity\>.
 * </tt>
 *
 * This target produces a gnuplot script which plots the solution
 * in the specified rectangle.
 *
 * The user will be updated through the command line about the
 * progress of the algorithm if <tt>CMDL</tt> is set.
 *
 * This File is a part of the HelmholtzTransmissionProblemBEM library.
 *
 * (c) 2024 Luka Marohnić
 */

#include <complex>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
#include <execution>
#include <string>
#include <sstream>
#include <gsl/gsl_spline.h>
#include "parametrized_line.hpp"
#include "gen_sol_op.hpp"
#include "gen_sol.hpp"
#include "continuous_space.hpp"
#include "discontinuous_space.hpp"
#include "mass_matrix.hpp"
#include "cbessel.hpp"
#include "incoming.hpp"
#include "scatterer.hpp"

using namespace std::chrono;

typedef std::complex<double> complex_t;
complex_t ii = complex_t(0,1.);

int main(int argc, char** argv) {

    // check whether we have the correct number of input arguments
    if (argc < 15) {
        std::cerr << "Error: too few input arguments" << std::endl;
        return 1;
    }
    if (argc > 15) {
        std::cerr << "Error: too many input arguments" << std::endl;
        return 1;
    }

    // read filenames for obstacle and incoming wave
    string fname_scatterer = argv[1], fname_incoming = argv[2];

    // define refraction index, wavenumber, order of quadrature and grid size
    double c_i = atof(argv[3]);
    double c_o = atof(argv[4]);
    complex_t k = atof(argv[5]);
    bool k_real_positive = k.imag() == 0. && k.real() > 0.;

    // define grid size and order of quadrature rule used to compute matrix
    // entries and which singular value to evaluate
    unsigned order = atoi(argv[7]);
    unsigned grid_size = atoi(argv[8]);

    // drawing area
    Eigen::Vector2d lower_left_corner(atof(argv[9]), atof(argv[10]));
    Eigen::Vector2d upper_right_corner(atof(argv[11]), atof(argv[12]));

    // drawing mode
    int mode = atoi(argv[13]);
    bool add_u_inc = false;
    if (mode > 2) {
        mode -= 3;
        add_u_inc = true;
    }
    bool animate = mode == 1;
    double intensity = atof(argv[14]);

    // read incoming wave from file
    incoming::wave u_inc_spec;
    if (!incoming::load(fname_incoming, u_inc_spec)) {
        std::cerr << "Error: failed to read incoming wave from file" << std::endl;
        return 1;
    }

    // read polygonal scatterer from file
    Eigen::VectorXd poly_x, poly_y;
    if (!read_polygon(fname_scatterer, poly_x, poly_y)) {
        std::cerr << "Error: failed to read scatterer from file" << std::endl;
        return 1;
    }

    // construction of a ParametrizedMesh object from the vector of panels
    unsigned Npanels;
    if (strlen(argv[6]) > 1 && argv[6][1] == '.') {
        double f = atof(argv[6]);
        Npanels = auto_num_panels(poly_x, poly_y, f);
    } else Npanels = atoi(argv[6]);
    using PanelVector = PanelVector;
    PanelVector panels = make_scatterer(poly_x, poly_y, Npanels, 0.5);
    ParametrizedMesh mesh(panels);
    unsigned numpanels = panels.size();
    for (const auto &p : panels) {
        if (p->length() > M_PI / (5. * k.real())) {
#ifdef CMDL
            std::cout << "Warning: the number of panels may be too small" << std::endl;
#endif
            break;
        }
    }

    // generate output filename with set parameters
    std::string base_name = "file_plot_solution_polygon_";
    std::string suffix = ".dat";
    std::string sep = "_";
    std::string fname = base_name;
    fname.append(argv[3]).append(sep).append(argv[5]).append(sep).append(argv[8]);

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
                << "set size ratio -1" << std::endl;
    if (animate)
        file_script << "set term png" << std::endl;

    // create the incoming wave function and its gradient
    auto u_inc = [&](const Eigen::Vector2d &x) {
        return incoming::compute(u_inc_spec, x, k);
    };
    auto u_inc_del = [&](const Eigen::Vector2d &x) {
        return incoming::compute_del(u_inc_spec, x, k);
    };

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
    SolutionsOperator sol_op(builder_data, false);

    // assemble solutions operator matrix
#ifdef CMDL
    std::cout << "Assembling solutions operator matrix..." << std::endl;
#endif
    Eigen::MatrixXcd A, B, K_i, K_o, V_i, V_o, W_i, W_o, Id;
    Eigen::MatrixXd M;
    A.resize(2 * numpanels, 2 * numpanels);
    B.resize(2 * numpanels, 2 * numpanels);
    Id.setIdentity(2 * numpanels, 2 * numpanels);
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

    // create traces
#ifdef CMDL
    std::cout << "Computing boundary traces..." << std::endl;
#endif
    // analytic solution data
    double *a_n = new double[5];
    for (unsigned i = 0; i < 5; ++i) {
        a_n[i] = 1.;
    }
    Eigen::VectorXcd gamma_o_uinc, gamma_u;
    gamma_o_uinc.resize(2 * numpanels);
    double *t = new double[numpanels + 1];
    double t0 = 0., t1;
    for (unsigned i = 0; i < numpanels; ++i) {
        const auto &p = *panels[i];
        Eigen::Vector2d x = p(0), tangent = p.Derivative(0), normal_o;
        double plen_2 = p.length() / 2.;
        t0 += plen_2;
        t[i] = t0;
        t0 += plen_2;
        // normal vector
        normal_o << tangent(1), -tangent(0);
        gamma_o_uinc(i) = u_inc(x);
        gamma_o_uinc(numpanels + i) = u_inc_del(x).dot(normal_o.normalized());
    }
    double boundary_length = t0;
    t[numpanels] = boundary_length + panels[0]->length() / 2.;
    gamma_u = A.colPivHouseholderQr().solve(B * gamma_o_uinc);
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
    Eigen::MatrixXcd S;
    S.resize(grid_size, grid_size);
    S.setZero();
    QuadRule qr = getGaussQR(order, 0., 1.);
    size_t n = qr.n;
    Eigen::ArrayXXcd Y, Z, H0, H1, Trace_i_D, Trace_i_N, Trace_o_D, Trace_o_N, N, G;
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
    D.resize(numpanels, n);
    N.resize(numpanels, n);
    G.resize(numpanels, n);
    t0 = 0.;
    for (unsigned i = 0; i < numpanels; ++i) {
        const auto &p = *panels[i];
        double plen = p.length();
        for (unsigned j = 0; j < n; ++j) {
            Eigen::Vector2d y = p[qr.x(j)], tangent = p.Derivative_01(qr.x(j)), normal_o;
            Y(i, j) = -y(0) - ii * y(1);
            D(i, j) = tangent.norm() * qr.w(j);
            // normal vector
            normal_o << tangent(1), -tangent(0);
            normal_o.normalize();
            N(i, j) = normal_o(0) + ii * normal_o(1);
            t1 = t0 + qr.x(j) * plen;
            if (t1 < t[0])
                t1 += boundary_length;
            Trace_i_D(i, j) = gsl_spline_eval(spline_i_D_real, t1, acc) + ii * gsl_spline_eval(spline_i_D_imag, t1, acc);
            Trace_i_N(i, j) = gsl_spline_eval(spline_i_N_real, t1, acc) + ii * gsl_spline_eval(spline_i_N_imag, t1, acc);
            Trace_o_D(i, j) = Trace_i_D(i, j) - u_inc(y);
            Trace_o_N(i, j) = Trace_i_N(i, j) - u_inc_del(y).dot(normal_o);
        }
        t0 += plen;
    }

    // compute solution
    double x_step = (upper_right_corner(0) - lower_left_corner(0)) / (grid_size - 1.);
    double y_step = (upper_right_corner(1) - lower_left_corner(1)) / (grid_size - 1.);
    double excess;
    complex_t k_sqrt_ci = k * std::sqrt(c_i), k_sqrt_co = k * std::sqrt(c_o), kappa;
    unsigned ind;
    int pos;
    for (unsigned I = 0; I < grid_size; ++I) {
        for (unsigned J = 0; J < grid_size; ++J) {
            Eigen::Vector2d x;
            x << lower_left_corner(0) + I * x_step, lower_left_corner(1) + J * y_step;
            pos = ppoly(panels, x, ind, excess);
            if (pos == 0) { // on the boundary
                t0 = 0.;
                for (unsigned i = 0; i < ind; ++i)
                    t0 += panels[i]->length();
                t0 += excess * panels[ind]->length();
                if (t0 < t[0])
                    t0 += boundary_length;
                S(I, J) = gsl_spline_eval(spline_i_D_real, t0, acc) + ii * gsl_spline_eval(spline_i_D_imag, t0, acc);
            } else { // not on the boundary (pos = 1 if inside, else pos = -1)
                kappa = (pos == 1 ? k_sqrt_ci : k_sqrt_co);
                Z = Y + x(0) + ii * x(1);
                X = Z.cwiseAbs();
                if (k_real_positive)
                    complex_bessel::H1_01(kappa.real() * X, H0, H1);
                else
                    complex_bessel::H1_01_cplx(kappa * X, H0, H1);
                G = (H0 * (pos == 1 ? Trace_i_N : Trace_o_N)
                    - kappa * H1 * (pos == 1 ? Trace_i_D : Trace_o_D) * (Z.real() * N.real() + Z.imag() * N.imag()) / X) * D;
                S(I, J) = double(pos) * ii * 0.25 * G.sum();
                if (add_u_inc && pos == -1)
                    S(I, J) += u_inc(x);
            }
        }
    }

    toc = high_resolution_clock::now();

#ifdef CMDL
    std::cout << "Total time: " << 1e-3 * duration_cast<milliseconds>(toc - tic).count() << " sec" << std::endl;
#endif

    // free spline resources
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
    if (animate) {
        double cb_radius = S.cwiseAbs().maxCoeff() / intensity;
        file_script << "set cbrange [" << -cb_radius << ":" << cb_radius << "]" << std::endl;
    } else {
        double cb_min = (mode == 2 ? 0. : S.real().minCoeff()) / intensity;
        double cb_max = (mode == 2 ? S.cwiseAbs().maxCoeff() : S.real().maxCoeff()) / intensity;
        file_script << "set cbrange [" << cb_min << ":" << cb_max << "]" << std::endl;
    }
    unsigned nv = poly_x.size();
    for (unsigned count = 0; count < (animate ? 25 : 1); ++count) {
        stringstream ss;
        ss << fname;
        if (animate)
            ss << sep << 10 + count;
        string fname_count = ss.str();
        std::ofstream file_out;
        file_out.open("../data/img/" + fname_count + suffix, std::ofstream::out | std::ofstream::trunc);
        file_out.close();
        file_out.open("../data/img/" + fname_count + suffix, std::ios_base::app);
        if (!file_out.is_open()) {
            std::cerr << "Error: failed to open plotted data file for writing" << std::endl;
            return 1;
        }
        for (unsigned I = 0; I < grid_size; ++I) {
            for (unsigned J = 0; J < grid_size; ++J)
                file_out << lower_left_corner(0) + I * x_step << '\t' << lower_left_corner(1) + J * y_step << '\t'
                         << (mode == 2 ? std::abs(S(I, J)) : (S(I, J) * (animate ? exp(-ii * 0.08 * M_PI * double(count)) : 1.)).real())
                         << std::endl;
            file_out << std::endl;
        }
        file_out.close();
        file_script << "set object 1 polygon from ";
        for (unsigned i = 0; i < nv; ++i) {
            file_script << poly_x(i) << ", " << poly_y(i) << " to ";
        }
        file_script << poly_x(0) << ", " << poly_y(0) << " lw 0.5 lc rgb \'#000000\' front" << std::endl;
        if (animate)
            file_script << "set output \"img/" << fname_count << ".png\"" << std::endl;
        file_script << "splot \'img/" << fname_count << suffix << "\'" << std::endl;
    }
    file_script.close();
    return 0;
}
