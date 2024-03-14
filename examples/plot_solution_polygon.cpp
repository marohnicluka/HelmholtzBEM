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
 *     \<order of quadrature rule\> \<grid size\> \<animate\>.
 * </tt>
 *
 * This target produces a gnuplot script which plots the solution
 * in [-1,1]^2. The polygonal object should be contained in that set.
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
    if (argc < 10)
        throw std::runtime_error("Too few input arguments!");
    if (argc > 10)
        throw std::runtime_error("Too many input arguments!");

    // read filenames for obstacle and incoming wave
    string fname_scatterer = argv[1], fname_incoming = argv[2];

    // define refraction index, wavenumber, order of quadrature and grid size
    double c_i = atof(argv[3]);
    double c_o = atof(argv[4]);
    double k = atof(argv[5]);

    // define grid size and order of quadrature rule used to compute matrix
    // entries and which singular value to evaluate
    unsigned order = atoi(argv[7]);
    unsigned grid_size = atoi(argv[8]);

    // to animate or not to animate
    bool animate = atoi(argv[9]);

    // read incoming wave from file
    incoming::wave u_inc_spec;
    if (!incoming::load(fname_incoming, u_inc_spec)) {
        std::cerr << "Failed to read incoming wave from file" << std::endl;
        return 1;
    }

    // read polygonal scatterer from file
    Eigen::VectorXd poly_x, poly_y;
    if (!read_polygon(fname_scatterer, poly_x, poly_y)) {
        std::cerr << "Failed to read scatterer from file" << std::endl;
        return 1;
    }

    // construction of a ParametrizedMesh object from the vector of panels
    unsigned Npanels;
    if (strlen(argv[6]) > 1 && argv[6][1] == '.') {
        double f = atof(argv[6]);
        Npanels = auto_num_panels(poly_x, poly_y, f);
    } else Npanels = atoi(argv[6]);
    using PanelVector = PanelVector;
    PanelVector panels = make_scatterer(poly_x, poly_y, Npanels);
    ParametrizedMesh mesh(panels);
    unsigned numpanels = panels.size();

    // create Galerkin matrix builder
    ContinuousSpace<1> cont_space;
    DiscontinuousSpace<0> discont_space;
    BuilderData builder_data(mesh, cont_space, cont_space, 11);
    GalerkinMatrixBuilder builder(builder_data);
    SolutionsOperator sol_op(builder_data, false);

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
    file_script << "set view map" << std::endl
                << "load \'parula.pal\'" << std::endl
                << "set dgrid3d " << grid_size << ", " << grid_size << ", 1" << std::endl
                << "set pm3d at b" << std::endl
                << "unset key" << std::endl
                << "unset surface" << std::endl
                << "set size square" << std::endl;
    if (animate)
        file_script << "set term png" << std::endl;

    // create the incoming wave function and its gradient
    auto u_inc = [&](const Eigen::Vector2d &x) {
        return incoming::compute(u_inc_spec, x, k);
    };
    auto u_inc_del = [&](const Eigen::Vector2d &x) {
        return incoming::compute_del(u_inc_spec, x, k);
    };

    // parallelize Hankel function computation
    complex_bessel::parallelize(true);

    auto tic = high_resolution_clock::now();

    // assemble solutions operator matrix
#ifdef CMDL
    std::cout << "Assembling solutions operator matrix..." << std::endl;
#endif
    Eigen::MatrixXcd A, B, K_i, K_o, V_i, V_o, W_i, W_o, Id;
    Eigen::MatrixXd M;
    A.resize(2 * numpanels, 2 * numpanels);
    B.resize(2 * numpanels, 2 * numpanels);
    Id.setIdentity(2 * numpanels, 2 * numpanels);
    builder.assembleDoubleLayer(k, c_i);
    K_i = builder.getDoubleLayer();
    builder.assembleHypersingular(k, c_i);
    W_i = builder.getHypersingular();
    builder.assembleSingleLayer(k, c_i);
    V_i = builder.getSingleLayer();
    builder.assembleDoubleLayer(k, c_o);
    K_o = builder.getDoubleLayer();
    builder.assembleHypersingular(k, c_o);
    W_o = builder.getHypersingular();
    builder.assembleSingleLayer(k, c_o);
    V_o = builder.getSingleLayer();
    M = mass_matrix::GalerkinMatrix(builder_data.mesh, builder_data.trial_space, builder_data.test_space, builder_data.GaussQR);
    Eigen::MatrixXd MM;
    MM.setZero(2 * numpanels, 2 * numpanels);
    MM.block(0, 0, numpanels, numpanels) = M;
    MM.block(numpanels, numpanels, numpanels, numpanels) = M.transpose().eval();
    Eigen::LDLT<Eigen::MatrixXd> llt(MM);
    A.block(0, 0, numpanels, numpanels) = M - K_o + K_i;
    A.block(numpanels, 0, numpanels, numpanels) = W_o - W_i;
    A.block(0, numpanels, numpanels, numpanels) = V_o - V_i;
    A.block(numpanels, numpanels, numpanels, numpanels) = (M + K_o - K_i).transpose();
    //A = sol_op.project(A);
    // assemble the right-hand-side matrix
    B.block(0, 0, numpanels, numpanels) = 0.5 * M - K_o;
    B.block(numpanels, 0, numpanels, numpanels) = W_o;
    B.block(0, numpanels, numpanels, numpanels) = V_o;
    B.block(numpanels, numpanels, numpanels, numpanels) = (0.5 * M + K_o).transpose();
    //B = sol_op.project(B);

    // create traces
#ifdef CMDL
    std::cout << "Computing boundary traces..." << std::endl;
#endif
    Eigen::VectorXcd gamma_o_uinc, gamma_u;
    Eigen::VectorXd t;
    gamma_o_uinc.resize(2 * numpanels);
    t.resize(numpanels + 1);
    double t0 = 0., t1;
    for (unsigned i = 0; i < numpanels; ++i) {
        const auto &p = *panels[i];
        Eigen::Vector2d x = p(0), tangent = p.Derivative(0), normal_i;
        double plen_2 = p.length() / 2.;
        t0 += plen_2;
        t(i) = t0;
        t0 += plen_2;
        // inward normal vector
        normal_i << -tangent(1), tangent(0);
        gamma_o_uinc(i) = u_inc(x);
        gamma_o_uinc(numpanels + i) = u_inc_del(x).dot(normal_i.normalized());
    }
    double boundary_length = t0;
    t(numpanels) = boundary_length + panels[0]->length() / 2.;
    gamma_u = A.colPivHouseholderQr().solve(B * gamma_o_uinc);
    //gamma_u = llt.matrixL().transpose() * gamma_u;
    Eigen::VectorXcd trace_i_D, trace_i_N, trace_o_D, trace_o_N;
    trace_i_D.resize(numpanels + 1);
    trace_i_N.resize(numpanels + 1);
    trace_i_D.head(numpanels) = gamma_u.head(numpanels);
    trace_i_N.head(numpanels) = gamma_u.tail(numpanels);
    trace_i_D(numpanels) = trace_i_D(0);
    trace_i_N(numpanels) = trace_i_N(0);
    //gamma_u -= gamma_o_uinc;
    trace_o_D.resize(numpanels + 1);
    trace_o_N.resize(numpanels + 1);
    trace_o_D.head(numpanels) = gamma_u.head(numpanels);
    trace_o_N.head(numpanels) = -gamma_u.tail(numpanels);
    trace_o_D(numpanels) = trace_o_D(0);
    trace_o_N(numpanels) = trace_o_N(0);

    // interpolate traces
#ifdef CMDL
    std::cout << "Interpolating traces..." << std::endl;
#endif
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline_i_D_real = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
    gsl_spline *spline_i_D_imag = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
    gsl_spline_init(spline_i_D_real, t.data(), trace_i_D.real().data(), numpanels + 1);
    gsl_spline_init(spline_i_D_imag, t.data(), trace_i_D.imag().data(), numpanels + 1);
    gsl_spline *spline_i_N_real = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
    gsl_spline *spline_i_N_imag = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
    gsl_spline_init(spline_i_N_real, t.data(), trace_i_N.real().data(), numpanels + 1);
    gsl_spline_init(spline_i_N_imag, t.data(), trace_i_N.imag().data(), numpanels + 1);
    gsl_spline *spline_o_D_real = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
    gsl_spline *spline_o_D_imag = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
    gsl_spline_init(spline_o_D_real, t.data(), trace_o_D.real().data(), numpanels + 1);
    gsl_spline_init(spline_o_D_imag, t.data(), trace_o_D.imag().data(), numpanels + 1);
    gsl_spline *spline_o_N_real = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
    gsl_spline *spline_o_N_imag = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
    gsl_spline_init(spline_o_N_real, t.data(), trace_o_N.real().data(), numpanels + 1);
    gsl_spline_init(spline_o_N_imag, t.data(), trace_o_N.imag().data(), numpanels + 1);

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
            t1 = t0 + qr.x(j) * plen;
            if (t1 < t[0])
                t1 += boundary_length;
            Trace_i_D(i, j) = gsl_spline_eval(spline_i_D_real, t1, acc) + ii * gsl_spline_eval(spline_i_D_imag, t1, acc);
            Trace_i_N(i, j) = gsl_spline_eval(spline_i_N_real, t1, acc) + ii * gsl_spline_eval(spline_i_N_imag, t1, acc);
            Trace_o_D(i, j) = gsl_spline_eval(spline_o_D_real, t1, acc) + ii * gsl_spline_eval(spline_o_D_imag, t1, acc);
            Trace_o_N(i, j) = gsl_spline_eval(spline_o_N_real, t1, acc) + ii * gsl_spline_eval(spline_o_N_imag, t1, acc);
            Eigen::Vector2d y = p[qr.x(j)], tangent = p.Derivative_01(qr.x(j)), normal_o;
            Y(i, j) = -y(0) - ii * y(1);
            D(i, j) = tangent.norm() * qr.w(j);
            // outward normal vector
            normal_o << tangent(1), -tangent(0);
            normal_o.normalize();
            N(i, j) = normal_o(0) + ii * normal_o(1);
        }
        t0 += plen;
    }

    // compute solution in [0,1]^2
    double step = 1. / (grid_size - 1.);
    double k_sqrt_ci = k * std::sqrt(c_i), k_sqrt_co = k * std::sqrt(c_o), kappa, excess;
    unsigned ind;
    int pos;
    for (unsigned I = 0; I < grid_size; ++I) {
        for (unsigned J = 0; J < grid_size; ++J) {
            Eigen::Vector2d x;
            x << I * step, J * step;
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
                complex_bessel::H1_01(kappa * X, H0, H1);
                G = (H0 * (pos == 1 ? Trace_i_N : Trace_o_N)
                    + double(pos) * kappa * H1 * (pos == 1 ? Trace_i_D : Trace_o_D) * (Z.real() * N.real() + Z.imag() * N.imag()) / X) * D;
                S(I, J) = ii * 0.25 * G.sum();
#if 0
                if (pos == -1)
                    S(I, J) -= u_inc(x);
#endif
            }
        }
    }

    auto toc = high_resolution_clock::now();

#ifdef CMDL
    std::cout << "Total time: " << 1e-3 * duration_cast<milliseconds>(toc - tic).count() << " sec" << std::endl;
#endif

    // free spline resources
    gsl_spline_free(spline_i_D_real);
    gsl_spline_free(spline_i_D_imag);
    gsl_spline_free(spline_i_N_real);
    gsl_spline_free(spline_i_N_imag);
    gsl_spline_free(spline_o_D_real);
    gsl_spline_free(spline_o_D_imag);
    gsl_spline_free(spline_o_N_real);
    gsl_spline_free(spline_o_N_imag);
    gsl_interp_accel_free(acc);

    // output results to file
#ifdef CMDL
    std::cout << "Writing results to file..." << std::endl;
#endif
    if (animate) {
        double cb_radius = S.cwiseAbs().maxCoeff();
        file_script << "set cbrange [" << -cb_radius << ":" << cb_radius << "]" << std::endl;
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
        for (unsigned I = 0; I < grid_size; ++I) {
            for (unsigned J = 0; J < grid_size; ++J) {
                file_out << I * step << '\t' << J * step << '\t'
                         << (S(I, J) * (animate ? exp(-ii * 0.08 * M_PI * double(count)) : 1.)).real()
                         << std::endl;
            }
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
