/**
 * \file plot_solution_square.cpp
 * \brief This target builds a script that solves the
 * Helmholtz transmission problem and outputs the gnuplot file.
 * The scatterer is set to be a square. The results are
 * written to the <tt>data</tt> directory.
 * The script can be run as follows:
 *
 * <tt>
 *  /path/to/plot_solution_square \<side length of the square\>
 *     \<refraction inside\> \<refraction outside\> \<wavenumber\>
 *     \<\#panels\> \<grid size\> \<order of quadrature rule\> \<angle\>.
 * </tt>
 *
 * This target produces a gnuplot script which plots the solution inside
 * the square. If wavenumber is not positive, the script creates a
 * series of PNG images for k varying from 1.0 to 10.0. The following
 * commandline produces an animation out of these images (note that it
 * should be run from the <tt>data</tt> directory):
 *
 * <tt>
 *  convert -delay 20 -loop 0 img/file_plot_solution_square_XXXX_*.png output.gif
 * </tt>
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
#include "parametrized_line.hpp"
#include "gen_sol_op.hpp"
#include "continuous_space.hpp"
#include "mass_matrix.hpp"
#include "cbessel.hpp"
#include <gsl/gsl_spline.h>

using namespace std::chrono;

typedef std::complex<double> complex_t;
complex_t ii = complex_t(0,1.);

int main(int argc, char** argv) {

    // check whether we have the correct number of input arguments
    if (argc < 9)
        throw std::runtime_error("Too few input arguments!");
    if (argc > 9)
        throw std::runtime_error("Too many input arguments!");

    // define the side of square, refraction index and wavenumber
    double eps = atof(argv[1]);
    double c_i = atof(argv[2]);
    double c_o = atof(argv[3]);
    double k = atof(argv[4]);

    // define mesh in space and on wavenumber on which to perform verification
    unsigned numpanels = atoi(argv[5]);
    // compute mesh for numpanels
    using PanelVector = PanelVector;
    // corner points for the square
    Eigen::RowVectorXd x1(2);
    x1 << 0, 0; // point (0,0)
    Eigen::RowVectorXd x2(2);
    x2 << eps, 0; // point (1,0)
    Eigen::RowVectorXd x3(2);
    x3 << eps, eps; // point (1,1)
    Eigen::RowVectorXd x4(2);
    x4 << 0, eps; // point (0,1)
    // parametrized line segments forming the edges of the polygon
    ParametrizedLine line1(x1, x2);
    ParametrizedLine line2(x2, x3);
    ParametrizedLine line3(x3, x4);
    ParametrizedLine line4(x4, x1);
    // splitting the parametrized lines into panels for a mesh to be used for
    // BEM (Discretization).
    PanelVector line1panels = line1.split(numpanels/4);
    PanelVector line2panels = line2.split(numpanels/4);
    PanelVector line3panels = line3.split(numpanels/4);
    PanelVector line4panels = line4.split(numpanels/4);
    PanelVector panels;
    // storing all the panels in order so that they form a polygon
    panels.insert(panels.end(), line1panels.begin(), line1panels.end());
    panels.insert(panels.end(), line2panels.begin(), line2panels.end());
    panels.insert(panels.end(), line3panels.begin(), line3panels.end());
    panels.insert(panels.end(), line4panels.begin(), line4panels.end());
    // construction of a ParametrizedMesh object from the vector of panels
    ParametrizedMesh mesh(panels);

    // define grid size and order of quadrature rule used to compute matrix
    // entries and which singular value to evaluate
    unsigned grid_size = atoi(argv[6]);
    unsigned order = atoi(argv[7]);

    // direction vector
    double angle = atoi(argv[8]) * M_PI / 180.;
    Eigen::Vector2d d(std::sin(angle), std::cos(angle));

    ContinuousSpace<1> cont_space;
    BuilderData builder_data(mesh, cont_space, cont_space, order);
    GalerkinMatrixBuilder builder(builder_data);

    // generate output filename with set parameters
    std::string base_name = "file_plot_solution_square_";
    std::string suffix = ".dat";
    std::string sep = "_";
    std::string fname = base_name;
    fname.append(argv[1]).append(sep).append(argv[2]).append(sep).append(argv[3]).append(sep).append(argv[4]).append(sep)
    .append(argv[5]).append(sep).append(argv[6]).append(sep).append(argv[7]).append(sep).append(argv[8]);

    // open script file
    std::ofstream file_script;
    string script_fname = "../data/" + fname + ".gnuplot";
    file_script.open(script_fname, std::ofstream::out | std::ofstream::trunc);
    file_script.close();
    file_script.open(script_fname, std::ios_base::app);
    file_script << "set view map" << std::endl
                << "set dgrid3d 100, 100, 1" << std::endl
                << "set pm3d at b" << std::endl
                << "unset key" << std::endl
                << "unset surface" << std::endl
                << "set size square" << std::endl;

    auto tic = high_resolution_clock::now();

    bool animate = k <= 0.0;
    std::vector<unsigned int> ind;
    if (animate) {
        file_script << "set term png" << std::endl;
        ind.resize(90);
        std::iota(ind.begin(), ind.end(), 0);
    } else ind.push_back(0);

    for_each (std::execution::seq, ind.begin(), ind.end(), [&](unsigned int count) {
        if (animate)
            k = 1.0 + double(count) / 10.0;
#ifdef CMDL
        if (!animate)
            std::cout << "---> k = " << k << std::endl;
#endif
        // assemble solutions operator matrix
#ifdef CMDL
        if (!animate)
            std::cout << "Assembling solutions operator matrix..." << std::endl;
#endif
        Eigen::MatrixXcd M, A, B, K_i, K_o, V_i, V_o, W_i, W_o;
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
        //M.resize(numpanels, numpanels); M.setIdentity();
        A.block(0, 0, numpanels, numpanels) = M - K_o + K_i;
        A.block(numpanels, 0, numpanels, numpanels) = W_o - W_i;
        A.block(0, numpanels, numpanels, numpanels) = V_o - V_i;
        A.block(numpanels, numpanels, numpanels, numpanels) = (M + K_o - K_i).transpose();
        // assemble the right-hand-side matrix
        B.block(0, 0, numpanels, numpanels) = 0.5 * M - K_o;
        B.block(numpanels, 0, numpanels, numpanels) = W_o;
        B.block(0, numpanels, numpanels, numpanels) = V_o;
        B.block(numpanels, numpanels, numpanels, numpanels) = (0.5 * M + K_o).transpose();

        // create traces
#ifdef CMDL
        if (!animate)
            std::cout << "Computing boundary traces..." << std::endl;
#endif
        Eigen::VectorXcd gamma_o_uinc, gamma_u;
        Eigen::VectorXd t;
        gamma_o_uinc.resize(2 * numpanels);
        t.resize(numpanels + 1);
        double t0 = 0., t1, boundary_length = 0.;
        for (unsigned i = 0; i < numpanels; ++i) {
            const auto &p = *panels[i];
            Eigen::Vector2d x = p(0), tangent = p.Derivative(0), normal_i;
            double len = p.length();
            boundary_length += len;
            t0 += len / 2.;
            t(i) = t0;
            t0 += len / 2.;
            // inward normal vector
            normal_i << -tangent(1), tangent(0);
            complex_t e = std::exp(ii * k * (d.transpose() * x)(0));
            gamma_o_uinc(i) = e;
            gamma_o_uinc(numpanels + i) = ii * k * e * (d.transpose() * normal_i.normalized())(0);
        }
        t0 += panels[0]->length() / 2.;
        t(numpanels) = t0;
        gamma_u = A.colPivHouseholderQr().solve(B * gamma_o_uinc);
        Eigen::VectorXcd trace_D, trace_N;
        trace_D.resize(numpanels + 1);
        trace_N.resize(numpanels + 1);
        trace_D.head(numpanels) = gamma_u.head(numpanels);
        trace_N.head(numpanels) = gamma_u.tail(numpanels);
        trace_D(numpanels) = trace_D(0);
        trace_N(numpanels) = trace_N(0);

        // interpolate traces
#ifdef CMDL
        if (!animate)
            std::cout << "Interpolating traces..." << std::endl;
#endif
        gsl_interp_accel *acc = gsl_interp_accel_alloc();
        gsl_spline *spline_D_real = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
        gsl_spline *spline_D_imag = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
        gsl_spline_init(spline_D_real, t.data(), trace_D.real().data(), numpanels + 1);
        gsl_spline_init(spline_D_imag, t.data(), trace_D.imag().data(), numpanels + 1);
        gsl_spline *spline_N_real = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
        gsl_spline *spline_N_imag = gsl_spline_alloc(gsl_interp_cspline_periodic, numpanels + 1);
        gsl_spline_init(spline_N_real, t.data(), trace_N.real().data(), numpanels + 1);
        gsl_spline_init(spline_N_imag, t.data(), trace_N.imag().data(), numpanels + 1);

        // lift the solution from traces
#ifdef CMDL
        if (!animate)
            std::cout << "Lifting the solution from traces..." << std::endl;
#endif
        Eigen::MatrixXcd S;
        S.resize(grid_size, grid_size);
        S.setZero();
        QuadRule qr = getGaussQR(order, 0., 1.);
        size_t n = qr.n;
        Eigen::ArrayXXcd Y, Z, H0, H1, Trace_D, Trace_N, N, G;
        Eigen::ArrayXXd X, D, W;
        X.resize(numpanels, n);
        Y.resize(numpanels, n);
        Z.resize(numpanels, n);
        H0.resize(numpanels, n);
        H1.resize(numpanels, n);
        Trace_D.resize(numpanels, n);
        Trace_N.resize(numpanels, n);
        D.resize(numpanels, n);
        N.resize(numpanels, n);
        G.resize(numpanels, n);
        W.resize(numpanels, n);
        t0 = 0.;
        for (unsigned i = 0; i < numpanels; ++i) {
            const auto &p = *panels[i];
            double plen = p.length();
            for (unsigned j = 0; j < n; ++j) {
                t1 = t0 + qr.x(j) * plen;
                if (t1 < t[0])
                    t1 += boundary_length;
                Trace_D(i, j) = gsl_spline_eval(spline_D_real, t1, acc) + ii * gsl_spline_eval(spline_D_imag, t1, acc);
                Trace_N(i, j) = gsl_spline_eval(spline_N_real, t1, acc) + ii * gsl_spline_eval(spline_N_imag, t1, acc);
                Eigen::Vector2d y = -p[qr.x(j)];
                Y(i, j) = y(0) + ii * y(1);
                Eigen::Vector2d tangent = p.Derivative_01(qr.x(j)), normal_o;
                D(i, j) = tangent.norm();
                // outward normal vector
                normal_o << tangent(1), -tangent(0);
                normal_o.normalize();
                N(i, j) = normal_o(0) + ii * normal_o(1);
                W(i, j) = qr.w(j);
            }
            t0 += plen;
        }
        // compute solution in the interior
        double step = eps / (grid_size - 1.), k_sqrt_ci = k * std::sqrt(c_i);
        D *= W;
        for (unsigned I = 1; I < grid_size - 1; ++I) {
            for (unsigned J = 1; J < grid_size - 1; ++J) {
                G.setZero();
                Z = Y + (I * step) + ii * (J * step);
                X = Z.cwiseAbs();
                complex_bessel::H1_01(k_sqrt_ci * X, H0, H1);
                G += H0 * Trace_N;
                G += k_sqrt_ci * H1 * Trace_D * (Z.real() * N.real() + Z.imag() * N.imag()) / X;
                G *= D;
                S(I, J) = ii * 0.25 * G.sum();
            }
        }
        // compute solution at the boundary
        t0 = 0.;
        for (unsigned J = 0; J < grid_size - 1; ++J, t0 += step) {
            t1 = t0 + (t0 < t[0] ? boundary_length : 0.);
            S(0, J) = gsl_spline_eval(spline_D_real, t1, acc) + ii * gsl_spline_eval(spline_D_imag, t1, acc);
        }
        for (unsigned I = 0; I < grid_size - 1; ++I, t0 += step) {
            S(I, grid_size - 1) = gsl_spline_eval(spline_D_real, t0, acc) + ii * gsl_spline_eval(spline_D_imag, t0, acc);
        }
        for (unsigned J = grid_size; J-->1; t0 += step) {
            S(grid_size - 1, J) = gsl_spline_eval(spline_D_real, t0, acc) + ii * gsl_spline_eval(spline_D_imag, t0, acc);
        }
        for (unsigned I = grid_size; I-->1; t0 += step) {
            S(I, 0) = gsl_spline_eval(spline_D_real, t0, acc) + ii * gsl_spline_eval(spline_D_imag, t0, acc);
        }

        // output results to file
#ifdef CMDL
        if (!animate)
            std::cout << "Writing results to file..." << std::endl;
#endif
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
                file_out << I * step << '\t' << J * step << '\t' << S(I, J).real()
                //- std::exp(ii * k * (I * d(0) + J * d(1)) * step).real()
                << std::endl;
            }
        }
        file_out.close();
        if (animate)
            file_script << "set output \"img/" << fname_count << ".png\"" << std::endl;
        file_script << "splot \'img/" << fname_count << suffix << "\'" << std::endl;
        if (animate)
            file_script << "unset label" << std::endl
                        << "set label left front nopoint \"k = " << k << "\" at 0.1,0.1" << std::endl
                        << "show label" << std::endl;

        // free spline resources
        gsl_spline_free(spline_D_real);
        gsl_spline_free(spline_D_imag);
        gsl_spline_free(spline_N_real);
        gsl_spline_free(spline_N_imag);
        gsl_interp_accel_free(acc);
    });

    file_script.close();

    auto toc = high_resolution_clock::now();

#ifdef CMDL
    std::cout << "Total time: " << 1e-3 * duration_cast<milliseconds>(toc - tic).count() << " sec" << std::endl;
#endif

    return 0;
}
