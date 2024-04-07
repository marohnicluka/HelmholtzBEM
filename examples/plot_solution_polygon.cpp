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
#include "parametrized_line.hpp"
#include "solvers.hpp"
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
    if (!scatterer::read_polygon(fname_scatterer, poly_x, poly_y)) {
        std::cerr << "Error: failed to read scatterer from file" << std::endl;
        return 1;
    }

    // construction of a ParametrizedMesh object from the vector of panels
    unsigned Npanels;
    if (strlen(argv[6]) > 1 && argv[6][1] == '.') {
        double f = atof(argv[6]);
        Npanels = scatterer::auto_num_panels(poly_x, poly_y, f);
    } else Npanels = atoi(argv[6]);
    auto mesh = scatterer::panelize(poly_x, poly_y, Npanels, 0.25);
    for (const auto &p : mesh.getPanels()) {
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
    auto u_inc = [&](double x1, double x2) {
        return incoming::compute(u_inc_spec, Eigen::Vector2d(x1, x2), k);
    };
    auto u_inc_del = [&](double x1, double x2) {
        return incoming::compute_del(u_inc_spec, Eigen::Vector2d(x1, x2), k);
    };

    // enable parallelization
    complex_bessel::parallelize(true);
    parallelize_builder(true);

    auto tic = high_resolution_clock::now();

    Eigen::ArrayXXd grid_X, grid_Y;
    Eigen::ArrayXXcd S = tp::direct_second_kind::solve_in_rectangle(mesh, u_inc, u_inc_del, 11, order, k, c_o, c_i,
                                                                    lower_left_corner, upper_right_corner, grid_size, grid_size,
                                                                    grid_X, grid_Y, add_u_inc);

    auto toc = high_resolution_clock::now();

#ifdef CMDL
    std::cout << "Total time: " << 1e-3 * duration_cast<milliseconds>(toc - tic).count() << " sec" << std::endl;
#endif

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
                file_out << grid_X(I, J) << '\t' << grid_Y(I, J) << '\t'
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
