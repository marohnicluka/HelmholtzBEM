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
#include <Eigen/Core>
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
    if (argc < 13) {
        std::cerr << "Error: too few input arguments" << std::endl;
        return 1;
    }
    if (argc > 15) {
        std::cerr << "Error: too many input arguments" << std::endl;
        return 1;
    }

    Eigen::initParallel();
    Eigen::setNbThreads(1);

    // read filenames for obstacle and incoming wave
    string fname_scatterer = argv[1], fname_incoming = argv[2];
    bool smooth_scatterer = fname_scatterer.back() == '@';
    if (smooth_scatterer)
        fname_scatterer.pop_back();

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
    int mode = argc > 13 ? atoi(argv[13]) : 0;
    bool add_u_inc = false;
    if (mode > 2) {
        mode -= 3;
        add_u_inc = true;
    }
    bool animate = mode == 1;
    double intensity = argc > 14 ? atof(argv[14]) : 1.0;

    // read incoming wave from file
    incoming::wave u_inc_spec;
    if (!incoming::load(fname_incoming, u_inc_spec)) {
        std::cerr << "Error: failed to read incoming wave from file" << std::endl;
        return 1;
    }

    scatterer::SmoothScatterer *sscat = NULL;
    ParametrizedMesh *mesh;
    Eigen::VectorXd vertices_x, vertices_y;

    if (smooth_scatterer) {
#ifdef CMDL
        std::cout << "Using smooth scatterer" << std::endl;
#endif
        sscat = new scatterer::SmoothScatterer(fname_scatterer);
        unsigned Npanels = atoi(argv[6]);
        mesh = new ParametrizedMesh(sscat->panelize(Npanels), MESH_TYPE_SMOOTH, static_cast<void*>(sscat));
        sscat->sample_vertices(std::max(upper_right_corner(0) - lower_left_corner(0), upper_right_corner(1) - lower_left_corner(1)),
                               250, vertices_x, vertices_y);
    } else {
#ifdef CMDL
        std::cout << "Using polygonal scatterer" << std::endl;
#endif
        // read polygonal scatterer from file
        if (!scatterer::read(fname_scatterer, vertices_x, vertices_y)) {
            std::cerr << "Error: failed to read scatterer from file" << std::endl;
            return 1;
        }
        // construction of a ParametrizedMesh object from the vector of panels
        unsigned Npanels;
        if (strlen(argv[6]) > 1 && argv[6][1] == '.') {
            double f = atof(argv[6]);
            Npanels = scatterer::auto_num_panels(vertices_x, vertices_y, f);
        } else Npanels = atoi(argv[6]);
        auto panels = scatterer::make_N_polygonal_panels(vertices_x, vertices_y, Npanels);
        mesh = new ParametrizedMesh(panels, MESH_TYPE_POLYGONAL);
    }

#ifdef CMDL
    scatterer::print_panels_info(mesh->getPanels());
#endif
    for (const auto &p : mesh->getPanels()) {
        if (p->length() > M_PI / (5. * k.real())) {
            std::cerr << "Warning: the number of panels may be too small" << std::endl;
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

    auto tic = high_resolution_clock::now();

    Eigen::ArrayXXd grid_X, grid_Y;
    Eigen::ArrayXXcd S = tp::direct_second_kind::solve_in_rectangle(*mesh, u_inc, u_inc_del, 10, order, k, c_o, c_i,
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
    unsigned nv = vertices_x.size();
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
        for (unsigned ii = 0; ii < grid_size; ++ii) {
            for (unsigned jj = 0; jj < grid_size; ++jj)
                file_out << grid_X(ii, jj) << '\t' << grid_Y(ii, jj) << '\t'
                         << (mode == 2 ? std::abs(S(ii, jj)) : (S(ii, jj) * (animate ? exp(-ii * 0.08 * M_PI * double(count)) : 1.)).real())
                         << std::endl;
            file_out << std::endl;
        }
        file_out.close();
        file_script << "set object 1 polygon from ";
        for (unsigned i = 0; i < nv; ++i) {
            file_script << vertices_x(i) << ", " << vertices_y(i) << " to ";
        }
        file_script << vertices_x(0) << ", " << vertices_y(0) << " lw 0.5 lc rgb \'#000000\' front" << std::endl;
        if (animate)
            file_script << "set output \"img/" << fname_count << ".png\"" << std::endl;
        file_script << "splot \'img/" << fname_count << suffix << "\'" << std::endl;
    }
    file_script.close();
    delete mesh;
    if (sscat != NULL)
        delete sscat;
    return 0;
}
