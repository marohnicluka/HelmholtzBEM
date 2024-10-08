/**
 * \file verify_solution_analytic.cpp
 * \brief This target builds a script that solves the
 * Helmholtz transmission problem in a circle with BesselJ
 * incoming wave and compares the result with the analytic
 * solution in the square [(-1,-1),(1,1)].
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
#include "parametrized_circular_arc.hpp"
#include "parametrized_line.hpp"
#include "solvers.hpp"
#include "gen_sol.hpp"
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
    //std::complex<double> k(atof(argv[5]), 1.0);
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
#if 0 // make linear panels
    for (auto &p : panels) {
        p = std::make_shared<ParametrizedLine>(p->operator[](0), p->operator[](1));
    }
#endif
    ParametrizedMesh mesh(panels);
    if (mesh.isPolygonal())
        std::cout << "Working with polygonal mesh" << std::endl;

    // generate output filename with set parameters
    std::string base_name = "file_verify_solution_analytic_error_";
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

    // analytic solution data
    double *a_n = new double[2 * bessel_order + 1];
    for (unsigned i = 0; i < 2 * bessel_order + 1; ++i) {
        a_n[i] = 1.;
    }

    // incoming wave
    auto u_inc = [&](double x1, double x2) {
        return sol::u_i(x1, x2, bessel_order, a_n, k);
    };
    auto u_inc_del = [&](double x1, double x2) {
        return sol::u_i_del(x1, x2, bessel_order, a_n, k);
    };

    auto tic = high_resolution_clock::now();

    Eigen::ArrayXXd grid_X, grid_Y;
    Eigen::Vector2d lower_left_corner(-1., -1.), upper_right_corner(1., 1.);
    Eigen::ArrayXXcd S = tp::direct_second_kind::solve_in_rectangle(mesh, u_inc, u_inc_del, 10, order, k, c_o, c_i,
                                                                    lower_left_corner, upper_right_corner, grid_size, grid_size,
                                                                    grid_X, grid_Y, false);

    // compute analytic solution in [-1,1]^2
    Eigen::ArrayXXcd S_a(grid_size, grid_size);
    for (unsigned ii = 0; ii < grid_size; ++ii) {
        for (unsigned jj = 0; jj < grid_size; ++jj) {
            Eigen::Vector2d x;
            x << grid_X(ii, jj), grid_Y(ii, jj);
            double r = x.norm();
            S_a(ii, jj) = r <= radius ? sol::u_t(x(0), x(1), bessel_order, radius, a_n, k, c_i)
                                      : sol::u_s(x(0), x(1), bessel_order, radius, a_n, k, c_i);
        }
    }

    Eigen::ArrayXXd err = (S - S_a).cwiseAbs();
    double sol_err = (S - S_a).matrix().norm() / S_a.matrix().norm();
#ifdef CMDL
    std::cout << "Solution error: " << sol_err << ", " << (err / S_a.cwiseAbs()).mean() << std::endl;
    std::cout << "Max relative error: " << (err / S_a.cwiseAbs()).matrix().maxCoeff() << std::endl;
#endif

    //S = (S - S_a).cwiseAbs() / S_a.norm();

    auto toc = high_resolution_clock::now();

#ifdef CMDL
    std::cout << "Total time: " << 1e-3 * duration_cast<milliseconds>(toc - tic).count() << " sec" << std::endl;
#endif

    // free resources
    delete[] a_n;

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
    for (unsigned ii = 0; ii < grid_size; ++ii) {
        for (unsigned jj = 0; jj < grid_size; ++jj)
            file_out << grid_X(ii, jj) << '\t' << grid_Y(ii, jj) << '\t' << err(ii, jj) << std::endl;
        file_out << std::endl;
    }
    file_out.close();
    //file_script << "set cbrange [" << 0 << ":" << S.real().maxCoeff() << "]" << std::endl;
    file_script << "splot \'img/" << fname << suffix << "\'" << std::endl;
    file_script.close();
#ifdef COMPUTE_BESSEL_STATS
    complex_bessel::print_bessel_stats();
#endif
    return 0;
}
