/**
 * \file snapshots.cpp
 * \brief This target builds a script that snapshots the scattered
 * wave on the boundary.
 *
 * <tt>
 *  /path/to/plot_solution_polygon \<scatterer file\> \<incoming wave file\>
 *     \<refraction inside\> \<refraction outside\>
 *     \<min wavenumber\> \<max wavenumber\>
 *     \<#panels\> \<quadrature order\>.
 * </tt>
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
    if (argc < 9) {
        std::cerr << "Error: too few input arguments" << std::endl;
        return 1;
    }
    if (argc > 9) {
        std::cerr << "Error: too many input arguments" << std::endl;
        return 1;
    }

    Eigen::initParallel();

    // read filenames for obstacle and incoming wave
    string fname_scatterer = argv[1], fname_incoming = argv[2];

    // define refraction index, wavenumber, order of quadrature and grid size
    double c_i = atof(argv[3]);
    double c_o = atof(argv[4]);
    complex_t k_min = atof(argv[5]);
    complex_t k_max = atof(argv[6]);


    // define grid size and order of quadrature rule used to compute matrix
    // entries and which singular value to evaluate
    unsigned order = atoi(argv[8]);

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
    if (strlen(argv[7]) > 1 && argv[7][1] == '.') {
        double f = atof(argv[7]);
        Npanels = scatterer::auto_num_panels(poly_x, poly_y, f);
    } else Npanels = atoi(argv[7]);
    auto mesh = scatterer::panelize(poly_x, poly_y, Npanels, 0.25);
    for (const auto &p : mesh.getPanels()) {
        if (p->length() > M_PI / (5. * k_max.real())) {
#ifdef CMDL
            std::cout << "Warning: the number of panels may be too small" << std::endl;
#endif
            break;
        }
    }

    std::cout << "Number of panels: " << mesh.getNumPanels() << std::endl;

    // generate output filename with set parameters
    std::string base_name = "file_snapshots_";
    std::string suffix = ".dat";
    std::string sep = "_";
    std::string fname = base_name;
    fname.append(argv[3]).append(sep).append(argv[5]).append(sep).append(argv[8]);

    // open script file
    std::ofstream outfile;
    string script_fname = "../data/" + fname + ".dat";
    outfile.open(script_fname, std::ofstream::out | std::ofstream::trunc);
    outfile.close();
    outfile.open(script_fname, std::ios_base::app);
    if (!outfile.is_open()) {
        std::cerr << "Error: failed to open output file for writing" << std::endl;
        return 1;
    }

    unsigned n_kvals = 30;

    auto tic = high_resolution_clock::now();
    Eigen::ArrayXXcd S;
    S.resize(2 * mesh.getNumPanels(), n_kvals);

    for (unsigned i = 0; i < n_kvals; ++i) {
        // create the incoming wave function and its gradient
        complex_t k = k_min + ((k_max - k_min) * double(i)) / double(n_kvals-1);
        std::cout << "Creating snapshot " << i << " for k = " << k << " ..." << std::endl;
        auto u_inc = [&](double x1, double x2) {
            return incoming::compute(u_inc_spec, Eigen::Vector2d(x1, x2), k);
        };
        auto u_inc_del = [&](double x1, double x2) {
            return incoming::compute_del(u_inc_spec, Eigen::Vector2d(x1, x2), k);
        };
        auto res = tp::direct_second_kind::solve(mesh, u_inc, u_inc_del, order, k, c_o, c_i);
        std::cout << "RES (" << res.size() << "): " << res << std::endl;
    }

    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(S.matrix());
    std::cout << svd.singularValues() << std::endl;

    auto toc = high_resolution_clock::now();

#ifdef CMDL
    std::cout << "Total time: " << 1e-3 * duration_cast<milliseconds>(toc - tic).count() << " sec" << std::endl;
#endif

    outfile.close();

    return 0;
}

