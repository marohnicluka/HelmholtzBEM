/**
 * \file roots_brent_polygon.cpp
 * \brief This target builds a script that computes minimas in
 * the smallest singular value of the Galerkin BEM
 * approximated solutions operator for the sedond-kind
 * direct BIEs of the Helmholtz transmission problem
 * using Brent's method.
 * The results are written to the <tt>data</tt> directory.
 * The script can be run as follows:
 *
 * <tt>
 *  /path/to/roots_newton_square_rsvd \<scatterer filename\> \<refraction inside\>
 *     \<refraction outside\> \<min wavenumber\> \<max wavenumber\>
 *     \<tolerance\> \<\#panels\> \<quadrature order\>
 *     \<accuracy\> \<#subspace iterations\>.
 * </tt>
 *
 * The resulting file will contain the roots in the first column. The next three
 * columns will contain the minimal singular value and its first two derivatives
 * at root points, respectively.
 * The user will be updated through the command line about the
 * progress of the algorithm if <tt>CMDL</tt> is set.
 *
 * This File is a part of the HelmholtzTransmissionProblemBEM library.
 *
 * (c) 2023 Luka Marohnić
 */

//#define COMPUTE_DERIVATIVES_AT_MINIMA

#include <complex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <execution>
#include <algorithm>
#include <string>
#include "parametrized_line.hpp"
#include "gen_sol_op.hpp"
#include "randsvd.hpp"
#include "find_roots.hpp"
#include "continuous_space.hpp"
#include "scatterer.hpp"
#include "cbessel.hpp"

// define shorthand for time benchmarking tools, complex data type and immaginary unit
using namespace std::chrono;
typedef std::complex<double> complex_t;
complex_t ii = complex_t(0,1.);

int main(int argc, char** argv) {

    // check whether we have the correct number of input arguments
    if (argc < 11) {
        std::cerr << "Error: too few input arguments!" << std::endl;
        return 1;
    }
    if (argc > 11) {
        std::cerr << "Error: too many input arguments" << std::endl;
        return 1;
    }

    double c_i = atof(argv[2]);
    double c_o = atof(argv[3]);
    double k_min = atof(argv[4]);
    double k_max = atof(argv[5]);
    double tol = atof(argv[6]);

    // read polygonal scatterer from file
    string fname_scatterer = argv[1];
    Eigen::VectorXd poly_x, poly_y;
    if (!scatterer::read_polygon(fname_scatterer, poly_x, poly_y)) {
        std::cerr << "Error: failed to read scatterer from file" << std::endl;
        return 1;
    }
    // construction of a ParametrizedMesh object from the vector of panels
    unsigned Npanels;
    double rfac = 0.25; // panel length shrink factor
    if (strlen(argv[7]) > 1 && argv[7][1] == '.') {
        double f = atof(argv[7]);
        Npanels = scatterer::auto_num_panels(poly_x, poly_y, f);
    } else Npanels = atoi(argv[7]);

    // define order of quadrature rule used to compute matrix entries and which singular value to evaluate
    unsigned order = atoi(argv[8]);

    // define accurracy of arnoldi algorithm
    double acc = std::max(atof(argv[9]), std::numeric_limits<double>::epsilon());

    // define the number of subspace iterations
    int q = atoi(argv[10]);

    // generate output filename with set parameters
    std::string base_name = "../data/file_roots_brent_polygon_rsvd_";
    std::string file_plot = base_name + "plot.m";
    std::string suffix = ".dat";
    std::string sep = "_";
    std::string file_minimas = base_name;
    file_minimas.append(argv[2]).append(sep).append(argv[6]).append(sep).append(argv[9]).append(sep).append(argv[10]);
    file_minimas += suffix;
    // clear existing files
    std::ofstream file_out;
    file_out.open(file_minimas, std::ofstream::out | std::ofstream::trunc);
    file_out.close();
    //file_out.open(file_plot, std::ofstream::out | std::ofstream::trunc);
    //file_out.close();

    // Inform user of started computation.
#ifdef CMDL
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Finding resonances using rSVD approximation and Brent's method." << std::endl;
    std::cout << "Computing on user-defined problem using the specified domain." << std::endl;
    std::cout << std::endl;
#endif

    auto policy = std::execution::par;
    ContinuousSpace<1> cont_space;

    auto tic = high_resolution_clock::now();

    auto mesh = scatterer::panelize(poly_x, poly_y, Npanels, rfac);
    int nr = 2 * mesh.getNumPanels(), nc = 4;
    auto W = randomized_svd::randGaussian(nr, nc);
    BuilderData builder_data(mesh, cont_space, cont_space, order);
    GalerkinBuilder builder(builder_data);
    SolutionsOperator so(builder_data);

    std::function<double(double)> func = [&](double k) {
        Eigen::MatrixXcd T;
        so.gen_sol_op(builder, k, c_o, c_i, T);
        return randomized_svd::sv(T, W, q);
    };

    BrentMinimaFinder br(func);

    auto loc_min = br.find_minima(k_min, k_max, tol);
    unsigned ict = br.get_eval_count(), loc_min_count = loc_min.size();

    #ifdef CMDL
    std::cout << "Total function evaluations: " << ict << std::endl;
    Eigen::MatrixXcd T, T_der, T_der_2;
    std::vector<int> ind(loc_min_count);
    std::iota(ind.begin(), ind.end(), 0);
#ifdef COMPUTE_DERIVATIVES_AT_MINIMA
    std::cout << "Computing derivatives at local minima..." << std::endl;
    Eigen::ArrayXXd der(loc_min_count, 3);
    for_each (policy, ind.cbegin(), ind.cend(), [&](unsigned int i) {
        so.gen_sol_op_2nd_der(loc_min[i], c_o, c_i, T, T_der, T_der_2);
        der.row(i) = randomized_svd::sv_der2(T, T_der, T_der_2, W, q);
    });
#endif
    std::cout << "Point(s) of local minima:" << std::endl;
    for_each(ind.cbegin(), ind.cend(), [&](unsigned int i) {
        std::cout << loc_min[i]; std::flush(std::cout);
#ifdef COMPUTE_DERIVATIVES_AT_MINIMA
        std::cout << " (sv = " << der(i, 0) << ", sv' = " << der(i, 1) << ", sv'' = " << der(i, 2) << ")";
#endif
        std::cout << std::endl;
    });
#endif

    // output results to file
    file_out.open(file_minimas, std::ios_base::app);
    file_out << std::setprecision((int)std::ceil(-std::log10(acc)));
    loc_min_count = loc_min.size();
    for (size_t i = 0; i < loc_min_count; ++i)
#ifdef COMPUTE_DERIVATIVES_AT_MINIMA
        file_out << loc_min[i] << '\t' << der(i, 0) << '\t' << der(i, 1) << '\t' << der(i, 2) << std::endl;
#else
        file_out << loc_min[i] << std::endl;
#endif
    file_out.close();
    auto toc = high_resolution_clock::now();
#ifdef CMDL
    std::cout << "Total time: " << 1e-3 * duration_cast<milliseconds>(toc - tic).count() << " sec" << std::endl;
    std::cout << std::endl;
#endif
    return 0;
}
