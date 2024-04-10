/**
 * \file roots_brent_polygon_rsvd.cpp
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
 *     \<\#grid points for root search\> \<\#panels\> \<quadrature order\>
 *     \<accuracy\> \<#subspace iterations\>.
 * </tt>
 *
 * The resulting file will contain the roots in a single column.
 * The user will be updated through the command line about the
 * progress of the algorithm if <tt>CMDL</tt> is set.
 *
 * This File is a part of the HelmholtzTransmissionProblemBEM library.
 *
 * (c) 2023 Luka Marohnić
 */

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
    if (argc < 10) {
        std::cerr << "Error: too few input arguments!" << std::endl;
        return 1;
    }
    if (argc > 10) {
        std::cerr << "Error: too many input arguments" << std::endl;
        return 1;
    }

    // define the side of square, refraction index and initial wavenumber
    double c_i = atof(argv[2]);
    double c_o = atof(argv[3]);
    double k_min = atof(argv[4]);

    // define mesh in space and on wavenumber on which to perform verification
    unsigned n_points_k = atoi(argv[5]);
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
    if (strlen(argv[6]) > 1 && argv[6][1] == '.') {
        double f = atof(argv[6]);
        Npanels = scatterer::auto_num_panels(poly_x, poly_y, f);
    } else Npanels = atoi(argv[6]);

    // define order of quadrature rule used to compute matrix entries and which singular value to evaluate
    unsigned order = atoi(argv[7]);

    // define accurracy of arnoldi algorithm
    double acc = std::max(atof(argv[8]), std::numeric_limits<double>::epsilon());

    // define the number of subspace iterations
    int q = atoi(argv[9]);

    // generate output filename with set parameters
    std::string base_name = "../data/file_roots_brent_polygon_rsvd_";
    std::string file_plot = base_name + "plot.m";
    std::string suffix = ".dat";
    std::string sep = "_";
    std::string file_minimas = base_name;
    file_minimas.append(argv[2]).append(sep).append(argv[5]).append(sep).append(argv[8]).append(sep).append(argv[9]);
    file_minimas += suffix;
    // clear existing files
    std::ofstream file_out;
    file_out.open(file_minimas, std::ofstream::out | std::ofstream::trunc);
    file_out.close();
    //file_out.open(file_plot, std::ofstream::out | std::ofstream::trunc);
    //file_out.close();

    int nc = 10;
    double k_max = k_min + 0.1, k_step = (k_max - k_min) / (n_points_k - 1);
    size_t disc, loc_min_count;

    // Inform user of started computation.
#ifdef CMDL
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Finding resonances using rSVD approximation and Brent's method." << std::endl;
    std::cout << "Computing on user-defined problem using the specified domain." << std::endl;
    std::cout << std::endl;
#endif

    auto policy = std::execution::par;
    parallelize_builder(true);
    ContinuousSpace<1> cont_space;

    std::vector<size_t> ind(n_points_k);
    std::vector<double> rsv(n_points_k), loc_min, a, b;
    std::iota(ind.begin(), ind.end(), 0);

    auto tic = high_resolution_clock::now();

    auto mesh = scatterer::panelize(poly_x, poly_y, Npanels, rfac);
    int nr = 2 * mesh.getNumPanels();
    auto W = randomized_svd::randGaussian(nr, nc);
    BuilderData builder_data(mesh, cont_space, cont_space, order);
    SolutionsOperator so(builder_data);

#ifdef CMDL
    std::cout << "Bracketing local extrema..." << std::endl;
#endif
    // Sweep the k interval with subdivision of size n_points_k, do this in parallel.
    // For each value k, approximate the smallest singular value by using rSVD.
    std::transform(policy, ind.cbegin(), ind.cend(), rsv.begin(), [&](size_t i) {
        Eigen::MatrixXcd T;
        so.gen_sol_op(k_min + k_step * i, c_o, c_i, T);
        auto res = randomized_svd::sv(T, W, q);
        return res;
    });

    // Bracket the local minima of the rSVD curve.
    // If n_points_k is not too large, the obtained intervals will
    // contain the true minima as well (rSVD approximates them
    // with a relative error of about 1e-3).
    // However, if n_points_k is too small, some minima may
    // be missed or the curve may not be convex in the intervals.
    for (size_t i = 0; i < rsv.size() - 2; ++i) {
        double &c = rsv[i+1], L = rsv[i], R = rsv[i+2];
        double k = k_min + i * k_step;
        if (L - c > 0. && R - c > 0.) { // local minimum
            a.push_back(k);
            loc_min.push_back(k + k_step);
            b.push_back(k + 2. * k_step);
        }
    }
    loc_min_count = a.size(); // number of local minima
#ifdef CMDL
    std::cout << "Found " << loc_min_count << " candidates." << std::endl;
#endif
    ind.resize(loc_min_count);

#ifdef CMDL
    std::cout << "Starting local search..." << std::endl;
#endif
    // Search for minima in the bracketed regions by using Brent method
    unsigned ict = 0;
    auto old_loc_min = loc_min;
    std::for_each(policy, ind.cbegin(), ind.cend(), [&](size_t i) {
        unsigned ic = 0;
        int status = 0;
        BrentMinimizer br_min(a[i], b[i] + 2. * k_step, acc);
        double arg = br_min.local_min_rc(status, 0.);
        Eigen::MatrixXcd T;
        GalerkinMatrixBuilder builder(builder_data);
        while (status) {
            so.gen_sol_op(builder, arg, c_o, c_i, T);
            double p = randomized_svd::sv(T, W, q);
            arg = br_min.local_min_rc(status, p);
            ++ic;
        }
        loc_min[i] = arg;
        ict += ic;
    });
#ifdef CMDL
    std::cout << "Total function evaluations: " << ict << std::endl;
    std::cout << "Local minima found:" << std::endl;
    Eigen::MatrixXcd T, T_der;
    for_each (loc_min.cbegin(), loc_min.cend(), [&](double p) {
        std::cout << p;
#ifdef COMPUTE_DERIVATIVE_AT_MINIMA
        so.gen_sol_op_1st_der(p, c_o, c_i, T, T_der);
        double d = randomized_svd::sv_der(T, T_der, W, q)(1);
        std::cout << " (1st derivative: " << d << ")";
#endif
        std::cout << std::endl;
    });
#endif

    // output results to file
    file_out.open(file_minimas, std::ios_base::app);
    file_out << std::setprecision((int)std::ceil(-std::log10(acc)));
    loc_min_count = loc_min.size();
    for (size_t i = 0; i < loc_min_count; ++i)
        file_out << loc_min[i] << std::endl;
    file_out.close();
    auto toc = high_resolution_clock::now();
#ifdef CMDL
    std::cout << "Total time: " << 1e-3 * duration_cast<milliseconds>(toc - tic).count() << " sec" << std::endl;
    std::cout << std::endl;
#endif
    return 0;
}
