/**
 * \file find_resonances.cpp
 * \brief This target builds a script that computes minimas in
 * the smallest singular value of the Galerkin BEM
 * approximated solutions operator for the sedond-kind
 * direct BIEs of the Helmholtz transmission problem
 * using Brent's method.
 * The results are written to the <tt>data</tt> directory.
 * The script can be run as follows:
 *
 * <tt>
 *  /path/to/find_resonances \<scatterer filename\> \<refraction inside\>
 *     \<min wavenumber\> \<max wavenumber\> [\<quadrature order\> [\<accuracy\>]].
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

#include <complex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <execution>
#include <algorithm>
#include <string>
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <gsl/gsl_errno.h>
#include "parametrized_line.hpp"
#include "gen_sol_op.hpp"
#include "randsvd.hpp"
#include "singular_values_arnoldi.hpp"
#include "find_roots.hpp"
#include "continuous_space.hpp"
#include "scatterer.hpp"
#include "cbessel.hpp"

// define shorthand for time benchmarking tools, complex data type and immaginary unit
using namespace std::chrono;

int main(int argc, char** argv) {

    // check whether we have the correct number of input arguments
    if (argc < 5) {
        std::cerr << "Error: too few input arguments!" << std::endl;
        return 1;
    }
    if (argc > 7) {
        std::cerr << "Error: too many input arguments" << std::endl;
        return 1;
    }

    gsl_set_error_handler_off();

    double c_i = atof(argv[2]), c_o = 1.0;
    double k_min = atof(argv[3]);
    double k_max = atof(argv[4]);
    double tol = 1e-6;
    double rfac = 0.5;

    // read polygonal scatterer from file
    string fname_scatterer = argv[1];
    Eigen::VectorXd poly_x, poly_y;
    if (!scatterer::read_polygon(fname_scatterer, poly_x, poly_y)) {
        std::cerr << "Error: failed to read scatterer from file" << std::endl;
        return 1;
    }
    // define order of quadrature rule and the number of subspace iterations
    unsigned order = 10, q = 2, loc_min_count, degf = 10;
    if (argc > 5)
        order = atoi(argv[5]);
    if (argc > 6)
        tol = atof(argv[6]);

    // generate output filename with set parameters
    std::string base_name = "../data/file_find_resonances_";
    std::string file_plot = base_name + "plot.m";
    std::string suffix = ".dat";
    std::string sep = "_";
    std::string file_minimas = base_name;
    file_minimas.append(argv[2]).append(sep).append(argv[3]).append(sep).append(argv[4]);
    file_minimas += suffix;
    // clear existing files
    std::ofstream file_out;
    file_out.open(file_minimas, std::ofstream::out | std::ofstream::trunc);
    file_out.close();

    // Inform user of started computation.
#ifdef CMDL
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Finding resonances using heuristic and hybrid root-finding method." << std::endl;
    std::cout << "Computing on user-defined problem using the specified domain." << std::endl;
    std::cout << std::endl;
#endif

    auto policy = std::execution::par_unseq;
    ContinuousSpace<1> cont_space;

    std::vector<size_t> ind, loc;
    std::vector<double> rsv, loc_min, a, b, kvals;

    auto tic = high_resolution_clock::now();

    double L = scatterer::length(poly_x, poly_y), Np = degf * L * k_max * .5 * M_1_PI;
#ifdef CMDL
    std::cout << "Finding the optimal number of panels..." << std::endl;
#endif
    BrentMinimizer br(Np, Np / rfac, 1e-3);
    int status = 0;
    Np = br.local_min_rc(status, 0.);
    while (status) {
        L = scatterer::panelize(poly_x, poly_y, (unsigned int)std::round(Np), rfac, false).maxPanelLength();
        Np = br.local_min_rc(status, std::abs(2. * M_PI / (degf * k_max) - L));
    }

    unsigned Npanels = (unsigned int)std::round(Np);
    auto mesh = scatterer::panelize(poly_x, poly_y, Npanels, rfac);
    unsigned nc = 2, nr = 2 * mesh.getNumPanels();
    auto W = randomized_svd::randGaussian(nr, nc);
    BuilderData builder_data(mesh, cont_space, cont_space, order);
    GalerkinBuilder builder(builder_data);
    SolutionsOperator so(builder_data);

    std::mutex mtx;

    auto f_rsvd = [&](double k) {
        Eigen::MatrixXcd T;
        so.gen_sol_op(k, c_o, c_i, T);
        return randomized_svd::sv(T, W, q);
    };
#ifdef CMDL
    std::cout << "Sampling k-interval..." << std::endl;
#endif
    size_t n_points_k = opt_subdiv(f_rsvd, k_min, k_max, std::thread::hardware_concurrency(), 10, 1e-4 * (k_max - k_min), kvals, rsv, loc);
#ifdef CMDL
    std::cout << "Optimal number of samples: " << n_points_k << std::endl;
#endif
    loc_min_count = loc.size(); // number of local minima
#ifdef CMDL
    std::cout << "Found " << loc_min_count << " candidate(s)." << std::endl;
#endif
    ind.resize(loc_min_count);
    std::iota(ind.begin(), ind.end(), 0);
    a.resize(loc_min_count);
    b.resize(loc_min_count);
    loc_min.resize(loc_min_count);

    std::vector<std::vector<double> > xvals(loc_min_count), yvals(loc_min_count);

    // output sampling results to matlab file
    file_out.open(file_plot, std::ios_base::app);
    file_out << std::setprecision(18);
    file_out << "k = [";
    for (size_t i = 0; i < n_points_k; ++i) {
        if (i > 0)
            file_out << ",";
        file_out << kvals[i];
    }
    file_out << "];" << std::endl << "rsv = [";
    for (size_t i = 0; i < n_points_k; ++i) {
        if (i > 0)
            file_out << ",";
        file_out << rsv[i];
    }
    file_out << "];" << std::endl;
    file_out.close();

#if 1
    // improve candidates using interpolation
#ifdef CMDL
    std::cout << "Improving candidates..." << std::endl;
#endif
    unsigned ord = 5;
    std::for_each(ind.cbegin(), ind.cend(), [&](size_t i) {
        size_t pos = loc[i], il, ir;
        assert(pos > 0 && pos + 1 < n_points_k);
        for (ir = pos; ir + 1 < n_points_k && rsv[ir-1] + rsv[ir+1] > 2. * rsv[ir]; ++ir);
        for (il = pos; il > 0 && rsv[il-1] + rsv[il+1] > 2. * rsv[il]; --il);
        auto slice_x = std::vector<double>(kvals.begin() + il, kvals.begin() + ir + 1);
        auto slice_y = std::vector<double>(rsv.begin() + il, rsv.begin() + ir + 1);
        xvals[i] = slice_x;
        yvals[i] = slice_y;
        unsigned ic, N = slice_x.size();
        boost::math::barycentric_rational<double> interp(slice_x.data(), slice_y.data(), N, ord >= N ? N - 1 : ord);
        a[i] = kvals[pos - 1];
        b[i] = kvals[pos + 1];
        try {
            loc_min[i] = brent_guess(a[i], b[i], kvals[pos], interp(kvals[pos]), interp, tol, ic);
            if (loc_min[i] >= b[i] || loc_min[i] <= a[i])
                loc_min[i] = kvals[pos];
        } catch (std::runtime_error &e) {
#ifdef CMDL
            std::cerr << e.what() << std::endl;
#endif
        }
    });
#endif

#ifdef CMDL
    std::cout << "Starting local search using Brent's algorithm..." << std::endl;
#endif
    // Search for minima in the bracketed regions using Brent method
    unsigned ict = 0;
    double prog = 0;
    auto tic_b = high_resolution_clock::now();
    std::function<void(double)> rp = [&](double t) {
        std::unique_lock lck(mtx);
        prog += t;
#ifdef CMDL
        std::cout << "\r" << (int)std::round(100. * prog / double(loc_min_count)) << "% done ";
        std::flush(std::cout);
#endif
    };
    std::for_each(policy, ind.cbegin(), ind.cend(), [&](size_t i) {
        unsigned ic;
        GalerkinBuilder builder(builder_data);
        std::function<double(double)> fa = [&](double k) {
            Eigen::MatrixXcd T;
            so.gen_sol_op(builder, k, c_o, c_i, T);
#if 1
            return arnoldi::sv(T, 1, tol)(0);
#else
            return randomized_svd::sv(T, W, 2);
#endif
        };
        loc_min[i] = brent_gsl(a[i], b[i], loc_min[i], fa, tol, ic, &rp);
        std::unique_lock lck(mtx);
        if (!std::isfinite(loc_min[i])) {
#ifdef CMDL
            std::cout << "Discarding candidate " << i + 1 << std::endl;
#endif
        }
        ict += ic;
    });
    #ifdef CMDL
        std::cout << std::endl;
    #endif
    for (size_t i = loc_min_count; i-->0;) {
        if (!std::isfinite(loc_min[i])) {
            loc_min.erase(loc_min.begin() + i);
            loc_min_count--;
        }
    }
    //ind.resize(loc_min_count);

    auto toc = high_resolution_clock::now();
#ifdef CMDL
    std::cout << "Total function evaluations: " << ict << std::endl;
    std::cout << "Brent time: " << duration_cast<milliseconds>(toc - tic_b).count() * 1e-3 << std::endl;
    std::cout << "Total time: " << 1e-3 * duration_cast<milliseconds>(toc - tic).count() << " sec" << std::endl;
    std::cout << "Writing results to file..." << std::endl;
#endif

    // output results to file
    file_out.open(file_minimas, std::ios_base::app);
    file_out << std::setprecision((int)std::ceil(-std::log10(tol)));
    for (size_t i = 0; i < loc_min_count; ++i)
        file_out << loc_min[i] << std::endl;
    file_out.close();
#ifdef CMDL
    std::cout << std::endl;
#endif

    return 0;
}
