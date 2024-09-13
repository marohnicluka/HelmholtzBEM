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
 *  /path/to/roots_brent_polygon_rsvd \<scatterer filename\> \<refraction inside\>
 *     \<refraction outside\> \<min wavenumber\> \<max wavenumber\>
 *     \<\#grid points for root search\> \<\#panels\> \<quadrature order\>
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

#include <complex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <execution>
#include <algorithm>
#include <string>
#include <random>
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

#define DEGREES_OF_FREEDOM 10
//#define WITH_ARNOLDI 1
//#define COMPUTE_DERIVATIVES_AT_MINIMA 1

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

    gsl_set_error_handler_off();

    // define the side of square, refraction index and initial wavenumber
    double c_i = atof(argv[2]);
    double c_o = atof(argv[3]);
    double k_min = atof(argv[4]);
    double k_max = atof(argv[5]);

    // define mesh in space and on wavenumber on which to perform verification
    unsigned n_points_k = atoi(argv[6]);
    // read polygonal scatterer from file
    string fname_scatterer = argv[1];
    Eigen::VectorXd poly_x, poly_y;
    if (!scatterer::read_polygon(fname_scatterer, poly_x, poly_y)) {
        std::cerr << "Error: failed to read scatterer from file" << std::endl;
        return 1;
    }
    // construction of a ParametrizedMesh object from the vector of panels
    unsigned Npanels;
    double rfac = 0.5; // panel length shrink factor
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

    double k_step = (k_max - k_min) / (n_points_k - 1);
    double eps_mach = std::numeric_limits<double>::epsilon();
    size_t loc_min_count;

    // Inform user of started computation.
#ifdef CMDL
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Finding resonances using rSVD approximation and Brent's method." << std::endl;
    std::cout << "Computing on user-defined problem using the specified domain." << std::endl;
    std::cout << std::endl;
#endif

    auto policy = std::execution::par_unseq;
    ContinuousSpace<1> cont_space;

    std::vector<size_t> ind(n_points_k), loc;
    std::vector<double> rsv(n_points_k), loc_min, a, b, kvals(n_points_k);
    std::iota(ind.begin(), ind.end(), 0);

    auto tic = high_resolution_clock::now();

    auto mesh = scatterer::panelize(poly_x, poly_y, Npanels, rfac);
    int nc = 4, nr = 2 * mesh.getNumPanels();
    auto W = randomized_svd::randGaussian(nr, nc);
    BuilderData builder_data(mesh, cont_space, cont_space, order);
    SolutionsOperator so(builder_data);

#ifdef DEGREES_OF_FREEDOM
    double L = mesh.getTotalLength(), Np = DEGREES_OF_FREEDOM * L * k_max * .5 * M_1_PI;
    if (double(Npanels) <= Np) {
        std::cerr << "Error: number of panels must be larger than " << (unsigned)std::ceil(Np) << std::endl;
        return 1;
    }
#ifdef CMDL
    std::cout << "Finding number of test panels..." << std::endl;
#endif
    BrentMinimizer br(Np, Npanels, 1e-3);
    int status = 0;
    Np = br.local_min_rc(status, 0.);
    while (status) {
        L = scatterer::panelize(poly_x, poly_y, (unsigned int)std::round(Np), rfac, false).maxPanelLength();
        Np = br.local_min_rc(status, std::abs(2. * M_PI / (DEGREES_OF_FREEDOM * k_max) - L));
    }

    unsigned Npanels_min = (unsigned int)std::round(Np);
    auto mesh_test = scatterer::panelize(poly_x, poly_y, Npanels_min, rfac);
    nr = 2 * mesh_test.getNumPanels();
    auto W_test = randomized_svd::randGaussian(nr, nc);
    BuilderData builder_data_test(mesh_test, cont_space, cont_space, order);
    SolutionsOperator so_test(builder_data_test);
#endif

#ifdef CMDL
    std::cout << "Bracketing local extrema"
#ifdef DEGREES_OF_FREEDOM
              << " with " << mesh_test.getNumPanels() << " panels"
#endif
              << "..." << std::endl;
#endif
    // Sweep the k interval with subdivision of size n_points_k, do this in parallel.
    // For each value k, approximate the smallest singular value by using rSVD.
    std::mutex mtx;
    unsigned done = 0;
    std::function<double(double)> sf = [&](double k) {
        Eigen::MatrixXcd T;
#ifdef DEGREES_OF_FREEDOM
        so_test.gen_sol_op(k, c_o, c_i, T);
#ifdef WITH_ARNOLDI
        double res = arnoldi::sv(T, 1, 1e-8)(0);
#else
        double res = randomized_svd::sv(T, W_test, q);
#endif
#else
        so.gen_sol_op(k, c_o, c_i, T);
#ifdef WITH_ARNOLDI
        double res = arnoldi::sv(T, 1, 1e-8)(0);
#else
        double res = randomized_svd::sv(T, W, q);
#endif
#endif
        return res;
    };
    std::transform(policy, ind.cbegin(), ind.cend(), rsv.begin(), [&](size_t i) {
        Eigen::MatrixXcd T;
        kvals[i] = k_min + k_step * i;
        double res = sf(kvals[i]);
        std::unique_lock lck(mtx);
        ++done;
#ifdef CMDL
        std::cout << "\r" << (100 * done) / n_points_k << "% done ";
        std::flush(std::cout);
#endif
        return res;
    });
#ifdef CMDL
    std::cout << std::endl;
#endif

    // Bracket the local minima of the rSVD curve.
    for (size_t i = 0; i < rsv.size() - 2; ++i) {
        double &c = rsv[i+1], L = rsv[i], R = rsv[i+2];
        double k = k_min + i * k_step;
        if (L - c > 0. && R - c > 0.) { // local minimum
            loc_min.push_back(k + k_step);
            loc.push_back(i + 1);
        }
    }
    loc_min_count = loc_min.size(); // number of local minima
#ifdef CMDL
    std::cout << "Found " << loc_min_count << " candidate(s)." << std::endl;
#endif
    ind.resize(loc_min_count);
    a.resize(loc_min_count);
    b.resize(loc_min_count);

    std::vector<std::vector<double> > xvals(loc_min_count), yvals(loc_min_count);

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
        unsigned ic, N;
        auto slice_x = std::vector<double>(kvals.begin() + il, kvals.begin() + ir + 1);
        auto slice_y = std::vector<double>(rsv.begin() + il, rsv.begin() + ir + 1);
        xvals[i] = slice_x;
        yvals[i] = slice_y;
        N = slice_x.size();
        boost::math::barycentric_rational<double> interp(slice_x.data(), slice_y.data(), N, ord >= N ? N - 1 : ord);
        try {
            loc_min[i] = brent_guess(slice_x.front(), slice_x.back(), loc_min[i], interp(loc_min[i]), interp, eps_mach, ic);
        } catch (std::runtime_error &e) {
#ifdef CMDL
            std::cerr << e.what() << std::endl;
#endif
        }
        for (ir = pos; ir + 1 < n_points_k && rsv[ir] < rsv[ir+1]; ++ir);
        for (il = pos; il > 0 && rsv[il] < rsv[il-1]; --il);
        a[i] = kvals[il];
        b[i] = kvals[ir];
    });
#endif

#ifdef CMDL
    std::cout << "Candidates: ";
    for (const double &c : loc_min)
        std::cout << c << ' ';
    std::cout << std::endl << "Starting local search"
#ifdef DEGREES_OF_FREEDOM
              << " with " << mesh.getNumPanels() << " panels"
#endif
              << "..." << std::endl;
#endif
    // Search for minima in the bracketed regions using Brent method
    unsigned ict = 0;
    double prog = 0;
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
        std::function<double(double)> f = [&](double k) {
            Eigen::MatrixXcd T;
            so.gen_sol_op(builder, k, c_o, c_i, T);
#ifdef WITH_ARNOLDI
            return arnoldi::sv(T, 1, 1e-8)(0);
#else
            return randomized_svd::sv(T, W, q);
#endif
        };
#if 0
        std::function<void(double,double)> rv = [&](double x, double f) {
            std::unique_lock lck(mtx);
            for (double v : xvals[i]) {
                if (std::abs(v - x) <= std::numeric_limits<double>::epsilon())
                    return;
            }
            xvals[i].push_back(x);
            yvals[i].push_back(f);
        };
#endif
        loc_min[i] = brent_gsl(a[i], b[i], loc_min[i], f, acc, ic, &rp);
        if (!std::isfinite(loc_min[i])) {
#ifdef CMDL
            std::unique_lock lck(mtx);
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
    ind.resize(loc_min_count);
#ifdef CMDL
    std::cout << "Total function evaluations: " << ict << std::endl;
#ifdef COMPUTE_DERIVATIVES_AT_MINIMA
    std::cout << "Computing derivatives at local minima..." << std::endl;
    Eigen::ArrayXXd der(loc_min_count, 3);
    for_each (policy, ind.cbegin(), ind.cend(), [&](unsigned int i) {
        Eigen::MatrixXcd T, T_der, T_der_2;
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
