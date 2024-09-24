/**
 * \file roots_newton_polygon_rsvd.cpp
 * \brief This target builds a script that computes minimas in
 * the smallest singular value of the Galerkin BEM
 * approximated solutions operator for the sedond-kind
 * direct BIEs of the Helmholtz transmission problem
 * using the Newton-Raphson method.
 * The results are written to the <tt>data</tt> directory.
 * The script can be run as follows:
 *
 * <tt>
 *  /path/to/roots_newton_square_rsvd \<scatterer filename\> \<refraction inside\>
 *     \<refraction outside\> \<initial wavenumber\> \<\#grid points for root search\>
 *     \<\#panels\> \<quadrature order\> \<accuracy\> \<#subspace iterations\>.
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
#include <Eigen/Core>
#include "parametrized_line.hpp"
//#include "singular_values_arnoldi.hpp"
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

    Eigen::initParallel();

    // define the side of square, refraction index and initial wavenumber
    double c_i = atof(argv[2]);
    double c_o = atof(argv[3]);
    double k_min = atof(argv[4]);

    // define mesh in space and on wavenumber on which to perform verification
    unsigned n_points_k = atoi(argv[5]);
    // read polygonal scatterer from file
    string fname_scatterer = argv[1];
    Eigen::VectorXd poly_x, poly_y;
    if (!scatterer::read(fname_scatterer, poly_x, poly_y)) {
        std::cerr << "Error: failed to read scatterer from file" << std::endl;
        return 1;
    }
    // construction of a ParametrizedMesh object from the vector of panels
    unsigned Npanels;
    double rfac = 0.5; // panel length shrink factor
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
    std::string base_name = "../data/file_roots_newton_polygon_rsvd_";
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
    double k_max = k_min + 1.0, k_step = (k_max - k_min) / (n_points_k - 1);
    size_t disc, loc_min_count;

    // Inform user of started computation.
#ifdef CMDL
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Finding resonances using rSVD approximation and Newton's method." << std::endl;
    std::cout << "Computing on user-defined problem using the specified domain." << std::endl;
    std::cout << std::endl;
#endif

    auto policy = std::execution::par;
    ContinuousSpace<1> cont_space;

    std::vector<size_t> ind(n_points_k);
    std::vector<double> rsv(n_points_k), loc_min, pos_left, val, val_left, val_right;
    std::iota(ind.begin(), ind.end(), 0);
    std::vector<bool> accept;
    std::vector<double> der_left, der_right;
    auto discard_candidates = [&]() {
        for (size_t i = loc_min_count; i--> 0;) {
            if (!accept[i]) {
                pos_left.erase(pos_left.begin() + i);
                der_left.erase(der_left.begin() + i);
                der_right.erase(der_right.begin() + i);
                val.erase(val.begin() + i);
                val_left.erase(val_left.begin() + i);
                val_right.erase(val_right.begin() + i);
                loc_min_count--;
                disc++;
            }
        }
    };

    auto tic = high_resolution_clock::now();

    ParametrizedMesh mesh(scatterer::make_polygonal_panels(poly_x, poly_y, Npanels, rfac));
    int nr = 2 * mesh.getNumPanels();
    auto W = randomized_svd::randGaussian(nr, nc);
    BuilderData builder_data(mesh, cont_space, cont_space, order);
    SolutionsOperator so(builder_data);

    auto sv_der = [&](double k) {
        Eigen::MatrixXcd T, T_der;
        so.gen_sol_op_1st_der(k, c_o, c_i, T, T_der);
        auto res = randomized_svd::sv_der(T, T_der, W, q)(1);
        return res;
    };
    auto sv_der2 = [&](double k) {
        Eigen::MatrixXcd T, T_der, T_der2;
        Eigen::MatrixXd res(1, 2);
        so.gen_sol_op_2nd_der(k, c_o, c_i, T, T_der, T_der2);
        auto v = randomized_svd::sv_der2(T, T_der, T_der2, W, q);
        res(0, 0) = v(1);
        res(0, 1) = v(2);
        return res;
    };

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
            pos_left.push_back(k);
            val.push_back(c);
            val_left.push_back(L);
            val_right.push_back(R);
        }
    }
    disc = 0;
    loc_min_count = pos_left.size(); // number of local minima
#ifdef CMDL
    std::cout << "Found " << loc_min_count << " candidates. Validating..." << std::endl;
#endif
    // Discard candidates which do not approximate local minima
    ind.resize(loc_min_count);
    std::iota(ind.begin(), ind.end(), 0);
    accept.resize(loc_min_count, false);
    der_left.resize(loc_min_count);
    der_right.resize(loc_min_count);
    std::for_each(policy, ind.cbegin(), ind.cend(), [&](size_t i) {
        der_left[i] = sv_der(pos_left[i]);
        der_right[i] = sv_der(pos_left[i] + 2 * k_step);
        accept[i] = der_left[i] < 0. && der_right[i] > 0.;
    });
    if (disc > 0) {
        discard_candidates();
#ifdef CMDL
        std::cout << "Discarded " << disc << " candidate(s)." << std::endl;
#endif
    }
    loc_min.resize(loc_min_count);
    ind.resize(loc_min_count);

    // Refine approximations of local minima by using the interpolated quartic
#ifdef CMDL
    std::cout << "Improving candidates..." << std::endl;
#endif
    std::transform(ind.cbegin(), ind.cend(), loc_min.begin(), [&](size_t i) {
        int status = 0;
        double arg, h = k_step, a = pos_left[i], b = pos_left[i] + 2 * h;
        double f1 = val_left[i], f2 = val_right[i], f0 = val[i];
        double d1 = der_left[i], d2 = der_right[i];
        double scale = 0.25 * pow(h, -4), a2 = pow(a, 2);
        double p1 = a2 * (h * (d2 - d1) - 2. * (f1 + f2 - 2. * f0));
        double p2 = a * h * (h * (3. * d2 - 5. * d1) - (9. * f1 + 7. * f2 - 16. * f0));
        double p3 = pow(h, 2) * (2. * h * (d2 - 4. * d1) - (11. * f1 + 5. * f2 - 16. * f0));
        double A = scale * p1 / a2;
        double B = -scale * (4. * p1 + p2) / a;
        double C = scale * (6. * p1 + 3 * p2 + p3);
        double D = d1 - scale * a * (4. * p1 + 3. * p2 + 2. * p3);
        double E = f1 - a * d1 + scale * a2 * (p1 + p2 + p3);
        BrentMinimizer br_min(a, b, acc);
        arg = br_min.local_min_rc(status, 0.);
        while (status) {
            double p = (((A * arg + B) * arg + C) * arg + D) * arg + E;
            arg = br_min.local_min_rc(status, p);
        }
        return arg;
    });

#ifdef CMDL
    std::cout << "Starting local search..." << std::endl;
#endif
    // Search for minima in the bracketed regions by using Newton-Raphson method
    unsigned ict = 0;
    disc = 0;
    std::for_each(policy, ind.cbegin(), ind.cend(), [&](size_t i) {
        unsigned ic;
        bool rf = false;
        auto fn = [&](double x) {
            if (x == x)
                return double(NAN);
            return loc_min[i];
        };
        loc_min[i] = rtsafe(fn, sv_der2, pos_left[i], pos_left[i] + 2 * k_step, acc, rf, ic);
        ict += ic;
        accept[i] = rf;
        if (!rf) ++disc;
    });
    if (disc > 0) {
        discard_candidates();
        loc_min.resize(loc_min_count);
        ind.resize(loc_min_count);
#ifdef CMDL
        std::cout << "Discarded " << disc << " candidate(s)." << std::endl;
#endif
    }
#ifdef CMDL
    std::cout << "Total Newton iterations taken: " << ict << std::endl;
    std::cout << "Local minima found:" << std::endl;
    for (size_t i = 0; i < loc_min_count; ++i)
        std::cout << loc_min[i] << std::endl;
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
