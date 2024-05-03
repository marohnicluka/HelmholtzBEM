/**
 * \file rsvd_test.cpp
 * \brief This target builds a script that compares
 * rSVD and Arnoldi curves.
 *
 * This File is a part of the HelmholtzTransmissionProblemBEM library.
 *
 * (c) 2023 Luka Marohnić
 */

#include <complex>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <execution>
#include <algorithm>
#include <string>
#include <thread>
#include "parametrized_line.hpp"
#include "singular_values_arnoldi.hpp"
#include "find_roots.hpp"
#include "gen_sol_op.hpp"
#include "randsvd.hpp"
#include "find_roots.hpp"
#include "continuous_space.hpp"

// define shorthand for time benchmarking tools, complex data type and immaginary unit
using namespace std::chrono;

#define COMPUTE_ARNOLDI 1
#define COMPUTE_RSVD 1
//#define WITH_MULTIPLE_THREADS 1

int main(int argc, char** argv) {

    // check whether we have the correct number of input arguments
    if (argc < 10)
        throw std::runtime_error("Too few input arguments!");
    if (argc > 10)
        throw std::runtime_error("Too many input arguments!");

    // define radius of circle refraction index and initial wavenumber
    double eps = atof(argv[1]);
    double c_i = atof(argv[2]);
    double c_o = atof(argv[3]);
    double k_min = atof(argv[4]);

    // define mesh in space and on wavenumber on which to perform verification
    unsigned n_points_k = atoi(argv[5]);
    unsigned numpanels;
    numpanels = atoi(argv[6]);
    // compute mesh for numpanels
    using PanelVector = PanelVector;
    // corner points for the square
    Eigen::RowVectorXd x1(2);
    x1 << 0,0; // point (0,0)
    Eigen::RowVectorXd x2(2);
    x2 << eps, 0; // point (1,0)
    Eigen::RowVectorXd x3(2);
    x3 << eps, eps; // point (1,0.5)
    Eigen::RowVectorXd x4(2);
    x4 << 0, eps; // point (0,1.5)
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

    // define order of quadrature rule used to compute matrix entries and which singular value to evaluate
    unsigned order = atoi(argv[7]);

    // define accurracy of arnoldi algorithm
    double acc = std::max(atof(argv[8]), std::numeric_limits<double>::epsilon());

    // define the number of subspace iterations
    int q = atoi(argv[9]);

    std::string mfile = "../data/rsvd_test.m";
    std::ofstream file_out;
    file_out.open(mfile, std::ofstream::out | std::ofstream::trunc);
    file_out.close();

    int nc = 2;
    int nr = 2 * numpanels;
    Eigen::MatrixXcd W = randomized_svd::randGaussian(nr, nc);

    double k_max = k_min + 1., k_step = (k_max - k_min) / (n_points_k - 1.);

    std::vector<size_t> ind(n_points_k);
    std::vector<double> rsv(n_points_k), asv(n_points_k);
    std::vector<double>::const_iterator it;

    // create objects for assembling solutions operator and its derivatives
    ContinuousSpace<1> cont_space;
    BuilderData builder_data(mesh, cont_space, cont_space, order);
    SolutionsOperator so(builder_data);

    std::iota(ind.begin(), ind.end(), 0);

    unsigned assembly_time = 0, rsv_time = 0, asv_time = 0;

#ifdef WITH_MULTIPLE_THREADS
    auto policy = std::execution::par;
    unsigned n_threads = std::thread::hardware_concurrency();
#ifdef CMDL
    std::cout << "Working with " << n_threads << " threads concurrently" << std::endl;
#endif
#else
    auto policy = std::execution::seq;
    unsigned n_threads = 1;
    GalerkinBuilder builder(builder_data);
#endif
    unsigned n_batches = n_points_k / n_threads;

    auto tic_all = high_resolution_clock::now();
    for (unsigned ib = 0; ib < n_batches; ++ib) {
        unsigned bstart = ib * n_threads, bsize = ib < n_batches - 1 ? n_threads : n_points_k - bstart;
        auto start = ind.cbegin() + bstart;
        auto end = ib < n_batches - 1 ? ind.cbegin() + (ib + 1) * n_threads : ind.cend();
        Eigen::MatrixXcd Tall(nr, nr * bsize);
#ifdef CMDL
        std::cout << std::endl << "Processing batch " << ib + 1 << " of " << n_batches << "..." << std::endl;
#endif
        auto tic = high_resolution_clock::now();
        std::for_each(policy, start, end, [&](size_t i) {
            Eigen::MatrixXcd T_in;
#ifdef WITH_MULTIPLE_THREADS
            so.gen_sol_op(k_min + k_step * i, c_o, c_i, T_in);
#else
            so.gen_sol_op(builder, k_min + k_step * i, c_o, c_i, T_in);
#endif
            Tall.block(0, (i - bstart) * nr, nr, nr) = T_in;
        });
        auto toc = high_resolution_clock::now();
        auto dur_assembly = duration_cast<milliseconds>(toc - tic);
        assembly_time += dur_assembly.count();
        tic = high_resolution_clock::now();
#ifdef COMPUTE_ARNOLDI
        std::for_each(start, end, [&](size_t i) {
            const Eigen::MatrixXcd &T = Tall.block(0, (i - bstart) * nr, nr, nr);
            asv[i] = arnoldi::sv(T, 1, acc)(0);
        });
#endif
        toc = high_resolution_clock::now();
        auto dur_asv = duration_cast<milliseconds>(toc - tic);
        asv_time += dur_asv.count();
        tic = high_resolution_clock::now();
#ifdef COMPUTE_RSVD
        std::for_each(policy, start, end, [&](size_t i) {
            const Eigen::MatrixXcd &T = Tall.block(0, (i - bstart) * nr, nr, nr);
            rsv[i] = randomized_svd::sv(T, W, q);
        });
#endif
        toc = high_resolution_clock::now();
        auto dur_rsv = duration_cast<milliseconds>(toc - tic);
        rsv_time += dur_rsv.count();
#ifdef CMDL
        std::cout << "Elapsed time (assembly, Arnoldi, RSVD) [sec]:\t"
                  << 1e-3 * dur_assembly.count() << '\t'
                  << 1e-3 * dur_asv.count() << '\t'
                  << 1e-3 * dur_rsv.count() << std::endl;
#endif
    }
    auto toc_all = high_resolution_clock::now();
    auto dur_all = duration_cast<milliseconds>(toc_all - tic_all);
#ifdef CMDL
    std::cout << std::endl << "*Finished*"
              << std::endl << "Total time: " << 1e-3 * dur_all.count() << " sec" << std::endl
              << "Assembly time: " << (100 * assembly_time) / dur_all.count() << "%" << std::endl
              << "Arnoldi time: " << 1e-3 * asv_time << " sec" << std::endl
              << "RSVD time: " << 1e-3 * rsv_time << " sec" << std::endl
              << "Arnoldi time / RSVD time: " << double(asv_time) / double(rsv_time) << std::endl;
#endif
    std::vector<double> err(n_points_k);
    for (size_t i = 0; i < n_points_k; ++i) {
        err[i] = std::abs(asv[i] - rsv[i]) / std::abs(asv[i]);
    }
    auto err_sorted = err;
    std::sort(err_sorted.begin(), err_sorted.end());
    std::cout << "Error: min " << err_sorted[0] << ", max " << err_sorted[n_points_k - 1]
              << ", median " << err_sorted[n_points_k / 2] << std::endl;
#ifndef WITH_MULTIPLE_THREADS
    std::cout << std::endl;
    std::cout << "LU decomposition time: " << randomized_svd::get_lu_time() << std::endl
              << "QR decomposition time: " << randomized_svd::get_qr_time() << std::endl
              << "Subspace iterations time: " << randomized_svd::get_sub_iter_time() << std::endl
              << "SVD time: " << randomized_svd::get_svd_time() << std::endl
              << "Interaction matrix assembly time: " << builder.getInteractionMatrixAssemblyTime() * 1e-6 << std::endl
              << "Hankel computation time: " << builder.getHankelComputationTime() * 1e-6 << std::endl;
#endif

    file_out.open(mfile, std::ios_base::app);
    file_out << std::setprecision(18);
    file_out << "k = [";
    for (size_t i = 0; i < n_points_k; ++i) {
        if (i > 0)
            file_out << ",";
        file_out << k_min + i * k_step;
    }
    file_out << "];" << std::endl << "rsv = [";
    for (it = rsv.begin(); it != rsv.end(); ++it) {
        if (it != rsv.begin())
            file_out << ",";
        file_out << *it;
    }
    file_out << "];" << std::endl;
#ifdef COMPUTE_ARNOLDI
    file_out << "asv = [";
    for (it = asv.begin(); it != asv.end(); ++it) {
        if (it != asv.begin())
            file_out << ",";
        file_out << *it;
    }
    file_out << "];" << std::endl;
#endif
    file_out << "err = [";
    for (it = err.begin(); it != err.end(); ++it) {
        if (it != err.begin())
            file_out << ",";
        file_out << *it;
    }
    file_out << "];" << std::endl;
    file_out.close();

    return 0;
}
