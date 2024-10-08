#include <iostream>
#include <chrono>
#include <fstream>
#include <mutex>
#include "singular_values_arnoldi.hpp"
#include "../arpackpp_lib/include/arrscomp.h"
#include "arpp_eig_interface.hpp"
#include "st_vec_storage.hpp"

typedef std::complex<double> complex_t;

#define ARNOLDI_MAXITER std::numeric_limits<int>::max()
#define ARPACK_SILENT_MODE 1

namespace arnoldi {

    static std::mutex mtx;

    Eigen::VectorXd sv(const Eigen::MatrixXcd &T, unsigned count, double acc) {
        const unsigned N = T.cols();
        // precomputation for finding EVs with arpack++
        Eigen::PartialPivLU<Eigen::MatrixXcd> lu = T.partialPivLu();
        auto lu_adjoint = lu.adjoint();
        std::string which = std::to_string(2*count) + "L";
        Eigen::VectorXd res(count);
        std::unique_lock lck(mtx);
        try {
            ARrcCompStdEig<double> prob(2 * N, 2 * count, "LM", 0, acc, ARNOLDI_MAXITER, NULL);
            // iterate until entire space is searched
            while (!prob.ArnoldiBasisFound()) {
                // Calling ARPACK FORTRAN code. Almost all work needed to
                // find an Arnoldi basis is performed by TakeStep.
                prob.TakeStep();

                if ((prob.GetIdo() == 1) || (prob.GetIdo() == -1)) {
                    // Performing matrix-vector multiplication in Eigen.
                    // In regular mode, w = Av must be performed whenever
                    // GetIdo is equal to 1 or -1. GetVector supplies a pointer
                    // to the input vector, v, and PutVector a pointer to the
                    // output vector, w.

                    Eigen::VectorXcd PutVectorEig(2 * N);
                    Eigen::VectorXcd GetVectorEig(2 * N);

                    arpp_to_eig(prob.GetVector(), GetVectorEig);
                    PutVectorEig.segment(0, N) = lu_adjoint.solve(GetVectorEig.segment(N, N));
                    PutVectorEig.segment(N ,N) = lu.solve(GetVectorEig.segment(0, N));
                    eig_to_arpp(PutVectorEig, prob.PutVector());
                }
            }
            // write results into our common data structure
            prob.FindEigenvalues();
            for (unsigned i = 0; i < count; i++) {
                // DON'T USE abs, ONLY std::abs
                // abs inconsistent across platforms, can cast to int
                res[i] = std::abs(1. / prob.Eigenvalue(2 * count - 2 * i - 1).real());
            }
        } catch (ArpackError err) {
#ifdef CMDL
            if (err.Status() == -107)
                std::cout << "Arpack error: maximum number of iterations taken" << std::endl;
            else std::cout << "Arpack error " << err.Status() << std::endl;
#endif
        }
        return res;
    }
    // TODO: resort EVs & vectors after finding Eigenvectors
    Eigen::MatrixXd sv_1st_der(const Eigen::MatrixXcd &T,
                               const Eigen::MatrixXcd &T_der,
                               unsigned count,
                               double acc) {
        // get dimensions of operator
        const unsigned N = T.cols();
        Eigen::VectorXd res_vals(count);
        Eigen::MatrixXcd res_vectors(2*N, count);
        // precomputation for finding EVs with arpack++
        Eigen::PartialPivLU<Eigen::MatrixXcd> lu(T);
        {
            std::unique_lock lck(mtx);
            ARrcCompStdEig<double> prob(2 * N, 2*count, "LM",0,acc,std::numeric_limits<int>::max(), NULL);

            // iterate until entire space is searched
            while (!prob.ArnoldiBasisFound()) {

                // Calling ARPACK FORTRAN code. Almost all work needed to
                // find an Arnoldi basis is performed by TakeStep.
                prob.TakeStep();

                if ((prob.GetIdo() == 1) || (prob.GetIdo() == -1)) {
                    // Performing matrix-vector multiplication in Eigen.
                    // In regular mode, w = Av must be performed whenever
                    // GetIdo is equal to 1 or -1. GetVector supplies a pointer
                    // to the input vector, v, and PutVector a pointer to the
                    // output vector, w.

                    Eigen::VectorXcd PutVectorEig(2 * N);
                    Eigen::VectorXcd GetVectorEig(2 * N);

                    arpp_to_eig(prob.GetVector(), GetVectorEig);
                    PutVectorEig.segment(0, N) = lu.adjoint().solve(GetVectorEig.segment(N, N));
                    PutVectorEig.segment(N ,N) = lu.solve(GetVectorEig.segment(0, N));
                    eig_to_arpp(PutVectorEig, prob.PutVector());
                }
            }
            // write results into our common data structure
            prob.FindEigenvectors();
            prob.FindEigenvalues();
            arpp_to_eig(prob, res_vals, res_vectors);
        }
        Eigen::MatrixXcd W_der = Eigen::MatrixXcd::Zero(2 * N, 2 * N);
        W_der.block(0, N, N, N) = T_der;
        W_der.block(N, 0, N, N) = T_der.adjoint();
        // get eigenvalues and eigenvectors
        // get positive eigenvalue (corresponding to singular value and compute derivative
        Eigen::MatrixXd res(count, 2);
        Eigen::VectorXcd x(2 * N);
        for (unsigned i = 0; i < count; i++) {
            x = res_vectors.col(i).normalized();
            if (res_vals[i] < 0) {
                res(i, 0) = -res_vals[i];
                res(i, 1) = -(x.dot(W_der * x)).real();
            } else {
                res(i, 0) = res_vals[i];
                res(i, 1) = (x.dot(W_der * x)).real();
            }
        }
        return res;
    }

    // TODO: resort EVs & vectors after finding Eigenvectors
    Eigen::MatrixXd sv_2nd_der(const Eigen::MatrixXcd &T,
                               const Eigen::MatrixXcd &T_der,
                               const Eigen::MatrixXcd &T_der2,
                               unsigned count,
                               double acc) {
        // get dimensions of operator
        const unsigned N = T.cols();
        Eigen::VectorXd res_vals(count);
        Eigen::MatrixXcd res_vectors(2 * N, count);
        // precomputation for finding EVs with arpack++
        Eigen::PartialPivLU<Eigen::MatrixXcd> lu(T);
        {
            std::unique_lock lck(mtx);
            ARrcCompStdEig<double> prob(2 * N, 2*count, "LM", 0, acc, std::numeric_limits<int>::max(), NULL);

            // iterate until entire space is searched
            while (!prob.ArnoldiBasisFound()) {

                // Calling ARPACK FORTRAN code. Almost all work needed to
                // find an Arnoldi basis is performed by TakeStep.
                prob.TakeStep();

                if ((prob.GetIdo() == 1) || (prob.GetIdo() == -1)) {
                    // Performing matrix-vector multiplication in Eigen.
                    // In regular mode, w = Av must be performed whenever
                    // GetIdo is equal to 1 or -1. GetVector supplies a pointer
                    // to the input vector, v, and PutVector a pointer to the
                    // output vector, w.

                    Eigen::VectorXcd PutVectorEig(2 * N);
                    Eigen::VectorXcd GetVectorEig(2 * N);

                    arpp_to_eig(prob.GetVector(), GetVectorEig);
                    PutVectorEig.segment(0, N) = lu.adjoint().solve(GetVectorEig.segment(N, N));
                    PutVectorEig.segment(N ,N) = lu.solve(GetVectorEig.segment(0, N));
                    eig_to_arpp(PutVectorEig, prob.PutVector());
                }
            }
            // write results into our common data structure
            prob.FindEigenvectors();
            arpp_to_eig(prob, res_vals, res_vectors);
        }
        // build Wielandt matrix
        Eigen::MatrixXcd W = Eigen::MatrixXcd::Zero(2 * N, 2 * N);
        W.block(0, N, N, N) = T;
        W.block(N, 0, N, N) = T.adjoint();
        Eigen::MatrixXcd W_der = Eigen::MatrixXcd::Zero(2 * N, 2 * N);
        W_der.block(0, N, N, N) = T_der;
        W_der.block(N, 0, N, N) = T_der.adjoint();
        Eigen::MatrixXcd W_der2 = Eigen::MatrixXcd::Zero(2 * N, 2 * N);
        W_der2.block(0, N, N, N) = T_der2;
        W_der2.block(N, 0, N, N) = T_der2.adjoint();
        // get positive eigenvalue (corresponding to singular value) and compute derivative
        Eigen::MatrixXd res(count, 3);
        Eigen::MatrixXcd B(2 * N, 2 * N);
        Eigen::VectorXcd r(2 * N);
        Eigen::VectorXcd s(2 * N);
        Eigen::VectorXcd u_der(2 * N);
        Eigen::VectorXcd u_der_temp(2 * N);
        Eigen::VectorXcd u_der2(2 * N);
        for (unsigned i = 0; i < count; i++) {
            // choose which entry of eigenvector to normalize
            double temp = 0;
            int m = 5;
            for (unsigned l = 0; l < 2 * N; l++) {
                if (abs(res_vectors.coeff(l,i)) > temp) {
                    temp = abs(res_vectors.coeff(l,i));
                    m = l;
                }
            }
            m += 1;
            complex_t rescale = res_vectors.coeff(m - 1, i);
            // normalize eigenvector
            Eigen::VectorXcd u = res_vectors.col(i) / rescale;
            // build matrix with deleted column from Wielandt matrix and eigenvector
            Eigen::VectorXcd diag = res_vals[i] * Eigen::VectorXcd::Ones(2 * N);
            W -= diag.asDiagonal();
            B.block(0, 0, 2 * N, m - 1) = W.block(0, 0, 2 * N, m - 1);
            B.block(0, m - 1, 2 * N, 2 * N - m) = W.block(0, m, 2 * N, 2 * N - m);
            B.col(2 * N - 1) = -u;
            if (i < count - 1)
                W += diag.asDiagonal();
            // compute right hand side
            r = W_der * u;
            // solve linear system of equations for derivative of eigenvalue and eigenvector
            Eigen::PartialPivLU<Eigen::MatrixXcd> lu_B(B);
            u_der_temp = -lu_B.solve(r);
            complex_t ev_der = u_der_temp[2 * N - 1];
            // construct eigenvector using derivative of normalization condition
            u_der.segment(0, m - 1) = u_der_temp.segment(0, m - 1);
            u_der[m - 1] = 0;
            u_der.segment(m, 2 * N - m) = u_der_temp.segment(m - 1, 2 * N - m);
            // compute right hand side for second derivative
            diag = ev_der * Eigen::VectorXcd::Ones(2 * N);
            W_der -= diag.asDiagonal();
            s = W_der2 * u + 2. * (W_der * u_der);
            if (i < count - 1)
                W_der += diag.asDiagonal();
            // solve linear system of equations second derivative of eigenvector and eigenvalue
            u_der2 = -lu_B.solve(s);
            complex_t ev_der2 = u_der2[2 * N - 1];
            // check sign of eigenvalue of Wielandt matrix to retrieve correct eigenvalue of T
            if (res_vals[i] > 0) {
                res(i, 0) = res_vals[i];
                res(i, 1) = ev_der.real();
                res(i, 2) = ev_der2.real();
            } else {
                res(i, 0) = -res_vals[i];
                res(i, 1) = -ev_der.real();
                res(i, 2) = -ev_der2.real();
            }
        }
        return res;
    }

    double der_by_ext(std::function<double(double)> f, double x, double h0, double rtol, double atol) {
        const unsigned nit = 2; // Maximum depth of extrapolation
        Eigen::VectorXd h(nit);
        h[0] = h0; // Widths of difference quotients
        Eigen::VectorXd y(nit); // Approximations returned by difference quotients
        y[0] = (f(x + h0) - f(x - h0)) / (2 * h0); // Widest difference quotients
        // using Aitken-Neville scheme with x = 0, see Code 5.2.3.10
        for (unsigned i = 1; i < nit; ++i) {
            // create data points for extrapolation
            h[i] = h[i - 1] / 2; // Next width half as big
            y[i] = (f(x + h[i]) - f(x - h[i])) / h(i - 1);
            // Aitken-Neville update
            for (int k = i - 1; k >= 0; --k)
                y[k] = y[k + 1] - (y[k + 1] - y[k]) * h[i] / (h[i] - h[k]);
            // termination of extrapolation when desired tolerance is reached
            const double errest = std::abs(y[1] - y[0]); // error indicator
            if (errest < rtol * std::abs(y[0]) || errest < atol) //
                break;
        }
        return y[0]; // Return value extrapolated from largest number ofdifference quotients
    }
}
