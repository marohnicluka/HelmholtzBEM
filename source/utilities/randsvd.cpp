/**
 * \file randsvd.cpp
 *
 * \brief This file contains implementation of routines for creating
 * random Gaussian matrices and for approximating the smallest
 * singular value of a given invertible matrix.
 *
 * This file is a part of the HelmholtzTransmissionProblemBEM library.
 *
 * (c) 2023 Luka Marohnić
 */

#include "randsvd.hpp"
#include <complex>
#include <random>
#include <iostream>
#include <chrono>
#if 0
#include "lapacke.h"
#include "complex.h"
#endif

using namespace std::chrono;

namespace randomized_svd {

    static bool _benchmarking = false;
    void benchmarking(bool yes) {
        _benchmarking = yes;
    }

    static unsigned lu_time = 0, qr_time = 0, sub_iter_time = 0, svd_time = 0;
    double get_lu_time() {
        return double(lu_time) * 1e-3;
    }
    double get_qr_time() {
        return double(qr_time) * 1e-3;
    }
    double get_sub_iter_time() {
        return double(sub_iter_time) * 1e-3;
    }
    double get_svd_time() {
        return double(svd_time) * 1e-3;
    }
    void reset_timer() {
        lu_time = qr_time = sub_iter_time = svd_time = 0;
    }

    Eigen::MatrixXcd randGaussian(int nr, int nc) {
        std::random_device rd {};
        std::mt19937 gen { rd() };
        std::uniform_real_distribution<> d { 0, 1 };
        Eigen::MatrixXcd R(nr, nc);
        for (int i = 0; i < nr; ++i) for (int j = 0; j < nc; ++j) {
            R(i, j) = std::complex<double>(d(gen), d(gen));
        }
        return R * M_SQRT1_2;
    }

    Eigen::VectorXcd randGaussian(int n) {
        std::random_device rd {};
        std::mt19937 gen { rd() };
        std::uniform_real_distribution<> d { 0, 1 };
        Eigen::VectorXcd R(n);
        for (int i = 0; i < n; ++i) {
            R(i) = std::complex<double>(d(gen), d(gen));
        }
        return R;
    }

    Eigen::VectorXcd biCGSTAB(const Eigen::MatrixXcd &A, const Eigen::VectorXcd &b, double acc = 1e-6) {
        size_t n = b.rows();
        Eigen::VectorXcd v(n), x(n), t(n), s(n);
        x.setZero();
        Eigen::VectorXcd r = b - A * x, p = r, r0 = randGaussian(n);
        std::complex<double> alpha, omega, rho0 = r0.dot(r), rho1, beta;
        size_t i = 0;
        for (; i < 1000; ++i) {
            v = A * p;
            alpha = rho0 / r0.dot(v);
            x += alpha * p;
            s = r - alpha * v;
            if (s.norm() < acc)
                break;
            t = A * s;
            omega = t.dot(s) / t.dot(t);
            x += omega * s;
            r = s - omega * t;
            if (r.norm() < acc)
                break;
            rho1 = r0.dot(r);
            beta = rho1 / rho0 * alpha / omega;
            p = r + beta * (p - omega * v);
            rho0 = rho1;
        }
        return x;
    }

    double sv(const Eigen::MatrixXcd &T, const Eigen::MatrixXcd &R, int q) {
        int nr = R.rows(), nc = R.cols();
        Eigen::MatrixXcd Q, thinQ;
        thinQ.setIdentity(nr, nc);
        auto tic = high_resolution_clock::now();
        Eigen::PartialPivLU<Eigen::MatrixXcd> lu_decomp(T);
        auto lu_adjoint = lu_decomp.adjoint();
        auto toc = high_resolution_clock::now();
        if (_benchmarking)
            lu_time += duration_cast<milliseconds>(toc - tic).count();
        tic = high_resolution_clock::now();
        Q = lu_decomp.solve(R).colPivHouseholderQr().matrixQ() * thinQ;
        toc = high_resolution_clock::now();
        if (_benchmarking)
            qr_time += duration_cast<milliseconds>(toc - tic).count();
        tic = high_resolution_clock::now();
        for (int i = 0; i < q; ++i) {
            Q = lu_decomp.solve(lu_adjoint.solve(Q).householderQr().householderQ() * thinQ).householderQr().householderQ() * thinQ;
        }
        toc = high_resolution_clock::now();
        if (_benchmarking)
            sub_iter_time += duration_cast<milliseconds>(toc - tic).count();
        tic = high_resolution_clock::now();
        double res;
#if 0
        Q = lu_adjoint.solve(Q);
        char none = 'N';
        double *svd = new double[nr], *rwork = new double[5 * nr];
        lapack_complex_double wkopt, *a, *work;
        int lwork = -1, info, lda = nr, ldu = nr, ldvt = nc;
        a = reinterpret_cast<lapack_complex_double*>(Q.data());
        LAPACK_zgesvd(&none, &none, &nr, &nc, a, &lda, svd, NULL, &ldu, NULL, &ldvt, &wkopt, &lwork, rwork, &info);
        lwork = (int)creal(wkopt);
        work = new lapack_complex_double[lwork];
        LAPACK_zgesvd(&none, &none, &nr, &nc, a, &lda, svd, NULL, &ldu, NULL, &ldvt, work, &lwork, rwork, &info);
        res = 1.0 / svd[0];
        delete[] svd;
        delete[] rwork;
        delete[] work;
#else
        auto svd = lu_adjoint.solve(Q).jacobiSvd();
        res = 1.0 / svd.singularValues()(0);
#endif
        toc = high_resolution_clock::now();
        if (_benchmarking)
            svd_time += duration_cast<milliseconds>(toc - tic).count();
        return res;
    }

    Eigen::Vector2d sv_der(const Eigen::MatrixXcd &T, const Eigen::MatrixXcd &T_der, const Eigen::MatrixXcd &R, int q) {
        unsigned int N = R.rows();
        Eigen::Vector2d res;
        Eigen::MatrixXcd W;
        W.setZero(2 * N, 2 * N);
        W.block(0, N, N, N) = T;
        W.block(N, 0, N, N) = T.adjoint();
        // the smallest singular value
        double s = sv(T, R, q);
        // get the corresponding eigenvector of the Wielandt
        // matrix W as the last column in the matrix Q of
        // a QR factorization of W - s * I
        W.diagonal() -= s * Eigen::VectorXd::Ones(2 * N);
        Eigen::MatrixXcd Q = W.colPivHouseholderQr().matrixQ();
        Eigen::VectorXcd x = Q.col(2 * N - 1), p(2 * N);
        p.head(N) = T_der * x.tail(N);
        p.tail(N) = T_der.adjoint() * x.head(N);
        // compute the derivative of s
        res(0) = s;
        res(1) = x.dot(p).real();
        return res;
    }

    Eigen::Vector3d sv_der2(const Eigen::MatrixXcd &T, const Eigen::MatrixXcd &T_der, const Eigen::MatrixXcd T_der2, const Eigen::MatrixXcd &R, int q) {
        unsigned int N = R.rows();
        Eigen::Vector3d res;
        Eigen::MatrixXcd W;
        W.setZero(2 * N, 2 * N);
        W.block(0, N, N, N) = T;
        W.block(N, 0, N, N) = T.adjoint();
        // smallest singular value
        double s = sv(T, R, q);
        // compute the first derivative of s
        W.diagonal() -= s * Eigen::VectorXd::Ones(2 * N);
        Eigen::MatrixXcd Q = W.colPivHouseholderQr().matrixQ();
        Eigen::VectorXcd x = Q.col(2 * N - 1), u = x, p(2 * N);
        x.normalize();
        p.head(N) = T_der * x.tail(N);
        p.tail(N) = T_der.adjoint() * x.head(N);
        res(0) = s;
        res(1) = x.dot(p).real();
        // compute the second derivative of s
        double temp = 0;
        int m = 5;
        for (unsigned l = 0; l < 2 * N; l++) {
            if (abs(u.coeff(l)) > temp) {
                temp = abs(u.coeff(l));
                m = l;
            }
        }
        m += 1;
        // normalize eigenvector
        u /= u.coeff(m - 1);
        W.col(m - 1) = -u;
        // solve linear system of equations for derivative of eigenvalue and eigenvector
        Eigen::PartialPivLU<Eigen::Ref<Eigen::MatrixXcd> > lu_B(W);
        p.head(N) = T_der * u.tail(N);
        p.tail(N) = T_der.adjoint() * u.head(N);
        Eigen::VectorXcd u_der = -lu_B.solve(p), p2(2 * N);
        auto t = u_der[m - 1];
        u_der[m - 1] = 0;
        p.head(N) = T_der * u_der.tail(N);
        p.tail(N) = T_der.adjoint() * u_der.head(N);
        p2.head(N) = T_der2 * u.tail(N);
        p2.tail(N) = T_der2.adjoint() * u.head(N);
        // solve linear system of equations second derivative of eigenvector and eigenvalue
        res(2) = -lu_B.solve(p2 + 2. * (p - t * u_der))[m - 1].real();
        return res;
    }

}
