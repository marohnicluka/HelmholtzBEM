/**
 * \file h2lib_interface.cpp
 *
 * \brief This file contains the implementation of the
 * H2Lib interface.
 *
 * (c) 2024 Luka Marohnić
 */

#include "h2lib_interface.hpp"
#include <numeric>
extern "C" {
#include "aca.h"
#include "basic.h"
}

namespace h2lib {

    void init() {
        init_h2lib(NULL, NULL);
    }

    void uninit() {
        uninit_h2lib();
    }

    void matrix_entry_callback(const uint *ridx, const uint *cidx, void *data, const bool ntrans, pamatrix N) {
        const MatrixEntry &mat = *static_cast<MatrixEntry*>(data);
        size_t i, j;
        std::complex<double> elem;
        for (i = 0; i < N->rows; ++i) {
            for (j = 0; j < N->cols; ++j) {
                elem = mat(ridx[ntrans ? j : i], cidx[ntrans ? i : j]);
                setentry_amatrix(N, i, j, elem.real() + elem.imag() * I);
            }
        }
    }

    size_t aca_with_partial_pivoting(MatrixEntry &entry,
                                     const std::pair<size_t,size_t> &rowspan,
                                     const std::pair<size_t,size_t> &colspan,
                                     double eps,
                                     Eigen::MatrixXcd &U,
                                     Eigen::MatrixXcd &V) {
        uint rows = rowspan.second, cols = colspan.second;
        std::vector<uint> ridx(rows), cidx(cols);
        std::iota(ridx.begin(), ridx.end(), rowspan.first);
        std::iota(cidx.begin(), cidx.end(), colspan.first);
        prkmatrix R = new_rkmatrix(rows, cols, 1);
        decomp_partialaca_rkmatrix(matrix_entry_callback, static_cast<void*>(&entry), ridx.data(), rows, cidx.data(), cols, eps, NULL, NULL, R);
        uint rank = getrank_rkmatrix(R);
        U.resize(rows, rank);
        V.resize(cols, rank);
        pamatrix A = getA_rkmatrix(R), B = getB_rkmatrix(R);
        for (uint j = 0; j < rank; ++j) {
            for (uint i = 0; i < rows; ++i) {
                field elem = getentry_amatrix(A, i, j);
                U(i, j) = std::complex<double>(REAL(elem), IMAG(elem));
            }
            for (uint i = 0; i < cols; ++i) {
                field elem = getentry_amatrix(B, i, j);
                V(i, j) = std::complex<double>(REAL(elem), IMAG(elem));
            }
        }
        del_rkmatrix(R);
        return rank;
    }

}
