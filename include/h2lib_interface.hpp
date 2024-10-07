/**
 * \file h2lib_interface.hpp
 * \brief This file defines the interface to the H2Lib library.
 *
 * This File is a part of the HelmholtzTransmissionBEM
 *
 * (c) 2024 Luka Marohnić
 */

#ifndef H2LIB_INTERFACEHPP
#define H2LIB_INTERFACEHPP

#include <Eigen/Dense>

namespace h2lib {

    typedef std::function<std::complex<double>(size_t,size_t)> MatrixEntry;

    /**
     * Initialize H2Lib.
     */
    void init();

    /**
     * Uninitialize H2Lib.
     */
    void uninit();

    /**
     * This routine computes the adaptive cross approximation
     * UV* using partial pivoting of an implicitly given matrix A.
     * It calls decomp_partialaca_rkmatrix from the aca module
     * in H2Lib. See H2Lib documentation for details.
     *
     * @param A function which computes a single entry of the matrix A
     * @param rowspan row span
     * @param colspan column span
     * @param eps accuracy: |A-UV*|<=eps*|A|
     * @param U the resulting matrix U
     * @param V the resulting matrix V
     */
    size_t aca_with_partial_pivoting(MatrixEntry &entry,
                                     const std::pair<size_t,size_t> &rowspan,
                                     const std::pair<size_t,size_t> &colspan,
                                     double eps,
                                     Eigen::MatrixXcd &U,
                                     Eigen::MatrixXcd &V);

}


#endif // H2LIB_INTERFACEHPP
