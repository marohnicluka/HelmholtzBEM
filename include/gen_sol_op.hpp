/**
 * \file gen_sol_op.hpp
 * \brief This file contains functions that compute the approximation
 * of the operator and it's first two derivatives of the second-kind direct BIEs
 * for the Helmholtz transmission problemi using Galerkin BEM.
 *
 * This File is a part of the HelmholtzTransmissionProblemBEM library
 * (c) 2023 Luka Marohnić
 */

#ifndef GEN_SOL_OPHPP
#define GEN_SOL_OPHPP

#include "galerkin_builder.hpp"

class SolutionsOperator
{
    const BuilderData &builder_data;
    bool do_projection;
    Eigen::PartialPivLU<Eigen::MatrixXcd> lu;
    Eigen::MatrixXd M; // mass matrix
    Eigen::MatrixXcd K_i, K_o, V_i, V_o, W_i, W_o;
    size_t dim_test, dim_trial;
    // solutions operator matrix assembly routines
    void gen_sol_op_in(GalerkinBuilder &builder, const std::complex<double> &k, double c_o, double c_i,
                       Eigen::MatrixXcd &T);
    void gen_sol_op_1st_der_in(GalerkinBuilder &builder, const std::complex<double> &k, double c_o, double c_i,
                               Eigen::MatrixXcd &T, Eigen::MatrixXcd &T_der);
    void gen_sol_op_2nd_der_in(GalerkinBuilder &builder, const std::complex<double> &k, double c_o, double c_i,
                               Eigen::MatrixXcd &T, Eigen::MatrixXcd &T_der, Eigen::MatrixXcd &T_der2);

public:
    /**
     * Initialize solutions operator class.
     * @param builder_data_in bulder data object
     * @param profiling_in whether to do time profiling
     */
    SolutionsOperator(const BuilderData &builder_data_in, bool proj = true);
    // destructor
    ~SolutionsOperator() { }
    /**
     * Compute approximation of solutions operator for second-kind direct BIEs of
     * Helmholtz transmission problem using Galerkin BEM.
     * This routine will create a temporary Galerkin matrix builder.
     * @param k wavenumber (complex)
     * @param c_o refraction index of outer domain (should be 1)
     * @param c_i refraction indef of inner domain (must not be smaller than c_o)
     * @param T complex matrix to which the solutions operator matrix will be stored
     */
    void gen_sol_op(const std::complex<double> &k, double c_o, double c_i, Eigen::MatrixXcd &T);
    /**
     * Compute approximation of solutions operator for second-kind direct BIEs of
     * Helmholtz transmission problem using Galerkin BEM.
     * This routine will use the given Galerkin matrix builder.
     * @param builder Galerkin matrix builder
     * @param k wavenumber (complex)
     * @param c_o refraction index of outer domain (should be 1)
     * @param c_i refraction indef of inner domain (must not be smaller than c_o)
     * @param T complex matrix to which the solutions operator matrix will be stored
     */
    void gen_sol_op(GalerkinBuilder &builder, const std::complex<double> &k, double c_o, double c_i, Eigen::MatrixXcd &T);
    /**
     * Compute approximation of solutions operator and its 1st derivative
     * for second-kind direct BIEs of Helmholtz transmission problem using Galerkin BEM.
     * This routine will create a temporary Galerkin matrix builder.
     * @param k wavenumber (complex)
     * @param c_o refraction index of outer domain (should be 1)
     * @param c_i refraction indef of inner domain (must not be smaller than c_o)
     * @param T complex matrix to which the solutions operator matrix will be stored
     * @param T_der complex matrix to which the 1st derivative of the solutions operator matrix will be stored
     */
    void gen_sol_op_1st_der(const std::complex<double> &k, double c_o, double c_i,
                            Eigen::MatrixXcd &T, Eigen::MatrixXcd &T_der);
    /**
     * Compute approximation of solutions operator and its 1st derivative
     * for second-kind direct BIEs of Helmholtz transmission problem using Galerkin BEM.
     * This routine will use the given Galerkin matrix builder.
     * @param builder Galerkin matrix builder
     * @param k wavenumber (complex)
     * @param c_o refraction index of outer domain (should be 1)
     * @param c_i refraction indef of inner domain (must not be smaller than c_o)
     * @param T complex matrix to which the solutions operator matrix will be stored
     * @param T_der complex matrix to which the 1st derivative of the solutions operator matrix will be stored
     */
    void gen_sol_op_1st_der(GalerkinBuilder &builder, const std::complex<double> &k, double c_o, double c_i,
                            Eigen::MatrixXcd &T, Eigen::MatrixXcd &T_der);
    /**
     * Compute approximation of solutions operator and its 1st and 2nd derivatives
     * for second-kind direct BIEs of Helmholtz transmission problem using Galerkin BEM.
     * This routine will create a temporary Galerkin matrix builder.
     * @param k wavenumber (complex)
     * @param c_o refraction index of outer domain (should be 1)
     * @param c_i refraction indef of inner domain (must not be smaller than c_o)
     * @param T complex matrix to which the solutions operator matrix will be stored
     * @param T_der complex matrix to which the 1st derivative of the solutions operator matrix will be stored
     * @param T_der2 complex matrix to which the 2nd derivative of the solutions operator matrix will be stored
     */
    void gen_sol_op_2nd_der(const std::complex<double> &k, double c_o, double c_i,
                            Eigen::MatrixXcd &T, Eigen::MatrixXcd &T_der, Eigen::MatrixXcd &T_der2);
    /**
     * Compute approximation of solutions operator and its 1st and 2nd derivatives
     * for second-kind direct BIEs of Helmholtz transmission problem using Galerkin BEM.
     * This routine will use the given Galerkin matrix builder.
     * @param builder Galerkin matrix builder
     * @param k wavenumber (complex)
     * @param c_o refraction index of outer domain (should be 1)
     * @param c_i refraction indef of inner domain (must not be smaller than c_o)
     * @param T complex matrix to which the solutions operator matrix will be stored
     * @param T_der complex matrix to which the 1st derivative of the solutions operator matrix will be stored
     * @param T_der2 complex matrix to which the 2nd derivative of the solutions operator matrix will be stored
     */
    void gen_sol_op_2nd_der(GalerkinBuilder &builder, const std::complex<double> &k, double c_o, double c_i,
                            Eigen::MatrixXcd &T, Eigen::MatrixXcd &T_der, Eigen::MatrixXcd &T_der2);
    /**
     * Return reference to the mass matrix.
     */
    const Eigen::MatrixXd &mass_matrix() const { return M; }
    const Eigen::MatrixXcd &K_int() const { return K_i; }
    const Eigen::MatrixXcd &K_ext() const { return K_o; }
    const Eigen::MatrixXcd &V_int() const { return V_i; }
    const Eigen::MatrixXcd &V_ext() const { return V_o; }
    const Eigen::MatrixXcd &W_int() const { return W_i; }
    const Eigen::MatrixXcd &W_ext() const { return W_o; }
    Eigen::MatrixXcd projection(const Eigen::MatrixXcd &T) const;
    const BuilderData &getBuilderData() const { return builder_data; }
};

#endif //GEN_SOL_OPHPP
