/**
 * \file galerkin_builder.hpp
 * \brief This file declares the object to evaluate the entries of
 *        Galerkin matrices by using common and composite
 *        Gauss-Legendre quadrature rules.
 *
 * This File is a part of the HelmholtzTransmissionProblemBEM library.
 * (c) 2023 Luka Marohnić
 */

#ifndef GALERKIN_BUILDERHPP
#define GALERKIN_BUILDERHPP

#include <map>
#include <set>
#include "abstract_bem_space.hpp"
#include "abstract_parametrized_curve.hpp"
#include "gauleg.hpp"
#include "cbessel.hpp"

#define HANKEL_VECTORIZE 1

class BuilderData {
    // panel vector
    PanelVector panels;
    // dimensions
    size_t numpanels, Qtest, Qtrial, dim_test, dim_trial;
    bool _panels_are_lines;
public:
    // trial and test spaces
    const ParametrizedMesh &mesh;
    const AbstractBEMSpace &test_space;
    const AbstractBEMSpace &trial_space;
    // quadrature rules
    QuadRule GaussQR;
    QuadRule CGaussQR;
    /**
     * Construct the Galerkin builder data object from the given parameters.
     * @param mesh panel mesh
     * @param test_space_in the test space
     * @param trial_space_in the trial space
     * @param order the order of Gaussian quadrature
     */
    BuilderData(const ParametrizedMesh &mesh_in,
                const AbstractBEMSpace &test_space_in,
                const AbstractBEMSpace &trial_space_in,
                unsigned order);
    /**
     * Return the dimension of the test space.
     * @return unsigned int, the dimension
     */
    size_t getTestSpaceDimension() const { return dim_test; }
    /**
     * Return the dimension of the trial space.
     * @return unsigned int, the dimension
     */
    size_t getTrialSpaceDimension() const { return dim_trial; }
    /**
     * Return TRUE iff test and trial spaces are equal (i.e. the same space).
     * @return boolean true if spaces are equal, false otherwise
     */
    bool testTrialSpacesAreEqual() const { return &test_space == &trial_space; }
    /**
     * Return TRUE iff i-th and j-th panels are adjacent.
     * @return boolean true if panels are adjacent, false otherwise
     */
    bool is_adjacent(size_t i, size_t j, bool &swap) const;
    // data
    Eigen::ArrayXXd  m_tc, m_ta, m_tg, m_sc, m_sa, m_sg, m_wc, m_wa, m_wg,
                     m_dlp_cnc_fg, m_dlp_adj_fg, m_dlp_adj_fg_swap, m_dlp_gen_fg,
                     m_dlp_cnc_fg_t, m_dlp_adj_fg_t,
                     m_dlp_adj_fg_swap_t, m_dlp_gen_fg_t,
                     m_slp_cnc_fg, m_slp_adj_fg, m_slp_adj_fg_swap, m_slp_gen_fg,
                     m_hyp_cnc_fg, m_hyp_adj_fg, m_hyp_adj_fg_swap, m_hyp_gen_fg,
                     m_hyp_cnc_fg_arc, m_hyp_adj_fg_arc,
                     m_hyp_adj_fg_arc_swap, m_hyp_gen_fg_arc,
                     Derivative_01_sc_n, Derivative_01_tc_n, Derivative_01_sa_n, Derivative_01_ta_n, Derivative_01_sg_n, Derivative_01_tg_n,
                     Derivative_01_swapped_sa_n, Derivative_01_swapped_ta_n;
    Eigen::ArrayXXcd Derivative_01_sc, Derivative_01_tc, Derivative_01_sa, Derivative_01_ta, Derivative_01_sg, Derivative_01_tg,
                     Derivative_01_swapped_sa, Derivative_01_swapped_ta, op_sc, op_tc, op_sa, op_ta, op_sg, op_tg, swapped_op_sa, swapped_op_ta;
    // methods
    size_t getQtest() const { return Qtest; }
    size_t getQtrial() const { return Qtrial; }
    size_t getGaussQROrder() const { return GaussQR.n; }
    size_t getCGaussQROrder() const { return CGaussQR.n; }
    size_t getNumPanels() const { return numpanels; }
    bool getPanelsAreLines() const { return _panels_are_lines; }
};

class GalerkinBuilder {
    // builder data object
    const BuilderData &data;
    // wavenumber and refraction index
    std::complex<double> k, ksqrtc, kkc;
    double c, sqrtc, ksqrtca;
    bool k_real_positive;
    bool linp;
    bool tdottp_zero;
    // timing info
    unsigned interaction_matrix_assembly_time;
    unsigned hankel_computation_time;
    unsigned panel_interaction_data_time;
#ifdef HANKEL_VECTORIZE
    // Hankel function computation
    complex_bessel::Hankel1Real01 H1R01, H1R01_s;
#endif
    // workspace
    Eigen::ArrayXXd m_zero, m_zero_s;
    Eigen::ArrayXXcd m_h0[2], m_h1[2], m_v[2], m_tangent[2], m_tangent_p[2], m_1[2], m_2[2], m_3[2];
    Eigen::ArrayXXcd m_h0_s, m_h1_s, m_v_s, m_tangent_s, m_tangent_p_s, m_1_s, m_2_s, m_3_s;
    Eigen::ArrayXXd m_v_norm[2], m_v_norm2[2], m_tangent_norm[2], m_tangent_p_norm[2];
    Eigen::ArrayXXd m_v_norm_s, m_v_norm2_s, m_tangent_norm_s, m_tangent_p_norm_s;
    Eigen::ArrayXXd vdotn, tdottp, ttp_norm, cfr1, cfr2, cfr3, temp;
    Eigen::ArrayXXd vdotn_s, tdottp_s, ttp_norm_s, cfr1_s, cfr2_s, cfr3_s, temp_s;
    Eigen::ArrayXXcd cf, cf_s, m_1_hg, m_1_hg_s, m_2_g, m_3_g, m_2_g_s, m_3_g_s;
    // interaction matrices
    Eigen::ArrayXXcd double_layer_interaction_matrix, hypersingular_interaction_matrix, single_layer_interaction_matrix;
    Eigen::ArrayXXcd double_layer_der_interaction_matrix, hypersingular_der_interaction_matrix, single_layer_der_interaction_matrix;
    Eigen::ArrayXXcd double_layer_der2_interaction_matrix, hypersingular_der2_interaction_matrix, single_layer_der2_interaction_matrix;
    // assembled matrices
    Eigen::MatrixXcd double_layer_matrix, hypersingular_matrix, single_layer_matrix;
    Eigen::MatrixXcd double_layer_der_matrix, hypersingular_der_matrix, single_layer_der_matrix;
    Eigen::MatrixXcd double_layer_der2_matrix, hypersingular_der2_matrix, single_layer_der2_matrix;
    // storage for sparse computation
    std::map<std::pair<size_t,size_t>,std::complex<double> > dlp_sparse, hyp_sparse, slp_sparse;
    std::set<std::pair<size_t,size_t> > computed_imat_indices;
    // Local-global map
    unsigned test_space_map(unsigned ii, unsigned i) { return data.test_space.LocGlobMap(ii + 1, i + 1, data.getTestSpaceDimension()) - 1; }
    unsigned trial_space_map(unsigned ii, unsigned i) { return data.trial_space.LocGlobMap(ii + 1, i + 1, data.getTrialSpaceDimension()) - 1; }
    // routines for computing shared data
    void compute_coinciding(size_t i) throw();
    void compute_adjacent(size_t i, size_t j, bool swap) throw();
    void compute_general(size_t i, size_t j) throw();
    void compute_tdottp(size_t K);
    void compute_tdottp_s();
    // routines for interaction matrix construction
    void double_layer_coinciding(int der) throw();
    void double_layer_adjacent(bool swap, int der, bool transp) throw();
    void double_layer_general(int der, bool transp) throw();
    void hypersingular_coinciding(int der) throw();
    void hypersingular_adjacent(bool swap, int der) throw();
    void hypersingular_general(int der) throw();
    void single_layer_coinciding(int der) throw();
    void single_layer_adjacent(bool swap, int der) throw();
    void single_layer_general(int der) throw();
    void all_coinciding(int der) throw();
    void all_adjacent(bool swap, int der) throw();
    void all_general(int der) throw();
    void initialize_parameters(const std::complex<double> &k_in, double c_in);
    void iH1_01_cplx(const Eigen::ArrayXXcd &x, Eigen::ArrayXXcd &h_0, Eigen::ArrayXXcd &h_1);

public:
    /**
     * Construct the Galerkin matrix builder from the given parameters.
     * @param builder_data the Galerkin builder data object
     */
    GalerkinBuilder(const BuilderData &builder_data);
    /**
     * Get the DER-th derivative of the previously assembled K submatrix.
     * @param der the order of derivative (default 0)
     * @return the reference to the assembled matrix
     */
    const Eigen::MatrixXcd &getDoubleLayer(int der = 0) const;
    /**
     * Get the DER-th derivative of the previously assembled W submatrix.
     * @param der the order of derivative (default 0)
     * @return the reference to the assembled matrix
     */
    const Eigen::MatrixXcd &getHypersingular(int der = 0) const;
    /**
     * Get the DER-th derivative of the previously assembled V submatrix.
     * @param der the order of derivative (default 0)
     * @return the reference to the assembled matrix
     */
    const Eigen::MatrixXcd &getSingleLayer(int der = 0) const;
    /**
     * Assemble the K submatrix and optionally its first DER derivatives
     * for the given wavenumber and refraction index.
     * @param k_in wavenumber (complex)
     * @param c_in refraction index (real, at least 1 otherwise an error is thrown)
     * @param der the order of derivative (default 0, derivatives are not computed)
     */
    void assembleDoubleLayer(const std::complex<double> &k_in, double c_in, int der = 0);
    /**
     * Assemble the W submatrix and optionally its first DER derivatives
     * for the given wavenumber and refraction index.
     * @param k_in wavenumber (complex)
     * @param c_in refraction index (real, at least 1 otherwise an error is thrown)
     * @param der the order of derivative (default 0, derivatives are not computed)
     */
    void assembleHypersingular(const std::complex<double> &k_in, double c_in, int der = 0);
    /**
     * Assemble the V submatrix and optionally its first DER derivatives
     * for the given wavenumber and refraction index.
     * @param k_in wavenumber (complex)
     * @param c_in refraction index (real, at least 1 otherwise an error is thrown)
     * @param der the order of derivative (default 0, derivatives are not computed)
     */
    void assembleSingleLayer(const std::complex<double> &k_in, double c_in, int der = 0);
    /**
     * Assemble K, W and V submatrices and optionally their first DER derivatives
     * for the given wavenumber and refraction index. An error will be thrown
     * if the test and trial spaces are not equal.
     * @param k_in wavenumber (complex)
     * @param c_in refraction index (real, at least 1 otherwise an error is thrown)
     * @param der the order of derivative (default 0, derivatives are not computed)
     */
    void assembleAll(const std::complex<double> &k_in, double c_in, int der = 0);
    /**
     * This function initializes sparse assembly for the wavenumber k and
     * the refraction number c_in.
     *
     * @param k_in wavenumber
     * @param c_in refraction inside
     */
    void initializeSparseAssembly(const std::complex<double> &k_in, double c_in);
    /**
     * The following commands retrieve elements of single layer, double layer,
     * and hypersingular layer, by computing only the required interaction matrices.
     * Note that you should build double layer first, followed by single layer,
     * and hypersingular should come last.
     *
     * @param row row index
     * @param col column index
     */
    std::complex<double> getSingleLayerElement(size_t row, size_t col);
    std::complex<double> getHypersingularElement(size_t row, size_t col);
    std::complex<double> getDoubleLayerElement(size_t row, size_t col);
    // get timing info for profiling
    unsigned getInteractionMatrixAssemblyTime();
    unsigned getHankelComputationTime();
    unsigned getPanelInteractionDataTime();
    const BuilderData &getData() const { return data; }
};

#endif // GALERKIN_BUILDERHPP
