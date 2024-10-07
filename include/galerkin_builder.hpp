/**
 * @file galerkin_builder.hpp
 * @brief This file declares the object to evaluate the entries of
 *        Galerkin matrices by using common and composite
 *        Gauss-Legendre quadrature rules.
 *
 * This File is a part of the HelmholtzTransmissionProblemBEM library.
 * (c) 2023 Luka Marohnić
 */

#ifndef GALERKIN_BUILDERHPP
#define GALERKIN_BUILDERHPP

#include <unordered_set>
#include <Eigen/Sparse>
#include "abstract_bem_space.hpp"
#include "abstract_parametrized_curve.hpp"
#include "gauleg.hpp"
#include "cbessel.hpp"

#define PARALLELIZE_BUILDER 1
#define HANKEL_VECTORIZE 1

class BuilderData {
    size_t npanels, Qtest, Qtrial, dim_test, dim_trial;
    bool panels_are_lines;
public:
    // trial and test spaces
    const ParametrizedMesh &mesh;
    const AbstractBEMSpace &test_space;
    const AbstractBEMSpace &trial_space;
    // quadrature rules
    QuadRule GaussQR;
    QuadRule CGaussQR;
    /**
     * @brief Construct the Galerkin builder data object from the given parameters.
     *
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
     * @brief Return the dimension of the test space.
     *
     * @return unsigned int, the dimension
     */
    size_t getTestSpaceDimension(size_t n_panels = 0) const { return n_panels == 0 ? dim_test : test_space.getSpaceDim(n_panels); }
    /**
     * @brief Return the dimension of the trial space.
     *
     * @return unsigned int, the dimension
     */
    size_t getTrialSpaceDimension(size_t n_panels = 0) const { return n_panels == 0 ? dim_trial : trial_space.getSpaceDim(n_panels); }
    /**
     * @brief Test whether the test and trial spaces are the same.
     *
     * @return boolean true if spaces are equal, false otherwise
     */
    bool testTrialSpacesAreEqual() const { return &test_space == &trial_space; }
    /**
     * @brief Return TRUE iff i-th and j-th panels are adjacent.
     *
     * @param i index of the first panel
     * @param j index of the second panel
     * @param swap whether panels need to be swapped
     * @return boolean true if panels are adjacent, false otherwise
     */
    inline bool is_adjacent(size_t i, size_t j, bool &swap) const noexcept;
    // data
    Eigen::ArrayXXd  m_tc, m_ta, m_tg, m_sc, m_sa, m_sg, m_wc, m_wa, m_wg,
                     m_dlp_cnc_fg, m_dlp_adj_fg, m_dlp_adj_fg_swap, m_dlp_gen_fg,
                     m_dlp_cnc_fg_t, m_dlp_adj_fg_t,
                     m_dlp_adj_fg_swap_t, m_dlp_gen_fg_t,
                     m_slp_cnc_fg, m_slp_adj_fg, m_slp_adj_fg_swap, m_slp_gen_fg,
                     m_hyp_cnc_fg, m_hyp_adj_fg, m_hyp_adj_fg_swap, m_hyp_gen_fg,
                     m_hyp_cnc_fg_arc, m_hyp_adj_fg_arc,
                     m_hyp_adj_fg_arc_swap, m_hyp_gen_fg_arc,
                     Derivative_01_sc_n, Derivative_01_tc_n, Derivative_01_sa_n,
                     Derivative_01_ta_n, Derivative_01_sg_n, Derivative_01_tg_n,
                     Derivative_01_swapped_sa_n, Derivative_01_swapped_ta_n;
    Eigen::ArrayXXcd Derivative_01_sc, Derivative_01_tc, Derivative_01_sa,
                     Derivative_01_ta, Derivative_01_sg, Derivative_01_tg,
                     Derivative_01_swapped_sa, Derivative_01_swapped_ta,
                     op_sc, op_tc, op_sa, op_ta, op_sg, op_tg, swapped_op_sa, swapped_op_ta;
    // methods
    const AbstractParametrizedCurve &panel(size_t i) const { return *mesh.getPanels()[i]; }
    size_t getQtest() const { return Qtest; }
    size_t getQtrial() const { return Qtrial; }
    size_t getGaussQROrder() const { return GaussQR.n; }
    size_t getCGaussQROrder() const { return CGaussQR.n; }
    size_t getNumPanels() const { return npanels; }
    bool getPanelsAreLines() const { return panels_are_lines; }
};

enum class LayerType : uint8_t {
    NONE = 0,
    DOUBLE = 1,
    SINGLE = 2,
    HYPERSINGULAR = 4,
    ALL = 7
};
inline LayerType operator|(LayerType lhs, LayerType rhs) {
    return static_cast<LayerType>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}
inline LayerType operator~(LayerType layer) {
    return static_cast<LayerType>(~static_cast<uint8_t>(layer) & 0x07);
}
inline uint8_t operator&(LayerType lhs, LayerType rhs) {
    return static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs);
}

class GalerkinBuilder {
    // builder data object
    const BuilderData &data;
    // wavenumber and refraction index
    std::complex<double> k, ksqrtc, k2ch, ksqrtc_inv, ksqrtc_two;
    double c, sqrtc, ksqrtca;
    bool k_rp, panels_are_lines;
    // timing info
    typedef struct Ws { // workspace struct
        size_t i, j;
        Eigen::ArrayXXcd m_h0[2], m_h1[2], m_v[2], m_tangent[2], m_tangent_p[2], m_1[2], m_2[2], m_3[2], cf, h1_vnorm,
                         m_1_hg, m_2_g, m_3_g, m_v_s, m_1_s, m_2_s, m_3_s, m_h0_s, m_h1_s, m_1_hg_s, cf_s, m_2_g_s, m_3_g_s;
        Eigen::ArrayXXd m_v_norm[2], m_v_norm2[2], m_tangent_norm[2], m_tangent_p_norm[2], vdotn, vdotn_t, tdottp, ttp_norm,
                        temp, cfr, m_v_norm_s, m_v_norm2_s, tdottp_s, ttp_norm_s, temp_s, vdotn_s, vdotn_t_s, cfr_s;
        bool tdottp_zero, compute_transposed;
        std::map<LayerType, Eigen::ArrayXXcd> interaction_matrix;
#if defined(HANKEL_VECTORIZE) && defined(PARALLELIZE_BUILDER)
        complex_bessel::Hankel1Real01 H1R01, H1R01_s;
#endif
        Ws() : compute_transposed(false) { }
    } workspace_t;
    // the following structure is used for generating indices of lower triangular matrix
    typedef std::pair<size_t, size_t> IndexPair;
#if defined(HANKEL_VECTORIZE) && !defined(PARALLELIZE_BUILDER)
    complex_bessel::Hankel1Real01 H1R01, H1R01_s;
#endif
    // storage for dense computation
    std::map<LayerType,Eigen::MatrixXcd[3]> dense_layer_matrix;
    // routines for computing shared data
    void compute_coinciding(workspace_t &ws) noexcept;
    void compute_adjacent(bool swap, workspace_t &ws) noexcept;
    void compute_general(workspace_t &ws) noexcept;
    void compute_tdottp(size_t K, workspace_t &ws) const noexcept;
    void compute_tdottp_s(workspace_t &ws) const noexcept;
    // routines for interaction matrix assembly
    void initialize_constants(const std::complex<double> &k_in, double c_in); // mandatory init function
    void double_layer_coinciding(size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept;
    void double_layer_adjacent(bool swap, size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept;
    void double_layer_general(size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept;
    void hypersingular_coinciding(size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept;
    void hypersingular_adjacent(bool swap, size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept;
    void hypersingular_general(size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept;
    void single_layer_coinciding(size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept;
    void single_layer_adjacent(bool swap, size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept;
    void single_layer_general(size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept;
    void panel_interaction(size_t d, workspace_t &ws, LayerType layers, bool comp_hyp_sl_ncoinc = true) noexcept;
    static void H1_01_cplx(const Eigen::ArrayXXcd &x, Eigen::ArrayXXcd &h_0, Eigen::ArrayXXcd &h_1);
    /**
     * @brief Test whether array AR is practically zero (the absolute
     * value of its largest coefficient is smaller than machine epsilon).
     *
     * @param ar dynamic sized array of type @p T
     * @return @p true iff the largest element is smaller than machine epsilon
     * */
    template <typename T>
    static bool isArrayZero(const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> &ar) noexcept;
    /**
     * @brief Find the coordinates of element @p kl of an interaction matrix
     * pertinent to the panel pair @p ij in the global operator matrix.
     *
     * @param ij panel pair indices
     * @param kl interaction matrix element indices (local coordinates)
     * @param layer layer specification
     * @return global coordinates of the element @p kl
     */
    IndexPair map_local_to_global(const IndexPair &ij, const IndexPair &kl, LayerType layer) const noexcept;
    /**
     * @brief Helper function which fetches the (@p d th derivative of) the
     * operator (sub)matrix corresponding to @p layer. This function performs
     * necessary checks and will throw an exception if @f$d > 2@f$ or if the
     * layer is not found.
     *
     * @param layer layer specification
     * @param d order of the derivative
     * @return the (@p d th derivative of) operator (sub)matrix for @p layer
     */
    const Eigen::MatrixXcd &get_dense_layer_matrix_safe(LayerType layer, size_t d) const;
public:
    /**
     * @brief Construct the Galerkin matrix builder from the given parameters.
     *
     * @param builder_data the Galerkin builder data object
     */
    GalerkinBuilder(const BuilderData &builder_data);
    /**
     * @brief Return builder data.
     */
    const BuilderData &getData() const { return data; }
    /**
     * @brief Assemble the layers for given @p k and @p c_i in
     * dense form. If rows and cols are not zero, then only the
     * respective submatrix is assembled. The result is stored
     * internally and can be fetched by @ref getDoubleLayer,
     * @ref getHypersingular and @ref getSingleLayer functions.
     *
     * @param k_in wavenumber
     * @param c_in refraction index inside
     * @param layers layers to compute
     * @param d the order of derivative (default 0)
     * @param row row index of the top-left corner
     * @param col column index of the top-left corner
     * @param rows number of rows in the submatrix
     * @param cols number of cols in the submatrix
     */
    void assembleDense(const std::complex<double> &k_in, double c_in, LayerType layers = LayerType::ALL, size_t d = 0,
                       size_t row = 0, size_t col = 0, size_t rows = 0, size_t cols = 0);
    /**
     * @brief Sparse assembly of a submatrix for given
     * parameters @p k and @p c_i.
     * Only @p i th row and @p j th column will be assembled.
     * The result is stored in a sparse matrix. Only
     * elements with row in @p ai or col in @p aj are read
     * (note that these lists will be updated with @p i and @p j).
     * The last argument is a map serving as a container
     * for computed coefficients of input matrix. Keys
     * in this map are layers for which coefficients are
     * to be computed. Sparse matrices must be resized
     * properly before calling this function.
     * Workspace must be provided on a caller side.
     *
     * @param k_in wavenumber
     * @param c_in refraction index inside
     * @param layers map of sparse matrices with layers as keys
     * @param row top-left corner row of the submatrix
     * @param col top-left corner column of the submatrix
     * @param rows number of rows in the submatrix
     * @param cols number of columns in the submatrix
     * @param ai list of available row indices
     * @param aj list of available column indices
     * @param i row to read
     * @param j column to read
     * @param ws workspace
     */
    void assembleSparse(const std::complex<double> &k_in, double c_in,
                        std::map<LayerType, Eigen::SparseMatrix<std::complex<double> > > &layers,
                        size_t row, size_t col, size_t rows, size_t cols,
                        std::vector<size_t> &ai, std::vector<size_t> &aj,
                        size_t i, size_t j,
                        workspace_t &ws);
    /**
     * @brief Get the @p d th derivative of the assembled K (sub)matrix.
     *
     * @param d the order of derivative (default 0)
     * @return the reference to the assembled matrix
     */
    const Eigen::MatrixXcd &getDoubleLayer(size_t d = 0) const;
    /**
     * @brief Get the @p d th derivative of the assembled W (sub)matrix.
     *
     * @param d the order of derivative (default 0)
     * @return the reference to the assembled matrix
     */
    const Eigen::MatrixXcd &getHypersingular(size_t d = 0) const;
    /**
     * @brief Get the @p d th derivative of the assembled V (sub)matrix.
     *
     * @param d the order of derivative (default 0)
     * @return the reference to the assembled matrix
     */
    const Eigen::MatrixXcd &getSingleLayer(size_t d = 0) const;
};

#endif // GALERKIN_BUILDERHPP
