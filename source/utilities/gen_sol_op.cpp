#include "gen_sol_op.hpp"
#include "mass_matrix.hpp"
#include "hmatrix.hpp"
#include <iostream>
#include <chrono>

//#define DIEGO_BUILDER 1
#ifdef DIEGO_BUILDER
#include "double_layer.hpp"
#include "double_layer_der.hpp"
#include "double_layer_der2.hpp"
#include "hypersingular.hpp"
#include "hypersingular_der.hpp"
#include "hypersingular_der2.hpp"
#include "single_layer.hpp"
#include "single_layer_der.hpp"
#include "single_layer_der2.hpp"
#include <fstream>
#endif

typedef std::complex<double> complex_t;

SolutionsOperator::SolutionsOperator(const BuilderData &builder_data_in, bool proj)
: builder_data(builder_data_in), do_projection(proj)
{
    dim_test = builder_data.getTestSpaceDimension();
    dim_trial = builder_data.getTrialSpaceDimension();
    size_t dim = dim_test + dim_trial;
    // assemble mass matrix
    if (builder_data.getPanelsAreLines()) {
        auto smallGaussQR = getGaussQR(2 * std::max(builder_data.getQtest(), builder_data.getQtrial()), 0., 1.);
        M = mass_matrix::GalerkinMatrix(builder_data.mesh, builder_data.trial_space, builder_data.test_space, smallGaussQR);
    } else M = mass_matrix::GalerkinMatrix(builder_data.mesh, builder_data.trial_space, builder_data.test_space, builder_data.GaussQR);
    if (do_projection) { // compute matrix for projection onto ortogonal FEM-spaces
        Eigen::MatrixXd A;
        A.setZero(dim, dim);
        A.block(0, 0, dim_test, dim_trial) = M;
        A.block(dim_test, dim_trial, dim_trial, dim_test) = M.transpose().eval();
        Eigen::LDLT<Eigen::MatrixXd> llt(A);
        lu = Eigen::PartialPivLU<Eigen::MatrixXcd>
            (llt.transpositionsP().transpose() * llt.matrixL().toDenseMatrix() * llt.vectorD().cwiseSqrt().asDiagonal());
    }
}

Eigen::MatrixXcd SolutionsOperator::projection(const Eigen::MatrixXcd &T) const {
    assert(do_projection);
    return lu.solve(lu.solve(T).transpose().eval()).transpose().eval();
}

void SolutionsOperator::gen_sol_op(const complex_t &k, double c_o, double c_i, Eigen::MatrixXcd &T) {
    GalerkinBuilder builder(builder_data);
    gen_sol_op_in(builder, k, c_o, c_i, T);
}

void SolutionsOperator::gen_sol_op(GalerkinBuilder &builder, const complex_t &k, double c_o, double c_i, Eigen::MatrixXcd &T) {
    gen_sol_op_in(builder, k, c_o, c_i, T);
}

void SolutionsOperator::assemble_operator(Eigen::MatrixXcd &T, bool mass,
                                          const Eigen::MatrixXcd &K_i, const Eigen::MatrixXcd &W_i, const Eigen::MatrixXcd &V_i,
                                          const Eigen::MatrixXcd &K_o, const Eigen::MatrixXcd &W_o, const Eigen::MatrixXcd &V_o) const {
    T.resize(dim_test + dim_trial, dim_trial + dim_test);
    auto T11=T.block(0, 0, dim_test, dim_trial),
         T21=T.block(dim_test, 0, dim_trial, dim_trial),
         T12=T.block(0, dim_trial, dim_test, dim_test),
         T22=T.block(dim_test, dim_trial, dim_trial, dim_test);
    T11 = -K_o + K_i; T21 = W_o - W_i; T12 = V_o - V_i; T22 = (K_o - K_i);
    if (mass) {
        T11 += M;
        T22 += M.transpose();
    }
    if (do_projection) // project onto orthogonal FEM spaces
        T = projection(T);
}

void SolutionsOperator::gen_sol_op_in(GalerkinBuilder &builder, const complex_t &k, double c_o, double c_i, Eigen::MatrixXcd &T) {
#ifdef DIEGO_BUILDER
    const auto &msh = builder_data.mesh;
    const auto &trlsp = builder_data.trial_space;
    const auto &tstsp = builder_data.test_space;
    unsigned N = builder_data.getGaussQROrder();
    auto K_i_diego = double_layer_helmholtz::GalerkinMatrix(msh, trlsp, tstsp, N, k, c_i);
    auto W_i_diego = hypersingular_helmholtz::GalerkinMatrix(msh, trlsp, N, k, c_i);
    auto V_i_diego = single_layer_helmholtz::GalerkinMatrix(msh, tstsp, N, k, c_i);
    auto K_o_diego = double_layer_helmholtz::GalerkinMatrix(msh, trlsp, tstsp, N, k, c_o);
    auto W_o_diego = hypersingular_helmholtz::GalerkinMatrix(msh, trlsp, N, k, c_o);
    auto V_o_diego = single_layer_helmholtz::GalerkinMatrix(msh, tstsp, N, k, c_o);
#endif
    auto tic = std::chrono::high_resolution_clock::now();
    hierarchical::PanelGeometry pg(builder_data.mesh.getPanels());
    hierarchical::BlockTree tree(pg, 0.8);
    auto toc = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count() * 1e-3;
    std::cout << "Block tree created in " << elapsed << " seconds" << std::endl;
    tic = std::chrono::high_resolution_clock::now();
    builder.assembleDense(k, c_i);
    toc = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count() * 1e-3;
    std::cout << "Dense assembly done in " << elapsed << " seconds" << std::endl;
    K_i = builder.getDoubleLayer();
    W_i = builder.getHypersingular();
    V_i = builder.getSingleLayer();
    assert(!(K_i.array() != K_i.array()).any());
    assert(!(W_i.array() != W_i.array()).any());
    assert(!(V_i.array() != V_i.array()).any());
#ifdef DIEGO_BUILDER
    std::cout << "Test 1: " << (K_i - K_i_diego).norm() / K_i_diego.norm() << std::endl;
    std::cout << "Test 2: " << (W_i - W_i_diego).norm() / W_i_diego.norm() << std::endl;
    std::cout << "Test 3: " << (V_i - V_i_diego).norm() / V_i_diego.norm() << std::endl;
#endif
    builder.assembleDense(k, c_o);
    K_o = builder.getDoubleLayer();
    W_o = builder.getHypersingular();
    V_o = builder.getSingleLayer();
    // assemble solutions operator matrix
    assemble_operator(T, true, K_i, W_i, V_i, K_o, W_o, V_o);
}

void SolutionsOperator::gen_sol_op_1st_der(const complex_t &k, double c_o, double c_i,
                                           Eigen::MatrixXcd &T, Eigen::MatrixXcd &T_der) {
    GalerkinBuilder builder(builder_data);
    gen_sol_op_1st_der_in(builder, k, c_o, c_i, T, T_der);
}

void SolutionsOperator::gen_sol_op_1st_der(GalerkinBuilder &builder, const complex_t &k, double c_o, double c_i,
                                           Eigen::MatrixXcd &T, Eigen::MatrixXcd &T_der) {
    gen_sol_op_1st_der_in(builder, k, c_o, c_i, T, T_der);
}

void SolutionsOperator::gen_sol_op_1st_der_in(GalerkinBuilder &builder, const complex_t &k, double c_o, double c_i,
                                              Eigen::MatrixXcd &T, Eigen::MatrixXcd &T_der) {
    T.resize(dim_test + dim_trial, dim_trial + dim_test);
    T_der.resize(dim_test + dim_trial, dim_trial + dim_test);
    Eigen::MatrixXcd K_der_i, K_der_o, V_der_i, V_der_o, W_der_i, W_der_o;
    builder.assembleDense(k, c_i, LayerType::ALL, 1);
    K_i = builder.getDoubleLayer(0);
    W_i = builder.getHypersingular(0);
    V_i = builder.getSingleLayer(0);
    K_der_i = builder.getDoubleLayer(1);
    W_der_i = builder.getHypersingular(1);
    V_der_i = builder.getSingleLayer(1);
    builder.assembleDense(k, c_o, LayerType::ALL, 1);
    K_o = builder.getDoubleLayer(0);
    W_o = builder.getHypersingular(0);
    V_o = builder.getSingleLayer(0);
    K_der_o = builder.getDoubleLayer(1);
    W_der_o = builder.getHypersingular(1);
    V_der_o = builder.getSingleLayer(1);
    // assemble solutions operator matrix and its derivative
    assemble_operator(T, true, K_i, W_i, V_i, K_o, W_o, V_o);
    assemble_operator(T, false, K_der_i, W_der_i, V_der_i, K_der_o, W_der_o, V_der_o);
}

void SolutionsOperator::gen_sol_op_2nd_der(const complex_t &k, double c_o, double c_i,
                                           Eigen::MatrixXcd &T, Eigen::MatrixXcd &T_der, Eigen::MatrixXcd &T_der2) {
    GalerkinBuilder builder(builder_data);
    gen_sol_op_2nd_der_in(builder, k, c_o, c_i, T, T_der, T_der2);
}

void SolutionsOperator::gen_sol_op_2nd_der(GalerkinBuilder &builder, const complex_t &k, double c_o, double c_i,
                                           Eigen::MatrixXcd &T, Eigen::MatrixXcd &T_der, Eigen::MatrixXcd &T_der2) {
    gen_sol_op_2nd_der_in(builder, k, c_o, c_i, T, T_der, T_der2);
}

void SolutionsOperator::gen_sol_op_2nd_der_in(GalerkinBuilder &builder, const complex_t &k, double c_o, double c_i,
                                              Eigen::MatrixXcd &T, Eigen::MatrixXcd &T_der, Eigen::MatrixXcd &T_der2) {
    T.resize(dim_test + dim_trial, dim_trial + dim_test);
    T_der.resize(dim_test + dim_trial, dim_trial + dim_test);
    T_der2.resize(dim_test + dim_trial, dim_trial + dim_test);
    Eigen::MatrixXcd K_der_i, K_der_o, V_der_i, V_der_o, W_der_i, W_der_o;
    Eigen::MatrixXcd K_der2_i, K_der2_o, V_der2_i, V_der2_o, W_der2_i, W_der2_o;
    builder.assembleDense(k, c_i, LayerType::ALL, 2);
    K_i = builder.getDoubleLayer(0);
    W_i = builder.getHypersingular(0);
    V_i = builder.getSingleLayer(0);
    K_der_i = builder.getDoubleLayer(1);
    W_der_i = builder.getHypersingular(1);
    V_der_i = builder.getSingleLayer(1);
    K_der2_i = builder.getDoubleLayer(2);
    W_der2_i = builder.getHypersingular(2);
    V_der2_i = builder.getSingleLayer(2);
    builder.assembleDense(k, c_o, LayerType::ALL, 2);
    K_o = builder.getDoubleLayer(0);
    W_o = builder.getHypersingular(0);
    V_o = builder.getSingleLayer(0);
    K_der_o = builder.getDoubleLayer(1);
    W_der_o = builder.getHypersingular(1);
    V_der_o = builder.getSingleLayer(1);
    K_der2_o = builder.getDoubleLayer(2);
    W_der2_o = builder.getHypersingular(2);
    V_der2_o = builder.getSingleLayer(2);
    // assemble solutions operator matrix and its derivatives
    assemble_operator(T, true, K_i, W_i, V_i, K_o, W_o, V_o);
    assemble_operator(T, false, K_der_i, W_der_i, V_der_i, K_der_o, W_der_o, V_der_o);
    assemble_operator(T, false, K_der2_i, W_der2_i, V_der2_i, K_der2_o, W_der2_o, V_der2_o);
}
