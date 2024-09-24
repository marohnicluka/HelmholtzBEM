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

void SolutionsOperator::gen_sol_op_in(GalerkinBuilder &builder, const complex_t &k, double c_o, double c_i, Eigen::MatrixXcd &T) {
    T.resize(dim_test + dim_trial, dim_trial + dim_test);
#ifdef DIEGO_BUILDER
    const auto &msh = builder_data.mesh;
    const auto &trlsp = builder_data.trial_space;
    const auto &tstsp = builder_data.test_space;
    unsigned N = builder_data.getGaussQROrder();
    K_i = double_layer_helmholtz::GalerkinMatrix(msh, trlsp, tstsp, N, k, c_i);
    W_i = hypersingular_helmholtz::GalerkinMatrix(msh, trlsp, N, k, c_i);
    V_i = single_layer_helmholtz::GalerkinMatrix(msh, tstsp, N, k, c_i);
    K_o = double_layer_helmholtz::GalerkinMatrix(msh, trlsp, tstsp, N, k, c_o);
    W_o = hypersingular_helmholtz::GalerkinMatrix(msh, trlsp, N, k, c_o);
    V_o = single_layer_helmholtz::GalerkinMatrix(msh, tstsp, N, k, c_o);
#else
    if (builder_data.testTrialSpacesAreEqual()) {

        builder.initializeSparseAssembly(k, c_i);
        std::function<complex_t(size_t,size_t)> f = [&](size_t row, size_t col) {
            complex_t ret = builder.getDoubleLayerElement(row, col);
            return ret;
        };
        hierarchical::PanelGeometry pg(builder_data.mesh.getPanels());
        hierarchical::BlockTree tree(pg, 1.2);
        std::cout << "Creating h-matrix..." << std::endl;
        auto tic = std::chrono::high_resolution_clock::now();
        hierarchical::Matrix hmat(f, tree, 1e-5);
        auto toc = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count() * 1e-3;
        std::cout << "H-matrix assembly done in " << elapsed << " seconds" << std::endl;
        std::cout << "HMAT FILLED: " << (100. * double(hmat.param_count())) / ((pg.size() - 1) * (pg.size() - 1)) << " %" << std::endl;
        hmat.truncate();
        std::cout << "HMAT FILLED AFTER TRUNCATION: " << (100. * double(hmat.param_count())) / ((pg.size() - 1) * (pg.size() - 1)) << " %" << std::endl;
        auto hmat_dense = hmat.to_dense_matrix();

        std::cout << "Dense assembly..." << std::endl;
        tic = std::chrono::high_resolution_clock::now();
        builder.assembleAll(k, c_i);
        toc = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count() * 1e-3;
        std::cout << "Dense assembly done in " << elapsed << " seconds" << std::endl;
        K_i = builder.getDoubleLayer();
        std::cout << "HMAT ERROR: " << (hmat_dense - K_i).norm() / K_i.norm() << std::endl;
        W_i = builder.getHypersingular();
        V_i = builder.getSingleLayer();
        builder.assembleAll(k, c_o);
        K_o = builder.getDoubleLayer();
        W_o = builder.getHypersingular();
        V_o = builder.getSingleLayer();
    } else {
        builder.assembleDoubleLayer(k, c_i);
        K_i = builder.getDoubleLayer();
        builder.assembleHypersingular(k, c_i);
        W_i = builder.getHypersingular();
        builder.assembleSingleLayer(k, c_i);
        V_i = builder.getSingleLayer();
        builder.assembleDoubleLayer(k, c_o);
        K_o = builder.getDoubleLayer();
        builder.assembleHypersingular(k, c_o);
        W_o = builder.getHypersingular();
        builder.assembleSingleLayer(k, c_o);
        V_o = builder.getSingleLayer();
    }
#endif

    // assemble solutions operator matrix
    auto T11=T.block(0, 0, dim_test, dim_trial),
         T21=T.block(dim_test, 0, dim_trial, dim_trial),
         T12=T.block(0, dim_trial, dim_test, dim_test),
         T22=T.block(dim_test, dim_trial, dim_trial, dim_test);
    T11 = M - K_o + K_i; T21 = W_o - W_i; T12 = V_o - V_i; T22 = (M + K_o - K_i).transpose();
    if (do_projection) // project onto orthogonal FEM spaces
        T = projection(T);
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
    if (builder_data.testTrialSpacesAreEqual()) {
        builder.assembleAll(k, c_i, 1);
        K_i = builder.getDoubleLayer(0);
        W_i = builder.getHypersingular(0);
        V_i = builder.getSingleLayer(0);
        K_der_i = builder.getDoubleLayer(1);
        W_der_i = builder.getHypersingular(1);
        V_der_i = builder.getSingleLayer(1);
        builder.assembleAll(k, c_o, 1);
        K_o = builder.getDoubleLayer(0);
        W_o = builder.getHypersingular(0);
        V_o = builder.getSingleLayer(0);
        K_der_o = builder.getDoubleLayer(1);
        W_der_o = builder.getHypersingular(1);
        V_der_o = builder.getSingleLayer(1);
    } else {
        builder.assembleDoubleLayer(k, c_i, 1);
        K_i = builder.getDoubleLayer(0);
        K_der_i = builder.getDoubleLayer(1);
        builder.assembleHypersingular(k, c_i, 1);
        W_i = builder.getHypersingular(0);
        W_der_i = builder.getHypersingular(1);
        builder.assembleSingleLayer(k, c_i, 1);
        V_i = builder.getSingleLayer(0);
        V_der_i = builder.getSingleLayer(1);
        builder.assembleDoubleLayer(k, c_o, 1);
        K_o = builder.getDoubleLayer(0);
        K_der_o = builder.getDoubleLayer(1);
        builder.assembleHypersingular(k, c_o, 1);
        W_o = builder.getHypersingular(0);
        W_der_o = builder.getHypersingular(1);
        builder.assembleSingleLayer(k, c_o, 1);
        V_o = builder.getSingleLayer(0);
        V_der_o = builder.getSingleLayer(1);
    }
    // assemble solutions operator matrix and its derivative
    T.block(0, 0, dim_test, dim_trial) = M - K_o + K_i;
    T.block(dim_test, 0, dim_trial, dim_trial) = W_o - W_i;
    T.block(0, dim_trial, dim_test, dim_test) = V_o - V_i;
    T.block(dim_test, dim_trial, dim_trial, dim_test) = (M + K_o - K_i).transpose();
    T_der.block(0, 0, dim_test, dim_trial) = -K_der_o + K_der_i;
    T_der.block(dim_test, 0, dim_trial, dim_trial) = W_der_o - W_der_i;
    T_der.block(0, dim_trial, dim_test, dim_test) = V_der_o - V_der_i;
    T_der.block(dim_test, dim_trial, dim_trial, dim_test) = (K_der_o - K_der_i).transpose();
    // project onto orthogonal FEM spaces
    T = projection(T);
    T_der = projection(T_der);
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
    if (builder_data.testTrialSpacesAreEqual()) {
        builder.assembleAll(k, c_i, 2);
        K_i = builder.getDoubleLayer(0);
        W_i = builder.getHypersingular(0);
        V_i = builder.getSingleLayer(0);
        K_der_i = builder.getDoubleLayer(1);
        W_der_i = builder.getHypersingular(1);
        V_der_i = builder.getSingleLayer(1);
        K_der2_i = builder.getDoubleLayer(2);
        W_der2_i = builder.getHypersingular(2);
        V_der2_i = builder.getSingleLayer(2);
        builder.assembleAll(k, c_o, 2);
        K_o = builder.getDoubleLayer(0);
        W_o = builder.getHypersingular(0);
        V_o = builder.getSingleLayer(0);
        K_der_o = builder.getDoubleLayer(1);
        W_der_o = builder.getHypersingular(1);
        V_der_o = builder.getSingleLayer(1);
        K_der2_o = builder.getDoubleLayer(2);
        W_der2_o = builder.getHypersingular(2);
        V_der2_o = builder.getSingleLayer(2);
    } else {
        builder.assembleDoubleLayer(k, c_i, 2);
        K_i = builder.getDoubleLayer(0);
        K_der_i = builder.getDoubleLayer(1);
        K_der2_i = builder.getDoubleLayer(2);
        builder.assembleHypersingular(k, c_i, 2);
        W_i = builder.getHypersingular(0);
        W_der_i = builder.getHypersingular(1);
        W_der2_i = builder.getHypersingular(2);
        builder.assembleSingleLayer(k, c_i, 2);
        V_i = builder.getSingleLayer(0);
        V_der_i = builder.getSingleLayer(1);
        V_der2_i = builder.getSingleLayer(2);
        builder.assembleDoubleLayer(k, c_o, 2);
        K_o = builder.getDoubleLayer(0);
        K_der_o = builder.getDoubleLayer(1);
        K_der2_o = builder.getDoubleLayer(2);
        builder.assembleHypersingular(k, c_o, 2);
        W_o = builder.getHypersingular(0);
        W_der_o = builder.getHypersingular(1);
        W_der2_o = builder.getHypersingular(2);
        builder.assembleSingleLayer(k, c_o, 2);
        V_o = builder.getSingleLayer(0);
        V_der_o = builder.getSingleLayer(1);
        V_der2_o = builder.getSingleLayer(2);
    }
#if 0
    auto der_i = double_layer_helmholtz_der2::GalerkinMatrix(builder.getData().mesh, builder.getData().trial_space, builder.getData().test_space, 10, k, c_i);
    std::cout << "DBL: " << (K_der2_i - der_i).norm() << std::endl;
    der_i = single_layer_helmholtz_der2::GalerkinMatrix(builder.getData().mesh, builder.getData().trial_space, 10, k, c_i);
    std::cout << "SNG: " << (V_der2_i - der_i).norm() << std::endl;
    der_i = hypersingular_helmholtz_der2::GalerkinMatrix(builder.getData().mesh, builder.getData().test_space, 10, k, c_i);
    std::cout << "HYP: " << (W_der2_i - der_i).norm() << std::endl;
#endif
    // assemble solutions operator matrix and its derivatives
    T.block(0, 0, dim_test, dim_trial) = M - K_o + K_i;
    T.block(dim_test, 0, dim_trial, dim_trial) = W_o - W_i;
    T.block(0, dim_trial, dim_test, dim_test) = V_o - V_i;
    T.block(dim_test, dim_trial, dim_trial, dim_test) = (M + K_o - K_i).transpose();
    T_der.block(0, 0, dim_test, dim_trial) = -K_der_o + K_der_i;
    T_der.block(dim_test, 0, dim_trial, dim_trial) = W_der_o - W_der_i;
    T_der.block(0, dim_trial, dim_test, dim_test) = V_der_o - V_der_i;
    T_der.block(dim_test, dim_trial, dim_trial, dim_test) = (K_der_o - K_der_i).transpose();
    T_der2.block(0, 0, dim_test, dim_trial) = -K_der2_o + K_der2_i;
    T_der2.block(dim_test, 0, dim_trial, dim_trial) = W_der2_o - W_der2_i;
    T_der2.block(0, dim_trial, dim_test, dim_test) = V_der2_o - V_der2_i;
    T_der2.block(dim_test, dim_trial, dim_trial, dim_test) = (K_der2_o - K_der2_i).transpose();
    // project onto orthogonal FEM spaces
    T = projection(T);
    T_der = projection(T_der);
    T_der2 = projection(T_der2);
}
