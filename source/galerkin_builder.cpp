#include "galerkin_builder.hpp"
#include <numeric>
#include <iostream>
#include <chrono>

static const double eps_mach = std::numeric_limits<double>::epsilon();

BuilderData::BuilderData(const ParametrizedMesh &mesh_in,
                         const AbstractBEMSpace &test_space_in,
                         const AbstractBEMSpace &trial_space_in,
                         unsigned order)
: mesh(mesh_in), test_space(test_space_in), trial_space(trial_space_in)
{
    _panels_are_lines = mesh_in.isPolygonal();
    GaussQR = getGaussQR(order, 0., 1.);
    CGaussQR = getCGaussQR(order);
    panels = mesh.getPanels();
    numpanels = mesh.getNumPanels();
    dim_test = test_space.getSpaceDim(numpanels);
    dim_trial = trial_space.getSpaceDim(numpanels);
    Qtest = test_space.getQ();
    Qtrial = trial_space.getQ();
    size_t N = CGaussQR.n, Ns = GaussQR.n;
    m_tc.resize(N, N);
    m_ta.resize(N, N);
    m_tg.resize(Ns, Ns);
    m_sc.resize(N, N);
    m_sa.resize(N, N);
    m_sg.resize(Ns, Ns);
    m_wc.resize(N, N);
    m_wa.resize(N, N);
    m_wg.resize(Ns, Ns);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double t = CGaussQR.x(j), s = t * (1. - CGaussQR.x(i)), w = t * CGaussQR.w(i) * CGaussQR.w(j);
            m_tc(i, j) = t; m_sc(i, j) = s; m_wc(i, j) = w;
            t = CGaussQR.x(i), s = t * CGaussQR.x(j), w = t * CGaussQR.w(i) * CGaussQR.w(j);
            m_ta(i, j) = t; m_sa(i, j) = s; m_wa(i, j) = w;
            if (i < Ns && j < Ns) {
                t = GaussQR.x(j), s = GaussQR.x(i), w = GaussQR.w(i) * GaussQR.w(j);
                m_tg(i, j) = t; m_sg(i, j) = s; m_wg(i, j) = w;
            }
        }
    }
    m_dlp_cnc_fg.resize(Qtest * N, Qtrial * N);
    m_dlp_cnc_fg_t.resize(Qtest * N, Qtrial * N);
    m_dlp_adj_fg.resize(Qtest * N, Qtrial * N);
    m_dlp_adj_fg_swap.resize(Qtest * N, Qtrial * N);
    m_dlp_adj_fg_t.resize(Qtest * N, Qtrial * N);
    m_dlp_adj_fg_swap_t.resize(Qtest * N, Qtrial * N);
    m_dlp_gen_fg.resize(Qtest * Ns, Qtrial * Ns);
    m_dlp_gen_fg_t.resize(Qtest * Ns, Qtrial * Ns);
    m_hyp_cnc_fg.resize(Qtrial * N, Qtrial * N);
    m_hyp_cnc_fg_arc.resize(Qtrial * N, Qtrial * N);
    m_hyp_adj_fg.resize(Qtrial * N, Qtrial * N);
    m_hyp_adj_fg_swap.resize(Qtrial * N, Qtrial * N);
    m_hyp_adj_fg_arc.resize(Qtrial * N, Qtrial * N);
    m_hyp_adj_fg_arc_swap.resize(Qtrial * N, Qtrial * N);
    m_hyp_gen_fg.resize(Qtrial * Ns, Qtrial * Ns);
    m_hyp_gen_fg_arc.resize(Qtrial * Ns, Qtrial * Ns);
    m_slp_cnc_fg.resize(Qtest * N, Qtest * N);
    m_slp_adj_fg.resize(Qtest * N, Qtest * N);
    m_slp_adj_fg_swap.resize(Qtest * N, Qtest * N);
    m_slp_gen_fg.resize(Qtest * Ns, Qtest * Ns);
    for (size_t j = 0; j < Qtrial; ++j) {
        for (size_t i = 0; i < Qtest; ++i) {
            auto dbl_g_fg = m_dlp_gen_fg.block(i * Ns, j * Ns, Ns, Ns);
            auto dbl_g_fg_t = m_dlp_gen_fg_t.block(i * Ns, j * Ns, Ns, Ns);
            for (size_t jj = 0; jj < N; ++jj) {
                for (size_t ii = 0; ii < N; ++ii) {
                    double sc = m_sc(ii, jj), sa = m_sa(ii, jj), tc = m_tc(ii, jj), ta = m_ta(ii, jj);
                    m_dlp_cnc_fg(i * N + ii, j * N + jj) =
                        test_space.evaluateShapeFunction(i, sc) * trial_space.evaluateShapeFunction(j, tc);
                    m_dlp_cnc_fg_t(i * N + ii, j * N + jj) =
                        test_space.evaluateShapeFunction(i, tc) * trial_space.evaluateShapeFunction(j, sc);
                    m_dlp_adj_fg(i * N + ii, j * N + jj) =
                        test_space.evaluateShapeFunction_01_swapped(i, sa) * trial_space.evaluateShapeFunction(j, ta);
                    m_dlp_adj_fg_swap(i * N + ii, j * N + jj) =
                        test_space.evaluateShapeFunction(i, sa) * trial_space.evaluateShapeFunction_01_swapped(j, ta);
                    m_dlp_adj_fg_t(i * N + ii, j * N + jj) =
                        test_space.evaluateShapeFunction_01_swapped(i, ta) * trial_space.evaluateShapeFunction(j, sa);
                    m_dlp_adj_fg_swap_t(i * N + ii, j * N + jj) =
                        test_space.evaluateShapeFunction(i, ta) * trial_space.evaluateShapeFunction_01_swapped(j, sa);
                    if (ii < Ns && jj < Ns) {
                        double s = m_sg(ii, jj), t = m_tg(ii, jj);
                        dbl_g_fg(ii, jj) =
                            test_space.evaluateShapeFunction(i, s) * trial_space.evaluateShapeFunction(j, t);
                        dbl_g_fg_t(ii, jj) =
                            test_space.evaluateShapeFunction(i, t) * trial_space.evaluateShapeFunction(j, s);
                    }
                }
            }
            m_dlp_cnc_fg.block(i * N, j * N, N, N) *= m_wc;
            m_dlp_cnc_fg_t.block(i * N, j * N, N, N) *= m_wc;
            m_dlp_adj_fg.block(i * N, j * N, N, N) *= m_wa;
            m_dlp_adj_fg_t.block(i * N, j * N, N, N) *= m_wa;
            m_dlp_adj_fg_swap.block(i * N, j * N, N, N) *= m_wa;
            m_dlp_adj_fg_swap_t.block(i * N, j * N, N, N) *= m_wa;
            dbl_g_fg *= m_wg;
            dbl_g_fg_t *= m_wg;
        }
        for (size_t i = 0; i < Qtrial; ++i) {
            auto hyp_c_fg = m_hyp_cnc_fg.block(i * N, j * N, N, N);
            auto hyp_c_fg_arc = m_hyp_cnc_fg_arc.block(i * N, j * N, N, N);
            auto hyp_a_fg = m_hyp_adj_fg.block(i * N, j * N, N, N);
            auto hyp_a_fg_swap = m_hyp_adj_fg_swap.block(i * N, j * N, N, N);
            auto hyp_a_fg_arc = m_hyp_adj_fg_arc.block(i * N, j * N, N, N);
            auto hyp_a_fg_arc_swap = m_hyp_adj_fg_arc_swap.block(i * N, j * N, N, N);
            auto hyp_g_fg = m_hyp_gen_fg.block(i * Ns, j * Ns, Ns, Ns);
            auto hyp_g_fg_arc = m_hyp_gen_fg_arc.block(i * Ns, j * Ns, Ns, Ns);
            for (size_t jj = 0; jj < N; ++jj) {
                for (size_t ii = 0; ii < N; ++ii) {
                    double sc = m_sc(ii, jj), sa = m_sa(ii, jj), tc = m_tc(ii, jj), ta = m_ta(ii, jj);
                    hyp_c_fg(ii, jj) =
                        trial_space.evaluateShapeFunction(i, sc) * trial_space.evaluateShapeFunction(j, tc);
                    hyp_c_fg_arc(ii, jj) =
                        trial_space.evaluateShapeFunctionDot_01(i, sc) * trial_space.evaluateShapeFunctionDot_01(j, tc);
                    hyp_a_fg(ii, jj) =
                        trial_space.evaluateShapeFunction_01_swapped(i, sa) * trial_space.evaluateShapeFunction(j, ta);
                    hyp_a_fg_swap(ii, jj) =
                        trial_space.evaluateShapeFunction(i, sa) * trial_space.evaluateShapeFunction_01_swapped(j, ta);
                    hyp_a_fg_arc(ii, jj) =
                        trial_space.evaluateShapeFunctionDot_01_swapped(i, sa) * trial_space.evaluateShapeFunctionDot_01(j, ta);
                    hyp_a_fg_arc_swap(ii, jj) =
                        trial_space.evaluateShapeFunctionDot_01(i, sa) * trial_space.evaluateShapeFunctionDot_01_swapped(j, ta);
                    if (ii < Ns && jj < Ns) {
                        double s = m_sg(ii, jj), t = m_tg(ii, jj);
                        hyp_g_fg(ii, jj) =
                            trial_space.evaluateShapeFunction(i, s) * trial_space.evaluateShapeFunction(j, t);
                        hyp_g_fg_arc(ii, jj) =
                            trial_space.evaluateShapeFunctionDot_01(i, s) * trial_space.evaluateShapeFunctionDot_01(j, t);
                    }
                }
            }
            hyp_c_fg *= m_wc;
            hyp_c_fg_arc *= m_wc;
            hyp_a_fg *= m_wa;
            hyp_a_fg_swap *= m_wa;
            hyp_a_fg_arc *= m_wa;
            hyp_a_fg_arc_swap *= m_wa;
            hyp_g_fg_arc *= m_wg;
            hyp_g_fg *= m_wg;
        }
    }
    for (size_t i = 0; i < Qtest; ++i) {
        for (size_t j = 0; j < Qtest; ++j) {
            auto sng_c_fg = m_slp_cnc_fg.block(i * N, j * N, N, N);
            auto sng_a_fg = m_slp_adj_fg.block(i * N, j * N, N, N);
            auto sng_a_fg_swap = m_slp_adj_fg_swap.block(i * N, j * N, N, N);
            auto sng_g_fg = m_slp_gen_fg.block(i * Ns, j * Ns, Ns, Ns);
            for (size_t jj = 0; jj < N; ++jj) {
                for (size_t ii = 0; ii < N; ++ii) {
                    double sc = m_sc(ii, jj), sa = m_sa(ii, jj), tc = m_tc(ii, jj), ta = m_ta(ii, jj);
                    sng_c_fg(ii, jj) =
                        test_space.evaluateShapeFunction(i, sc) * test_space.evaluateShapeFunction(j, tc);
                    sng_a_fg(ii, jj) =
                        test_space.evaluateShapeFunction_01_swapped(i, sa) * test_space.evaluateShapeFunction(j, ta);
                    sng_a_fg_swap(ii, jj) =
                        test_space.evaluateShapeFunction(i, sa) * test_space.evaluateShapeFunction_01_swapped(j, ta);
                    if (ii < Ns && jj < Ns) {
                        double s = m_sg(ii, jj), t = m_tg(ii, jj);
                        sng_g_fg(ii, jj) =
                            test_space.evaluateShapeFunction(i, s) * test_space.evaluateShapeFunction(j, t);
                    }
                }
            }
            sng_c_fg *= m_wc;
            sng_a_fg *= m_wa;
            sng_a_fg_swap *= m_wa;
            sng_g_fg *= m_wg;
        }
    }
    size_t Nsd = _panels_are_lines ? 1 : Ns;
    Derivative_01_sg.resize(Nsd, Nsd * numpanels);
    Derivative_01_tg.resize(Nsd, Nsd * numpanels);
    Derivative_01_sg_n.resize(Nsd, Nsd * numpanels);
    Derivative_01_tg_n.resize(Nsd, Nsd * numpanels);
    op_sg.resize(Ns, Ns * numpanels);
    op_tg.resize(Ns, Ns * numpanels);
    Eigen::ArrayXXcd tmps(Nsd, Nsd);
    Eigen::ArrayXXd tmps_n(Nsd, Nsd);
    for (size_t i = 0; i < numpanels; ++i) {
        const auto &p = *panels[i];
        p.Derivative_01(m_sg, tmps, tmps_n);
        Derivative_01_sg.block(0, Nsd * i, Nsd, Nsd) = tmps;
        Derivative_01_sg_n.block(0, Nsd * i, Nsd, Nsd) = tmps_n;
        p.Derivative_01(m_tg, tmps, tmps_n);
        Derivative_01_tg.block(0, Nsd * i, Nsd, Nsd) = tmps;
        Derivative_01_tg_n.block(0, Nsd * i, Nsd, Nsd) = tmps_n;
        op_sg.block(0, i * Ns, Ns, Ns) = p[m_sg];
        op_tg.block(0, i * Ns, Ns, Ns) = p[m_tg];
    }
}

// return true iff the panels P1 and P2 are adjacent, compute the SWAP boolean alongside
bool BuilderData::is_adjacent(size_t i, size_t j, bool &swap) const {
    const auto &p1 = *panels[i], &p2 = *panels[j];
    double eps = eps_mach * 100.;
    if ((p1(1) - p2(-1)).norm() < eps) {
        swap = false;
        return true;
    }
    if ((p1(-1) - p2(1)).norm() < eps) {
        swap = true;
        return true;
    }
    return false;
}

GalerkinBuilder::GalerkinBuilder(const BuilderData &builder_data) : data(builder_data) {
    linp = data.getPanelsAreLines();
    size_t N = data.getCGaussQROrder(), Ns = data.getGaussQROrder();
    size_t Qtest = data.getQtest(), Qtrial = data.getQtrial();
#ifdef HANKEL_VECTORIZE
    H1R01.initialize(N, N);
    H1R01_s.initialize(Ns, Ns);
#endif
    m_zero.setZero(N, N);
    m_zero_s.setZero(Ns, Ns);
    for (size_t K = 0; K < 2; ++K) {
        m_h0[K].resize(N, N);
        m_h1[K].resize(N, N);
        m_v[K].resize(N, N);
        m_tangent[K].resize(linp ? 1: N, linp ? 1 : N);
        m_tangent_p[K].resize(linp ? 1 : N, linp ? 1 : N);
        m_1[K].resize(N, N);
        m_2[K].resize(N, N);
        m_3[K].resize(N, N);
        m_v_norm[K].resize(N, N);
        m_v_norm2[K].resize(N, N);
        m_tangent_norm[K].resize(linp ? 1 : N, linp ? 1 : N);
        m_tangent_p_norm[K].resize(linp ? 1 : N, linp ? 1 : N);
    }
    tdottp.resize(N, N);
    ttp_norm.resize(N, N);
    tdottp_s.resize(Ns, Ns);
    ttp_norm_s.resize(Ns, Ns);
    vdotn.resize(N, N);
    vdotn_s.resize(Ns, Ns);
    m_h0_s.resize(Ns, Ns);
    m_h1_s.resize(Ns, Ns);
    m_v_s.resize(Ns, Ns);
    m_1_s.resize(Ns, Ns);
    m_2_s.resize(Ns, Ns);
    m_3_s.resize(Ns, Ns);
    m_v_norm_s.resize(Ns, Ns);
    m_v_norm2_s.resize(Ns, Ns);
    double_layer_interaction_matrix.resize(Qtest, Qtrial);
    double_layer_der_interaction_matrix.resize(Qtest, Qtrial);
    double_layer_der2_interaction_matrix.resize(Qtest, Qtrial);
    hypersingular_interaction_matrix.resize(Qtrial, Qtrial);
    hypersingular_der_interaction_matrix.resize(Qtrial, Qtrial);
    hypersingular_der2_interaction_matrix.resize(Qtrial, Qtrial);
    single_layer_interaction_matrix.resize(Qtest, Qtest);
    single_layer_der_interaction_matrix.resize(Qtest, Qtest);
    single_layer_der2_interaction_matrix.resize(Qtest, Qtest);
    interaction_matrix_assembly_time = hankel_computation_time = panel_interaction_data_time = 0;
}

void GalerkinBuilder::iH1_01_cplx(const Eigen::ArrayXXcd& x, Eigen::ArrayXXcd& h_0, Eigen::ArrayXXcd& h_1) {
    size_t nr = x.rows(), nc = x.cols(), i, j;
    for (i = 0; i < nr; ++i) for (j = 0; j < nc; ++j) {
        const auto &z = x(i, j);
        h_0(i, j) = complex_bessel::HankelH1(0., z);
        h_1(i, j) = complex_bessel::HankelH1(1., z);
    }
    h_0 *= 1i;
    h_1 *= 1i;
    auto zmat = Eigen::ArrayXXcd::Zero(x.rows(),x.cols());
    h_0 = (h_0.real() != h_0.real()).select(zmat,h_0);
    h_1 = (h_1.real() != h_1.real()).select(zmat,h_1);
}

// compute values required for the coinciding panels case
inline void GalerkinBuilder::compute_coinciding(size_t i) throw() {
    const auto &p = *data.mesh.getPanels()[i];
    auto tic = chrono::high_resolution_clock::now();
    p.Derivative_01(data.m_tc, m_tangent_p[0], m_tangent_p_norm[0]);
    p.Derivative_01(data.m_sc, m_tangent[0], m_tangent_norm[0]);
    m_v[0] = p[data.m_sc] - p[data.m_tc];
    m_v_norm2[0] = m_v[0].cwiseAbs2();
    m_v_norm[0] = m_v_norm2[0].cwiseSqrt();
    auto toc = chrono::high_resolution_clock::now();
    panel_interaction_data_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
    tic = chrono::high_resolution_clock::now();
    if (k_real_positive)
#ifdef HANKEL_VECTORIZE
        H1R01.ih1_01(ksqrtc.real() * m_v_norm[0], m_h0[0], m_h1[0]);
#else
        for (size_t j = 0; j < N; ++j) for (size_t i = 0; i < N; ++i)
            complex_bessel::iH1_01(ksqrtc.real() * m_v_norm[0](i, j), m_h0[0](i, j), m_h1[0](i, j));
#endif
    else
        iH1_01_cplx(ksqrtc * m_v_norm[0], m_h0[0], m_h1[0]);
    toc = chrono::high_resolution_clock::now();
    hankel_computation_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
}

// compute values required for the adjacent panels case
inline void GalerkinBuilder::compute_adjacent(size_t i, size_t j, bool swap) throw() {
    size_t K;
    const auto &p = *data.mesh.getPanels()[i], &q = *data.mesh.getPanels()[j];
    auto tic = chrono::high_resolution_clock::now();
    if (linp) {
        p.Derivative_01(data.m_sa, m_tangent[0], m_tangent_norm[0]);
        q.Derivative_01(data.m_ta, m_tangent_p[0], m_tangent_p_norm[0]);
    } else if (swap) {
        q.Derivative_01_swapped(data.m_ta, m_tangent_p[0], m_tangent_p_norm[0], true);
        q.Derivative_01_swapped(data.m_sa, m_tangent_p[1], m_tangent_p_norm[1], true);
        p.Derivative_01(data.m_sa, m_tangent[0], m_tangent_norm[0]);
        p.Derivative_01(data.m_ta, m_tangent[1], m_tangent_norm[1]);
    } else {
        q.Derivative_01(data.m_ta, m_tangent_p[0], m_tangent_p_norm[0]);
        q.Derivative_01(data.m_sa, m_tangent_p[1], m_tangent_p_norm[1]);
        p.Derivative_01_swapped(data.m_sa, m_tangent[0], m_tangent_norm[0], true);
        p.Derivative_01_swapped(data.m_ta, m_tangent[1], m_tangent_norm[1], true);
    }
    m_v[0] = swap ? p[data.m_sa] - q.swapped_op(data.m_ta) : p.swapped_op(data.m_sa) - q[data.m_ta];
    m_v[1] = swap ? p[data.m_ta] - q.swapped_op(data.m_sa) : p.swapped_op(data.m_ta) - q[data.m_sa];
    m_v_norm2[0] = m_v[0].cwiseAbs2();
    m_v_norm2[1] = m_v[1].cwiseAbs2();
    m_v_norm[0] = m_v_norm2[0].cwiseSqrt();
    m_v_norm[1] = m_v_norm2[1].cwiseSqrt();
    auto toc = chrono::high_resolution_clock::now();
    panel_interaction_data_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
    tic = chrono::high_resolution_clock::now();
    for (K = 0; K < 2; ++K) {
        if (k_real_positive)
#ifdef HANKEL_VECTORIZE
            H1R01.ih1_01(ksqrtc.real() * m_v_norm[K], m_h0[K], m_h1[K]);
#else
            for (size_t j = 0; j < N; ++j) for (size_t i = 0; i < N; ++i)
                complex_bessel::iH1_01(ksqrtc.real() * m_v_norm[K](i, j), m_h0[K](i, j), m_h1[K](i, j));
#endif
        else
            iH1_01_cplx(ksqrtc * m_v_norm[K], m_h0[K], m_h1[K]);
    }
    toc = chrono::high_resolution_clock::now();
    hankel_computation_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
}

// compute values required for the disjoint panels case
inline void GalerkinBuilder::compute_general(size_t i, size_t j) throw() {
    size_t N = data.getGaussQROrder(), Ns = linp ? 1 : N;
    auto tic = chrono::high_resolution_clock::now();
    m_tangent_p_s = data.Derivative_01_tg.block(0, Ns * j, Ns, Ns);
    m_tangent_p_norm_s = data.Derivative_01_tg_n.block(0, Ns * j, Ns, Ns);
    m_tangent_s = data.Derivative_01_sg.block(0, Ns * i, Ns, Ns);
    m_tangent_norm_s = data.Derivative_01_sg_n.block(0, Ns * i, Ns, Ns);
    m_v_s = data.op_sg.block(0, N * i, N, N) - data.op_tg.block(0, N * j, N, N);
    m_v_norm2_s = m_v_s.cwiseAbs2();
    m_v_norm_s = m_v_norm2_s.cwiseSqrt();
    auto toc = chrono::high_resolution_clock::now();
    panel_interaction_data_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
    tic = chrono::high_resolution_clock::now();
    if (k_real_positive)
#ifdef HANKEL_VECTORIZE
        H1R01_s.ih1_01(ksqrtc.real() * m_v_norm_s, m_h0_s, m_h1_s);
#else
        for (size_t j = 0; j < N; ++j) for (size_t i = 0; i < N; ++i)
            complex_bessel::iH1_01(ksqrtc.real() * m_v_norm_s(i, j), m_h0_s(i, j), m_h1_s(i, j));
#endif
    else
        iH1_01_cplx(ksqrtc * m_v_norm_s, m_h0_s, m_h1_s);
    toc = chrono::high_resolution_clock::now();
    hankel_computation_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
}

// compute interaction matrices for the K submatrix, coinciding panels case
void GalerkinBuilder::double_layer_coinciding(int der) throw() {
    size_t i, j, K, N = data.getCGaussQROrder(), Qtest = data.getQtest(), Qtrial = data.getQtrial();
    const auto &v = m_v[0];
    const auto &v_norm = m_v_norm[0];
    const auto &h1 = m_h1[0], &h0 = m_h0[0];
    double_layer_interaction_matrix.setZero();
    if (k_real_positive)
        m_1[0] = (ksqrtc.real() / v_norm) * h1;
    else m_1[0] = ksqrtc * h1 / v_norm;
    if (der > 0)
        double_layer_der_interaction_matrix.setZero();
    if (der > 1) {
        double_layer_der2_interaction_matrix.setZero();
        if (k_real_positive)
            m_3[0] = h0 - h1 * (ksqrtc.real() * v_norm);
        else m_3[0] = h0 - h1 * ksqrtc * v_norm;
    }
    for (K = 0; K < 2; ++K) {
        const auto &tangent = (K == 0 ? m_tangent_p : m_tangent)[0];
        const auto &tangent_norm = (K == 0 ? m_tangent_norm : m_tangent_p_norm)[0];
        if (linp)
            vdotn = (K * 2 - 1.) * tangent_norm(0, 0) * (v.imag() * tangent(0, 0).real() - v.real() * tangent(0, 0).imag());
        else vdotn = (K * 2 - 1.) * tangent_norm * (v.imag() * tangent.real() - v.real() * tangent.imag());
        if (vdotn.isZero())
            continue;
        for (j = 0; j < Qtrial; ++j) {
            for (i = 0; i < Qtest; ++i) {
                cf = vdotn * (K == 0 ? data.m_dlp_cnc_fg : data.m_dlp_cnc_fg_t).block(i * N, j * N, N, N);
                double_layer_interaction_matrix(i, j) += (cf * m_1[0]).sum();
                if (!der) continue;
                double_layer_der_interaction_matrix(i, j) += (cf * h0).sum();
                if (der > 1)
                    double_layer_der2_interaction_matrix(i, j) += (cf * m_3[0]).sum();
            }
        }
    }
    if (!der) return;
    double_layer_der_interaction_matrix *= k * c;
    if (der > 1)
        double_layer_der2_interaction_matrix *= c;
}

// compute interaction matrices for the K submatrix, adjacent panels case
void GalerkinBuilder::double_layer_adjacent(bool swap, int der, bool transp) throw() {
    size_t i, j, K, N = data.getCGaussQROrder(), Qtest = data.getQtest(), Qtrial = data.getQtrial();
    double_layer_interaction_matrix.setZero();
    if (der > 0)
        double_layer_der_interaction_matrix.setZero();
    if (der > 1)
        double_layer_der2_interaction_matrix.setZero();
    for (K = 0; K < 2; ++K) {
        const auto &tangent = (transp ? m_tangent : m_tangent_p)[linp ? 0 : K];
        const auto &tangent_norm = (transp ? m_tangent_p_norm : m_tangent_norm)[linp ? 0 : K];
        const auto &v = m_v[K];
        const auto &v_norm = m_v_norm[K];
        const auto &h1 = m_h1[K], &h0 = m_h0[K];
        if (!transp) {
            if (k_real_positive)
                m_1[K] = (ksqrtc.real() / v_norm) * h1;
            else m_1[K] = ksqrtc * h1 / v_norm;
            if (der > 1) {
                if (k_real_positive)
                    m_3[K] = h0 - (ksqrtc.real() * v_norm) * h1;
                else m_3[K] = h0 - ksqrtc * v_norm * h1;
            }
        }
        if (linp)
            vdotn = (transp? 1.0 : -1.0) * (v.imag() * tangent(0, 0).real() - v.real() * tangent(0, 0).imag()) * tangent_norm(0, 0);
        else vdotn = (transp? 1.0 : -1.0) * (v.imag() * tangent.real() - v.real() * tangent.imag()) * tangent_norm;
        if (vdotn.isZero())
            continue;
        const auto &fg = swap ? ((transp ? K == 0 : K == 1) ? data.m_dlp_adj_fg_swap_t
                                                            : data.m_dlp_adj_fg_swap)
                              : ((transp ? K == 0 : K == 1) ? data.m_dlp_adj_fg_t
                                                            : data.m_dlp_adj_fg);
        for (j = 0; j < Qtrial; ++j) {
            for (i = 0; i < Qtest; ++i) {
                cf = vdotn * fg.block(i * N, j * N, N, N);
                double_layer_interaction_matrix(i, j) += (cf * m_1[K]).sum();
                if (!der) continue;
                double_layer_der_interaction_matrix(i, j) += (cf * h0).sum();
                if (der > 1)
                    double_layer_der2_interaction_matrix(i, j) += (cf * m_3[K]).sum();
            }
        }
    }
    if (!der) return;
    double_layer_der_interaction_matrix *= k * c;
    if (der > 1)
        double_layer_der2_interaction_matrix *= c;
}

// compute interaction matrices for the K submatrix, disjoint panels case
void GalerkinBuilder::double_layer_general(int der, bool transp) throw() {
    size_t i, j, N = data.getGaussQROrder(), Qtest = data.getQtest(), Qtrial = data.getQtrial();
    const auto &fg = transp ? data.m_dlp_gen_fg_t : data.m_dlp_gen_fg;
    bool zr = false;
    if (!transp) {
        if (linp) {
            vdotn_s = m_v_s.real() * m_tangent_p_s(0, 0).imag() - m_v_s.imag() * m_tangent_p_s(0, 0).real();
            if (vdotn_s.isZero())
                zr = true;
            else vdotn_s *= m_tangent_norm_s(0, 0);
        } else {
            vdotn_s = m_v_s.real() * m_tangent_p_s.imag() - m_v_s.imag() * m_tangent_p_s.real();
            if (vdotn_s.isZero())
                zr = true;
            else vdotn_s *= m_tangent_norm_s;
        }
        if (!zr) {
            if (k_real_positive)
                m_1_s = m_h1_s * (ksqrtc.real() / m_v_norm_s);
            else m_1_s = m_h1_s * ksqrtc / m_v_norm_s;
            if (der > 1)
                m_3_s = m_h0_s - m_1_s * m_v_norm2_s;
        }
    } else {
        if (linp) {
            vdotn_s = m_v_s.imag() * m_tangent_s(0, 0).real() - m_v_s.real() * m_tangent_s(0, 0).imag();
            if (vdotn_s.isZero())
                zr = true;
            else vdotn_s *= m_tangent_p_norm_s(0, 0);
        } else {
            vdotn_s = m_v_s.imag() * m_tangent_s.real() - m_v_s.real() * m_tangent_s.imag();
            if (vdotn_s.isZero())
                zr = true;
            else vdotn_s *= m_tangent_p_norm_s;
        }
    }
    if (zr) {
        double_layer_interaction_matrix.setZero();
        if (der > 0)
            double_layer_der_interaction_matrix.setZero();
        if (der > 1)
            double_layer_der2_interaction_matrix.setZero();
        return;
    }
    if (Qtrial == 2 && Qtest == 2 && der == 0) {
        m_1_hg_s = m_1_s * vdotn_s;
        double_layer_interaction_matrix << (fg.block(0, 0, N, N) * m_1_hg_s).sum(), (fg.block(0, N, N, N) * m_1_hg_s).sum(),
                                           (fg.block(N, 0, N, N) * m_1_hg_s).sum(), (fg.block(N, N, N, N) * m_1_hg_s).sum();
        return;
    }
    for (j = 0; j < Qtrial; ++j) {
        for (i = 0; i < Qtest; ++i) {
            cf_s = vdotn_s * fg.block(i * N, j * N, N, N);
            double_layer_interaction_matrix(i, j) = (cf_s * m_1_s).sum();
            if (!der) continue;
            double_layer_der_interaction_matrix(i, j) = (cf_s * m_h0_s).sum();
            if (der > 1)
                double_layer_der2_interaction_matrix(i, j) = (cf_s * m_3_s).sum();
        }
    }
    if (!der) return;
    double_layer_der_interaction_matrix *= k * c;
    if (der > 1)
        double_layer_der2_interaction_matrix *= c;
}

// compute interaction matrices for the W submatrix, coinciding panels case
void GalerkinBuilder::hypersingular_coinciding(int der) throw() {
    size_t i, j, K, N = data.getCGaussQROrder(), Qtrial = data.getQtrial();
    const auto &h0 = m_h0[0];
    const auto &v_norm = m_v_norm[0], &v_norm2 = m_v_norm2[0];
    compute_tdottp(0);
    Eigen::ArrayXXcd h1_vnorm;
    hypersingular_interaction_matrix.setZero();
    if (der > 0) {
        hypersingular_der_interaction_matrix.setZero();
        h1_vnorm = m_h1[0] * v_norm;
    }
    if (der > 1)
        hypersingular_der2_interaction_matrix.setZero();
    for (K = 0; K < 2; ++K) {
        for (j = 0; j < Qtrial; ++j) {
            for (i = 0; i < Qtrial; ++i) {
                if (i < j) { // symmetry
                    hypersingular_interaction_matrix(i, j) = hypersingular_interaction_matrix(j, i);
                    if (der > 0)
                        hypersingular_der_interaction_matrix(i, j) = hypersingular_der_interaction_matrix(j, i);
                    if (der > 1)
                        hypersingular_der2_interaction_matrix(i, j) = hypersingular_der2_interaction_matrix(j, i);
                    continue;
                }
                cf = data.m_hyp_cnc_fg_arc.block((K > 0 ? j : i) * N, (K > 0 ? i : j) * N, N, N);
                if (!tdottp_zero) {
                    temp = data.m_hyp_cnc_fg.block((K > 0 ? j : i) * N, (K > 0 ? i : j) * N, N, N) * tdottp;
                    if (k_real_positive)
                        cf -= kkc.real() * temp;
                    else cf -= kkc * temp;
                }
                hypersingular_interaction_matrix(i, j) += (cf * h0).sum();
                if (!der) continue;
                if (tdottp_zero)
                    hypersingular_der_interaction_matrix(i, j) += (h1_vnorm * cf).sum();
                else hypersingular_der_interaction_matrix(i, j) += (h1_vnorm * cf + ksqrtc * h0 * temp).sum();
                if (der > 1) {
                    if (tdottp_zero)
                        hypersingular_der2_interaction_matrix(i, j) += ((h0 * v_norm2 - h1_vnorm / ksqrtc) * cf).sum();
                    else hypersingular_der2_interaction_matrix(i, j) += (
                            (h0 * v_norm2 - h1_vnorm / ksqrtc) * cf - temp * (2.0 * ksqrtc * h1_vnorm - h0)).sum();
                }
            }
        }
    }
    if (!der) return;
    hypersingular_der_interaction_matrix *= -sqrtc;
    if (der > 1)
        hypersingular_der2_interaction_matrix *= -c;
}

// compute interaction matrices for the W submatrix, adjacent panels case
void GalerkinBuilder::hypersingular_adjacent(bool swap, int der) throw() {
    size_t i, j, K, N = data.getCGaussQROrder(), Qtrial = data.getQtrial();
    hypersingular_interaction_matrix.setZero();
    if (der > 0)
        hypersingular_der_interaction_matrix.setZero();
    if (der > 1)
        hypersingular_der2_interaction_matrix.setZero();
    Eigen::ArrayXXcd h1_vnorm;
    for (K = 0; K < 2; ++K) {
        const auto &h0 = m_h0[K];
        const auto &v_norm = m_v_norm[K], &v_norm2 = m_v_norm2[K];
        if (!linp || K == 0)
            compute_tdottp(K);
        if (der > 0)
            h1_vnorm = m_h1[K] * v_norm;
        for (j = 0; j < Qtrial; ++j) {
            for (i = 0; i < Qtrial; ++i) {
                cf = swap ? (K > 0 ? data.m_hyp_adj_fg_arc.block(j * N, i * N, N, N)
                                   : data.m_hyp_adj_fg_arc_swap.block(i * N, j * N, N, N))
                          : (K > 0 ? data.m_hyp_adj_fg_arc_swap.block(j * N, i * N, N, N)
                                   : data.m_hyp_adj_fg_arc.block(i * N, j * N, N, N));
                if (!tdottp_zero) {
                    temp = (swap ? (K > 0 ? data.m_hyp_adj_fg.block(j * N, i * N, N, N)
                                        : data.m_hyp_adj_fg_swap.block(i * N, j * N, N, N))
                                 : (K > 0 ? data.m_hyp_adj_fg_swap.block(j * N, i * N, N, N)
                                        : data.m_hyp_adj_fg.block(i * N, j * N, N, N))) * tdottp;
                    if (k_real_positive)
                        cf -= kkc.real() * temp;
                    else cf -= kkc * temp;
                }
                hypersingular_interaction_matrix(i, j) += (cf * h0).sum();
                if (!der) continue;
                if (tdottp_zero)
                    hypersingular_der_interaction_matrix(i, j) += (h1_vnorm * cf).sum();
                else hypersingular_der_interaction_matrix(i, j) += (h1_vnorm * cf + ksqrtc * h0 * temp).sum();
                if (der > 1) {
                    if (tdottp_zero)
                        hypersingular_der2_interaction_matrix(i, j) += ((h0 * v_norm2 - h1_vnorm / ksqrtc) * cf).sum();
                    else hypersingular_der2_interaction_matrix(i, j) += (
                            (h0 * v_norm2 - h1_vnorm / ksqrtc) * cf - temp * (2.0 * ksqrtc * h1_vnorm - h0)).sum();
                }
            }
        }
    }
    if (!der) return;
    hypersingular_der_interaction_matrix *= -sqrtc;
    if (der > 1)
        hypersingular_der2_interaction_matrix *= -c;
}

// compute interaction matrices for the W submatrix, disjoint panels case
void GalerkinBuilder::hypersingular_general(int der) throw() {
    size_t i, j, N = data.getGaussQROrder(), Qtrial = data.getQtrial();
    const auto &h0 = m_h0_s;
    const auto &v_norm = m_v_norm_s, &v_norm2 = m_v_norm2_s;
    compute_tdottp_s();
    Eigen::ArrayXXcd h1_vnorm;
    if (der > 0)
        h1_vnorm = m_h1_s * v_norm;
    for (j = 0; j < Qtrial; ++j) {
        for (i = 0; i < Qtrial; ++i) {
            cf_s = data.m_hyp_gen_fg_arc.block(i * N, j * N, N, N);
            if (!tdottp_zero) {
                temp_s = data.m_hyp_gen_fg.block(i * N, j * N, N, N) * tdottp_s;
                if (k_real_positive)
                    cf_s -= kkc.real() * temp_s;
                else cf_s -= kkc * temp_s;
            }
            hypersingular_interaction_matrix(i, j) = (cf_s * h0).sum();
            if (!der) continue;
            if (tdottp_zero)
                hypersingular_der_interaction_matrix(i, j) = (h1_vnorm * cf_s).sum();
            else hypersingular_der_interaction_matrix(i, j) = (h1_vnorm * cf_s + ksqrtc * h0 * temp_s).sum();
            if (der > 1) {
                if (tdottp_zero)
                    hypersingular_der2_interaction_matrix(i, j) = ((h0 * v_norm2 - h1_vnorm / ksqrtc) * cf_s).sum();
                else hypersingular_der2_interaction_matrix(i, j) = (
                        (h0 * v_norm2 - h1_vnorm / ksqrtc) * cf_s - temp_s * (2.0 * ksqrtc * h1_vnorm - h0)).sum();
            }
        }
    }
    if (!der) return;
    hypersingular_der_interaction_matrix *= -sqrtc;
    if (der > 1)
        hypersingular_der2_interaction_matrix *= -c;
}

// compute interaction matrices for the V submatrix, coinciding panels case
void GalerkinBuilder::single_layer_coinciding(int der) throw() {
    size_t i, j, K, N = data.getCGaussQROrder(), Qtest = data.getQtest();
    const auto &v_norm = m_v_norm[0], &v_norm2 = m_v_norm2[0];
    const auto &h0 = m_h0[0], &h1 = m_h1[0];
    if (linp)
        ttp_norm.setConstant(m_tangent_norm[0](0, 0) * m_tangent_p_norm[0](0, 0));
    else ttp_norm = m_tangent_norm[0] * m_tangent_p_norm[0];
    single_layer_interaction_matrix.setZero();
    if (der > 0) {
        single_layer_der_interaction_matrix.setZero();
        m_2[0] = h1 * v_norm;
    }
    if (der > 1) {
        single_layer_der2_interaction_matrix.setZero();
        m_3[0] = m_2[0] / ksqrtc - h0 * v_norm2;
    }
    for (K = 0; K < 2; ++K) {
        for (j = 0; j < Qtest; ++j) {
            for (i = 0; i < Qtest; ++i) {
                if (i < j) { // symmetry
                    single_layer_interaction_matrix(i, j) = single_layer_interaction_matrix(j, i);
                    if (der > 0)
                        single_layer_der_interaction_matrix(i, j) = single_layer_der_interaction_matrix(j, i);
                    if (der > 1)
                        single_layer_der2_interaction_matrix(i, j) = single_layer_der2_interaction_matrix(j, i);
                    continue;
                }
                cf = ttp_norm * data.m_slp_cnc_fg.block((K > 0 ? j : i) * N, (K > 0 ? i : j) * N, N, N);
                single_layer_interaction_matrix(i, j) += (cf * h0).sum();
                if (!der) continue;
                single_layer_der_interaction_matrix(i, j) += (cf * m_2[0]).sum();
                if (der > 1)
                    single_layer_der2_interaction_matrix(i, j) += (cf * m_3[0]).sum();
            }
        }
    }
    if (!der) return;
    single_layer_der_interaction_matrix *= -sqrtc;
    if (der > 1)
        single_layer_der2_interaction_matrix *= c;
}

// compute interaction matrices for the V submatrix, adjacent panels case
void GalerkinBuilder::single_layer_adjacent(bool swap, int der) throw() {
    size_t i, j, K, N = data.getCGaussQROrder(), Qtest = data.getQtest();
    single_layer_interaction_matrix.setZero();
    if (der > 0)
        single_layer_der_interaction_matrix.setZero();
    if (der > 1)
        single_layer_der2_interaction_matrix.setZero();
    if (linp)
        ttp_norm.setConstant(m_tangent_norm[0](0, 0) * m_tangent_p_norm[0](0, 0));
    for (K = 0; K < 2; ++K) {
        const auto &v_norm = m_v_norm[K], &v_norm2 = m_v_norm2[K];
        const auto &h0 = m_h0[K], &h1 = m_h1[K];
        if (der > 0)
            m_2[K] = h1 * v_norm;
        if (der > 1)
            m_3[K] = m_2[K] / ksqrtc - h0 * v_norm2;
        if (!linp)
            ttp_norm = m_tangent_norm[K] * m_tangent_p_norm[K];
        for (j = 0; j < Qtest; ++j) {
            for (i = 0; i < Qtest; ++i) {
                cf = ttp_norm * (swap ? (K > 0 ? data.m_slp_adj_fg.block(j * N, i * N, N, N)
                                               : data.m_slp_adj_fg_swap.block(i * N, j * N, N, N))
                                      : (K > 0 ? data.m_slp_adj_fg_swap.block(j * N, i * N, N, N)
                                               : data.m_slp_adj_fg.block(i * N, j * N, N, N)));
                single_layer_interaction_matrix(i, j) += (cf * h0).sum();
                if (!der) continue;
                single_layer_der_interaction_matrix(i, j) += (cf * m_2[K]).sum();
                if (der > 1)
                    single_layer_der2_interaction_matrix(i, j) += (cf * m_3[K]).sum();
            }
        }
    }
    if (!der) return;
    single_layer_der_interaction_matrix *= -sqrtc;
    if (der > 1)
        single_layer_der2_interaction_matrix *= c;
}

// compute interaction matrices for the V submatrix, disjoint panels case
void GalerkinBuilder::single_layer_general(int der) throw() {
    size_t i, j, N = data.getGaussQROrder(), Qtest = data.getQtest();
    const auto &v_norm = m_v_norm_s, &v_norm2 = m_v_norm2_s;
    const auto &h0 = m_h0_s, &h1 = m_h1_s;
    if (linp)
        ttp_norm_s.setConstant(m_tangent_norm_s(0, 0) * m_tangent_p_norm_s(0, 0));
    else ttp_norm_s = m_tangent_norm_s * m_tangent_p_norm_s;
    if (der > 0)
        m_2_s = h1 * v_norm;
    if (der > 1)
        m_3_s = m_2_s / ksqrtc - h0 * v_norm2;
    for (j = 0; j < Qtest; ++j) {
        for (i = 0; i < Qtest; ++i) {
            cfr1_s = ttp_norm_s * data.m_slp_gen_fg.block(i * N, j * N, N, N);
            single_layer_interaction_matrix(i, j) = (cfr1_s * h0).sum();
            if (!der) continue;
            single_layer_der_interaction_matrix(i, j) = (cfr1_s * m_2_s).sum();
            if (der > 1)
                single_layer_der2_interaction_matrix(i, j) = (cfr1_s * m_3_s).sum();
        }
    }
    if (!der) return;
    single_layer_der_interaction_matrix *= -sqrtc;
    if (der > 1)
        single_layer_der2_interaction_matrix *= c;
}

void GalerkinBuilder::compute_tdottp(size_t K) {
    double v;
    if (linp) {
        const std::complex<double> &z = m_tangent[K](0, 0), &zp = m_tangent_p[K](0, 0);
        v = z.imag() * zp.imag() + z.real() * zp.real();
        tdottp_zero = std::abs(v) < eps_mach;
    } else {
        tdottp = m_tangent[K].imag() * m_tangent_p[K].imag() + m_tangent[K].real() * m_tangent_p[K].real();
        tdottp_zero = tdottp.isZero();
    }
    if (!tdottp_zero) {
        if (linp)
            tdottp.setConstant(2. * v);
        else tdottp *= 2.;
    } else if (linp)
        tdottp.setZero();
}

void GalerkinBuilder::compute_tdottp_s() {
    double v;
    if (linp) {
        const std::complex<double> &z = m_tangent_s(0, 0), &zp = m_tangent_p_s(0, 0);
        v = z.imag() * zp.imag() + z.real() * zp.real();
        tdottp_zero = std::abs(v) < eps_mach;
    } else {
        tdottp_s = m_tangent_s.imag() * m_tangent_p_s.imag() + m_tangent_s.real() * m_tangent_p_s.real();
        tdottp_zero = tdottp_s.isZero();
    }
    if (!tdottp_zero) {
        if (linp)
            tdottp_s.setConstant(2. * v);
        else tdottp_s *= 2.;
    } else if (linp)
        tdottp_s.setZero();
}

void GalerkinBuilder::all_coinciding(int der) throw() {
    size_t i, j, K, N = data.getCGaussQROrder(), Q = data.getQtest();
    const auto &v = m_v[0];
    const auto &v_norm = m_v_norm[0], &v_norm2 = m_v_norm2[0];
    const auto &h1 = m_h1[0], &h0 = m_h0[0];
    const auto &tangent_p = m_tangent_p[0], &tangent = m_tangent[0];
    compute_tdottp(0);
    if (k_real_positive)
        m_1[0] = (ksqrtc.real() / v_norm) * h1;
    else m_1[0] = ksqrtc * h1 / v_norm;
    m_1[0] = (v_norm<=0).select(m_zero + 1i*m_zero, m_1[0]);
    if (Q == 2 && k_real_positive && !der) {
        if (linp)
            m_1[1] = (v.imag() * tangent(0, 0).real() - v.real() * tangent(0, 0).imag()) * m_1[0] * m_tangent_p_norm[0](0, 0);
        else m_1[1] = (v.imag() * tangent.real() - v.real() * tangent.imag()) * m_1[0] * m_tangent_p_norm[0];
        if (linp)
            m_1[0] *= -(v.imag() * tangent_p(0, 0).real() - v.real() * tangent_p(0, 0).imag()) * m_tangent_norm[0](0, 0);
        else m_1[0] *= -(v.imag() * tangent_p.real() - v.real() * tangent_p.imag()) * m_tangent_norm[0];
        if (!m_1[0].isZero() || !m_1[1].isZero())
            double_layer_interaction_matrix <<
                (data.m_dlp_cnc_fg.block(0, 0, N, N) * m_1[0] + data.m_dlp_cnc_fg_t.block(0, 0, N, N) * m_1[1]).sum(),
                (data.m_dlp_cnc_fg.block(0, 1, N, N) * m_1[0] + data.m_dlp_cnc_fg_t.block(0, 1, N, N) * m_1[1]).sum(),
                (data.m_dlp_cnc_fg.block(1, 0, N, N) * m_1[0] + data.m_dlp_cnc_fg_t.block(1, 0, N, N) * m_1[1]).sum(),
                (data.m_dlp_cnc_fg.block(1, 1, N, N) * m_1[0] + data.m_dlp_cnc_fg_t.block(1, 1, N, N) * m_1[1]).sum();
        else double_layer_interaction_matrix.setZero();
        if (tdottp_zero) {
            hypersingular_interaction_matrix(0, 0) = 2. * (data.m_hyp_cnc_fg_arc.block(0, 0, N, N) * h0).sum();
            hypersingular_interaction_matrix(1, 1) = 2. * (data.m_hyp_cnc_fg_arc.block(N, N, N, N) * h0).sum();
            hypersingular_interaction_matrix(1, 0) = hypersingular_interaction_matrix(0, 1) =
                ((data.m_hyp_cnc_fg_arc.block(N, 0, N, N) + data.m_hyp_cnc_fg_arc.block(0, N, N, N)) * h0).sum();
        } else {
            tdottp *= kkc.real();
            hypersingular_interaction_matrix(0, 0) = 2. * ((data.m_hyp_cnc_fg_arc.block(0, 0, N, N) -
                data.m_hyp_cnc_fg.block(0, 0, N, N) * tdottp) * h0).sum();
            hypersingular_interaction_matrix(1, 1) = 2. * ((data.m_hyp_cnc_fg_arc.block(N, N, N, N) -
                data.m_hyp_cnc_fg.block(N, N, N, N) * tdottp) * h0).sum();
            hypersingular_interaction_matrix(1, 0) = hypersingular_interaction_matrix(0, 1) =
                ((data.m_hyp_cnc_fg_arc.block(N, 0, N, N) + data.m_hyp_cnc_fg_arc.block(0, N, N, N) -
                (data.m_hyp_cnc_fg.block(N, 0, N, N) + data.m_hyp_cnc_fg.block(0, N, N, N)) * tdottp) * h0).sum();
        }
        if (linp)
            m_1_hg = h0 * (m_tangent_norm[0](0, 0) * m_tangent_p_norm[0](0, 0));
        else m_1_hg = h0 * m_tangent_norm[0] * m_tangent_p_norm[0];
        single_layer_interaction_matrix(0, 0) = 2. * (data.m_slp_cnc_fg.block(0, 0, N, N) * m_1_hg).sum();
        single_layer_interaction_matrix(1, 1) = 2. * (data.m_slp_cnc_fg.block(N, N, N, N) * m_1_hg).sum();
        single_layer_interaction_matrix(1, 0) = single_layer_interaction_matrix(0, 1) =
            ((data.m_slp_cnc_fg.block(N, 0, N, N) + data.m_slp_cnc_fg.block(0, N, N, N)) * m_1_hg).sum();
        return;
    }
    if (linp)
        ttp_norm.setConstant(m_tangent_norm[0](0, 0) * m_tangent_p_norm[0](0, 0));
    else ttp_norm = m_tangent_norm[0] * m_tangent_p_norm[0];
    double_layer_interaction_matrix.setZero();
    hypersingular_interaction_matrix.setZero();
    single_layer_interaction_matrix.setZero();
    if (der > 0) {
        double_layer_der_interaction_matrix.setZero();
        hypersingular_der_interaction_matrix.setZero();
        single_layer_der_interaction_matrix.setZero();
        m_2_g = h1 * v_norm;
    }
    if (der > 1) {
        double_layer_der2_interaction_matrix.setZero();
        hypersingular_der2_interaction_matrix.setZero();
        single_layer_der2_interaction_matrix.setZero();
        m_3[0] = h0 - ksqrtc * m_2_g;
        m_3_g = m_2_g / ksqrtc - h0 * v_norm2;
    }
    for (K = 0; K < 2; ++K) {
        if (linp)
            vdotn = (K * 2. - 1.) * (K == 0 ? m_tangent_norm : m_tangent_p_norm)[0](0, 0) *
                (v.imag() * (K == 0 ? tangent_p : tangent)(0, 0).real() - v.real() * (K == 0 ? tangent_p : tangent)(0, 0).imag());
        else vdotn = (K * 2. - 1.) * (K == 0 ? m_tangent_norm : m_tangent_p_norm)[0] *
                (v.imag() * (K == 0 ? tangent_p : tangent).real() - v.real() * (K == 0 ? tangent_p : tangent).imag());
        bool vdotn_zero = vdotn.isZero();
        const auto &fg = K == 0 ? data.m_dlp_cnc_fg : data.m_dlp_cnc_fg_t;
        for (j = 0; j < Q; ++j) {
            for (i = 0; i < Q; ++i) {
                if (!vdotn_zero) {
                    cfr1 = vdotn * fg.block(i * N, j * N, N, N);
                    double_layer_interaction_matrix(i, j) += (cfr1 * m_1[0]).sum();
                }
                if (der > 0 && !vdotn_zero)
                    double_layer_der_interaction_matrix(i, j) += (cfr1 * h0).sum();
                if (der > 1 && !vdotn_zero)
                    double_layer_der2_interaction_matrix(i, j) += (cfr1 * m_3[0]).sum();
                if (i < j) { // symmetry
                    hypersingular_interaction_matrix(i, j) = hypersingular_interaction_matrix(j, i);
                    single_layer_interaction_matrix(i, j) = single_layer_interaction_matrix(j, i);
                    if (der > 0) {
                        hypersingular_der_interaction_matrix(i, j) = hypersingular_der_interaction_matrix(j, i);
                        single_layer_der_interaction_matrix(i, j) = single_layer_der_interaction_matrix(j, i);
                    }
                    if (der > 1) {
                        hypersingular_der2_interaction_matrix(i, j) = hypersingular_der2_interaction_matrix(j, i);
                        single_layer_der2_interaction_matrix(i, j) = single_layer_der2_interaction_matrix(j, i);
                    }
                } else {
                    if (tdottp_zero)
                        cfr1 = data.m_hyp_cnc_fg_arc.block((K > 0 ? j : i) * N, (K > 0 ? i : j) * N, N, N);
                    else {
                        temp = data.m_hyp_cnc_fg.block((K > 0 ? j : i) * N, (K > 0 ? i : j) * N, N, N) * tdottp;
                        if (k_real_positive)
                            cfr1 = data.m_hyp_cnc_fg_arc.block((K > 0 ? j : i) * N, (K > 0 ? i : j) * N, N, N) - kkc.real() * temp;
                        else cf = data.m_hyp_cnc_fg_arc.block((K > 0 ? j : i) * N, (K > 0 ? i : j) * N, N, N) - kkc * temp;
                    }
                    cfr2 = ttp_norm * data.m_slp_cnc_fg.block((K > 0 ? j : i) * N, (K > 0 ? i : j) * N, N, N);
                    single_layer_interaction_matrix(i, j) += (cfr2 * h0).sum();
                    if (k_real_positive || tdottp_zero)
                        hypersingular_interaction_matrix(i, j) += (cfr1 * h0).sum();
                    else hypersingular_interaction_matrix(i, j) += (cf * h0).sum();
                    if (!der) continue;
                    single_layer_der_interaction_matrix(i, j) += (cfr2 * m_2_g).sum();
                    if (tdottp_zero)
                        hypersingular_der_interaction_matrix(i, j) += (m_2_g * cfr1).sum();
                    else if (k_real_positive)
                        hypersingular_der_interaction_matrix(i, j) += (m_2_g * cfr1 + (ksqrtc.real() * temp) * h0).sum();
                    else hypersingular_der_interaction_matrix(i, j) += (m_2_g * cf + ksqrtc * h0 * temp).sum();
                    if (der > 1) {
                        single_layer_der2_interaction_matrix(i, j) += (cfr2 * m_3_g).sum();
                        if (tdottp_zero)
                            hypersingular_der2_interaction_matrix(i, j) -= (cfr1 * m_3_g).sum();
                        else if (k_real_positive)
                            hypersingular_der2_interaction_matrix(i, j) -= (cfr1 * m_3_g + temp * (h0 - 2.0 * m_3[0])).sum();
                        else hypersingular_der2_interaction_matrix(i, j) -= (m_3_g * cf + temp * (h0 - 2.0 * m_3[0])).sum();
                    }
                }
            }
        }
    }
    if (!der) return;
    double_layer_der_interaction_matrix *= k * c;
    hypersingular_der_interaction_matrix *= -sqrtc;
    single_layer_der_interaction_matrix *= -sqrtc;
    if (der > 1) {
        double_layer_der2_interaction_matrix *= c;
        hypersingular_der2_interaction_matrix *= -c;
        single_layer_der2_interaction_matrix *= c;
    }
}

void GalerkinBuilder::all_adjacent(bool swap, int der) throw() {
    size_t K, N = data.getCGaussQROrder(), Q = data.getQtest();
    double_layer_interaction_matrix.setZero();
    hypersingular_interaction_matrix.setZero();
    single_layer_interaction_matrix.setZero();
    if (der > 0) {
        double_layer_der_interaction_matrix.setZero();
        hypersingular_der_interaction_matrix.setZero();
        single_layer_der_interaction_matrix.setZero();
    }
    if (der > 1) {
        double_layer_der2_interaction_matrix.setZero();
        hypersingular_der2_interaction_matrix.setZero();
        single_layer_der2_interaction_matrix.setZero();
    }
    if (linp) {
        compute_tdottp(0);
        ttp_norm.setConstant(m_tangent_norm[0](0, 0) * m_tangent_p_norm[0](0, 0));
    }
    bool vdotn_zero;
    for (K = 0; K < 2; ++K) {
        const auto &tangent_p = m_tangent_p[linp ? 0 : K];
        const auto &tangent_norm = m_tangent_norm[linp ? 0 : K], &tangent_p_norm = m_tangent_p_norm[linp ? 0 : K];
        const auto &v = m_v[K];
        const auto &v_norm = m_v_norm[K], &v_norm2 = m_v_norm2[K];
        const auto &h1 = m_h1[K], &h0 = m_h0[K];
        if (!linp) {
            compute_tdottp(K);
            ttp_norm = tangent_norm * tangent_p_norm;
        }
        if (linp)
            vdotn = (v.real() * tangent_p(0, 0).imag() - v.imag() * tangent_p(0, 0).real()) * tangent_norm(0, 0);
        else vdotn = (v.real() * tangent_p.imag() - v.imag() * tangent_p.real()) * tangent_norm;
        m_1[K] = ksqrtc * h1 / v_norm;
        if (der > 0)
            m_2_g = h1 * v_norm;
        if (der > 1) {
            m_3[K] = h0 - ksqrtc * m_2_g;
            m_3_g = m_2_g / ksqrtc - h0 * v_norm2;
        }
        const auto &fg = swap ? (K == 1 ? data.m_dlp_adj_fg_swap_t : data.m_dlp_adj_fg_swap)
                              : (K == 1 ? data.m_dlp_adj_fg_t : data.m_dlp_adj_fg);
        vdotn_zero = vdotn.isZero();
        for (size_t j = 0; j < Q; ++j) {
            for (size_t i = 0; i < Q; ++i) {
                if (!vdotn_zero)
                    cfr1 = vdotn * fg.block(i * N, j * N, N, N);
                cfr2 = ttp_norm * (swap ? (K > 0 ? data.m_slp_adj_fg.block(j * N, i * N, N, N) : data.m_slp_adj_fg_swap.block(i * N, j * N, N, N))
                                        : (K > 0 ? data.m_slp_adj_fg_swap.block(j * N, i * N, N, N) : data.m_slp_adj_fg.block(i * N, j * N, N, N)));
                if (tdottp_zero) {
                    cfr3 = swap ? (K > 0 ? data.m_hyp_adj_fg_arc.block(j * N, i * N, N, N) : data.m_hyp_adj_fg_arc_swap.block(i * N, j * N, N, N))
                                : (K > 0 ? data.m_hyp_adj_fg_arc_swap.block(j * N, i * N, N, N) : data.m_hyp_adj_fg_arc.block(i * N, j * N, N, N));
                } else {
                    temp = (swap ? (K > 0 ? data.m_hyp_adj_fg.block(j * N, i * N, N, N) : data.m_hyp_adj_fg_swap.block(i * N, j * N, N, N))
                                : (K > 0 ? data.m_hyp_adj_fg_swap.block(j * N, i * N, N, N) : data.m_hyp_adj_fg.block(i * N, j * N, N, N))) * tdottp;
                    if (k_real_positive)
                        cfr3 = (swap ? (K > 0 ? data.m_hyp_adj_fg_arc.block(j * N, i * N, N, N) : data.m_hyp_adj_fg_arc_swap.block(i * N, j * N, N, N))
                                     : (K > 0 ? data.m_hyp_adj_fg_arc_swap.block(j * N, i * N, N, N) : data.m_hyp_adj_fg_arc.block(i * N, j * N, N, N)))
                                - kkc.real() * temp;
                    else cf = (swap ? (K > 0 ? data.m_hyp_adj_fg_arc.block(j * N, i * N, N, N) : data.m_hyp_adj_fg_arc_swap.block(i * N, j * N, N, N))
                                    : (K > 0 ? data.m_hyp_adj_fg_arc_swap.block(j * N, i * N, N, N) : data.m_hyp_adj_fg_arc.block(i * N, j * N, N, N)))
                                - kkc * temp;
                }
                if (!vdotn_zero)
                    double_layer_interaction_matrix(i, j) += (cfr1 * m_1[K]).sum();
                single_layer_interaction_matrix(i, j) += (cfr2 * h0).sum();
                if (k_real_positive || tdottp_zero)
                    hypersingular_interaction_matrix(i, j) += (cfr3 * h0).sum();
                else hypersingular_interaction_matrix(i, j) += (cf * h0).sum();
                if (!der) continue;
                if (!vdotn_zero)
                    double_layer_der_interaction_matrix(i, j) += (cfr1 * h0).sum();
                single_layer_der_interaction_matrix(i, j) += (cfr2 * m_2_g).sum();
                if (tdottp_zero)
                    hypersingular_der_interaction_matrix(i, j) += (m_2_g * cfr3).sum();
                else if (k_real_positive)
                    hypersingular_der_interaction_matrix(i, j) += (m_2_g * cfr3 + (ksqrtc.real() * temp) * h0).sum();
                else hypersingular_der_interaction_matrix(i, j) += (m_2_g * cf + ksqrtc * h0 * temp).sum();
                if (der > 1) {
                    if (!vdotn_zero)
                        double_layer_der2_interaction_matrix(i, j) += (cfr1 * m_3[K]).sum();
                    single_layer_der2_interaction_matrix(i, j) += (cfr2 * m_3_g).sum();
                    if (tdottp_zero)
                        hypersingular_der2_interaction_matrix(i, j) -= (m_3_g * cfr3).sum();
                    else if (k_real_positive)
                        hypersingular_der2_interaction_matrix(i, j) -= (m_3_g * cfr3 + temp * (h0 - 2.0 * m_3[K])).sum();
                    else hypersingular_der2_interaction_matrix(i, j) -= (m_3_g * cf + temp * (h0 - 2.0 * m_3[K])).sum();
                }
            }
        }
    }
    if (!der) return;
    if (!vdotn_zero)
        double_layer_der_interaction_matrix *= k * c;
    hypersingular_der_interaction_matrix *= -sqrtc;
    single_layer_der_interaction_matrix *= -sqrtc;
    if (der > 1) {
        if (!vdotn_zero)
            double_layer_der2_interaction_matrix *= c;
        hypersingular_der2_interaction_matrix *= -c;
        single_layer_der2_interaction_matrix *= c;
    }
}

void GalerkinBuilder::all_general(int der) throw() {
    size_t N = data.getGaussQROrder(), Q = data.getQtest();
    compute_tdottp_s();
    if (k_real_positive)
        m_1_s = (ksqrtc.real() / m_v_norm_s) * m_h1_s;
    else m_1_s = ksqrtc * m_h1_s / m_v_norm_s;
    if (Q == 2 && k_real_positive && der == 0) {
        if (tdottp_zero) {
            hypersingular_interaction_matrix <<
                (data.m_hyp_gen_fg_arc.block(0, 0, N, N) * m_h0_s).sum(), (data.m_hyp_gen_fg_arc.block(0, N, N, N) * m_h0_s).sum(),
                (data.m_hyp_gen_fg_arc.block(N, 0, N, N) * m_h0_s).sum(), (data.m_hyp_gen_fg_arc.block(N, N, N, N) * m_h0_s).sum();
        } else {
            tdottp_s *= kkc.real();
            hypersingular_interaction_matrix <<
                ((data.m_hyp_gen_fg_arc.block(0, 0, N, N) - data.m_hyp_gen_fg.block(0, 0, N, N) * tdottp_s) * m_h0_s).sum(),
                ((data.m_hyp_gen_fg_arc.block(0, N, N, N) - data.m_hyp_gen_fg.block(0, N, N, N) * tdottp_s) * m_h0_s).sum(),
                ((data.m_hyp_gen_fg_arc.block(N, 0, N, N) - data.m_hyp_gen_fg.block(N, 0, N, N) * tdottp_s) * m_h0_s).sum(),
                ((data.m_hyp_gen_fg_arc.block(N, N, N, N) - data.m_hyp_gen_fg.block(N, N, N, N) * tdottp_s) * m_h0_s).sum();
        }
        if (linp)
            m_1_hg_s = m_h0_s * (m_tangent_norm_s(0, 0) * m_tangent_p_norm_s(0, 0));
        else m_1_hg_s = m_h0_s * m_tangent_norm_s * m_tangent_p_norm_s;
        single_layer_interaction_matrix <<
            (data.m_slp_gen_fg.block(0, 0, N, N) * m_1_hg_s).sum(), (data.m_slp_gen_fg.block(0, N, N, N) * m_1_hg_s).sum(),
            (data.m_slp_gen_fg.block(N, 0, N, N) * m_1_hg_s).sum(), (data.m_slp_gen_fg.block(N, N, N, N) * m_1_hg_s).sum();
        if (linp)
            m_1_hg_s = m_v_s.real() * m_tangent_p_s(0, 0).imag() - m_v_s.imag() * m_tangent_p_s(0, 0).real();
        else m_1_hg_s = m_v_s.real() * m_tangent_p_s.imag() - m_v_s.imag() * m_tangent_p_s.real();
        if (m_1_hg_s.isZero())
            double_layer_interaction_matrix.setZero();
        else {
            if (linp)
                m_1_hg_s *= m_1_s * m_tangent_norm_s(0, 0);
            else m_1_hg_s *= m_1_s * m_tangent_norm_s;
            double_layer_interaction_matrix <<
                (data.m_dlp_gen_fg.block(0, 0, N, N) * m_1_hg_s).sum(), (data.m_dlp_gen_fg.block(0, N, N, N) * m_1_hg_s).sum(),
                (data.m_dlp_gen_fg.block(N, 0, N, N) * m_1_hg_s).sum(), (data.m_dlp_gen_fg.block(N, N, N, N) * m_1_hg_s).sum();
        }
        return;
    }
    if (linp) {
        ttp_norm_s.setConstant(m_tangent_norm_s(0, 0) * m_tangent_p_norm_s(0, 0));
        vdotn_s = m_tangent_norm_s(0, 0) * (m_v_s.real() * m_tangent_p_s(0, 0).imag() - m_v_s.imag() * m_tangent_p_s(0, 0).real());
    } else {
        ttp_norm_s = m_tangent_norm_s;
        vdotn_s = ttp_norm_s * (m_v_s.real() * m_tangent_p_s.imag() - m_v_s.imag() * m_tangent_p_s.real());
        ttp_norm_s *= m_tangent_p_norm_s;
    }
    if (der > 0)
        m_2_g_s = m_h1_s * m_v_norm_s;
    if (der > 1) {
        m_3_s = m_h0_s - ksqrtc * m_2_g_s;
        m_3_g_s = m_2_g_s / ksqrtc - m_h0_s * m_v_norm2_s;
    }
    bool vdotn_zero = vdotn.isZero();
    if (vdotn_zero) {
        double_layer_interaction_matrix.setZero();
        if (der > 0)
            double_layer_der_interaction_matrix.setZero();
        if (der > 1)
            double_layer_der2_interaction_matrix.setZero();
    }
    for (size_t j = 0; j < Q; ++j) {
        for (size_t i = 0; i < Q; ++i) {
            if (!vdotn_zero)
                cfr1_s = vdotn_s * data.m_dlp_gen_fg.block(i * N, j * N, N, N);
            cfr2_s = ttp_norm_s * data.m_slp_gen_fg.block(i * N, j * N, N, N);
            if (tdottp_zero)
                cfr3_s = data.m_hyp_gen_fg_arc.block(i * N, j * N, N, N);
            else {
                temp_s = data.m_hyp_gen_fg.block(i * N, j * N, N, N) * tdottp_s;
                if (k_real_positive)
                    cfr3_s = data.m_hyp_gen_fg_arc.block(i * N, j * N, N, N) - kkc.real() * temp_s;
                else cf_s = data.m_hyp_gen_fg_arc.block(i * N, j * N, N, N) - kkc * temp_s;
            }
            if (!vdotn_zero)
                double_layer_interaction_matrix(i, j) = (cfr1_s * m_1_s).sum();
            single_layer_interaction_matrix(i, j) = (cfr2_s * m_h0_s).sum();
            if (k_real_positive || tdottp_zero)
                hypersingular_interaction_matrix(i, j) = (cfr3_s * m_h0_s).sum();
            else hypersingular_interaction_matrix(i, j) = (cf_s * m_h0_s).sum();
            if (!der) continue;
            if (!vdotn_zero)
                double_layer_der_interaction_matrix(i, j) = (cfr1_s * m_h0_s).sum();
            single_layer_der_interaction_matrix(i, j) = (cfr2_s * m_2_g_s).sum();
            if (tdottp_zero)
                hypersingular_der_interaction_matrix(i, j) = (cfr3_s * m_2_g_s).sum();
            else if (k_real_positive)
                hypersingular_der_interaction_matrix(i, j) = (cfr3_s * m_2_g_s + (ksqrtc.real() * temp_s) * m_h0_s).sum();
            else hypersingular_der_interaction_matrix(i, j) = (m_2_g_s * cf_s + ksqrtc * m_h0_s * temp_s).sum();
            if (der > 1) {
                if (!vdotn_zero)
                    double_layer_der2_interaction_matrix(i, j) = (cfr1_s * m_3_s).sum();
                single_layer_der2_interaction_matrix(i, j) = (cfr2_s * m_3_g_s).sum();
                if (tdottp_zero)
                    hypersingular_der2_interaction_matrix(i, j) = -(cfr3_s * m_3_g_s).sum();
                else if (k_real_positive)
                    hypersingular_der2_interaction_matrix(i, j) = -(cfr3_s * m_3_g_s + temp_s * (m_h0_s - 2.0 * m_3_s)).sum();
                else hypersingular_der2_interaction_matrix(i, j) = -(m_3_g_s * cf_s + temp_s * (m_h0_s - 2.0 * m_3_s)).sum();
            }
        }
    }
    if (!der) return;
    if (!vdotn_zero)
        double_layer_der_interaction_matrix *= k * c;
    hypersingular_der_interaction_matrix *= -sqrtc;
    single_layer_der_interaction_matrix *= -sqrtc;
    if (der > 1) {
        if (!vdotn_zero)
            double_layer_der2_interaction_matrix *= c;
        hypersingular_der2_interaction_matrix *= -c;
        single_layer_der2_interaction_matrix *= c;
    }
}

// initialize wavenumber and refraction index with related values
void GalerkinBuilder::initialize_parameters(const std::complex<double>& k_in, double c_in) {
    if (c_in < 1.)
        throw std::runtime_error("Refraction index must not be smaller than 1");
    k = k_in;
    c = c_in;
    sqrtc = sqrt(c);
    ksqrtc = k * sqrtc;
    ksqrtca = std::abs(ksqrtc);
    kkc = ksqrtc * ksqrtc * .5;
    k_real_positive = k.imag() == 0. && k.real() > 0;
}

void GalerkinBuilder::assembleDoubleLayer(const std::complex<double>& k_in, double c_in, int der) {
    initialize_parameters(k_in, c_in);
    size_t dim_test = data.getTestSpaceDimension(), dim_trial = data.getTrialSpaceDimension(), numpanels = data.getNumPanels();
    double_layer_matrix.setZero(dim_test, dim_trial);
    if (der > 0)
        double_layer_der_matrix.setZero(dim_test, dim_trial);
    if (der > 1)
        double_layer_der2_matrix.setZero(dim_test, dim_trial);
    size_t i, j, ii, jj, II, JJ, Qtest = data.getQtest(), Qtrial = data.getQtrial();
    bool swap, adj;
    // Panel oriented assembly
    for (i = 0; i < numpanels; ++i) {
        for (j = 0; j <= i; ++j) {
            auto tic = chrono::high_resolution_clock::now();
            if (i == j) {
                // coinciding panels
                compute_coinciding(i);
                double_layer_coinciding(der);
            } else if ((adj = data.is_adjacent(i, j, swap))) {
                // adjacent panels
                compute_adjacent(i, j, swap);
                double_layer_adjacent(swap, der, false);
            } else {
                // disjoint panels
                compute_general(i, j);
                double_layer_general(der, false);
            }
            auto toc = chrono::high_resolution_clock::now();
            interaction_matrix_assembly_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
            // Local to global mapping of the elements in interaction matrix
            if (!double_layer_interaction_matrix.isZero())
                for (jj = 0; jj < Qtrial; ++jj) {
                    for (ii = 0; ii < Qtest; ++ii) {
                        II = data.test_space.LocGlobMap(ii + 1, i + 1, dim_test) - 1;
                        JJ = data.trial_space.LocGlobMap(jj + 1, j + 1, dim_trial) - 1;
                        double_layer_matrix(II, JJ) += double_layer_interaction_matrix(ii, jj);
                        if (!der) continue;
                        double_layer_der_matrix(II, JJ) += double_layer_der_interaction_matrix(ii, jj);
                        if (der > 1)
                            double_layer_der2_matrix(II, JJ) += double_layer_der2_interaction_matrix(ii, jj);
                    }
                }
            if (i == j)
                continue;
            // reuse data for the (j, i) case
            if (adj)
                double_layer_adjacent(!swap, der, true);
            else
                double_layer_general(der, true);
            interaction_matrix_assembly_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
            if (!double_layer_interaction_matrix.isZero())
                for (jj = 0; jj < Qtrial; ++jj) {
                    for (ii = 0; ii < Qtest; ++ii) {
                        II = data.test_space.LocGlobMap(ii + 1, j + 1, dim_test) - 1;
                        JJ = data.test_space.LocGlobMap(jj + 1, i + 1, dim_trial) - 1;
                        double_layer_matrix(II, JJ) += double_layer_interaction_matrix(ii, jj);
                        if (!der) continue;
                        double_layer_der_matrix(II, JJ) += double_layer_der_interaction_matrix(ii, jj);
                        if (der > 1)
                            double_layer_der2_matrix(II, JJ) += double_layer_der2_interaction_matrix(ii, jj);
                    }
                }
        }
    }
    double_layer_matrix *= 0.25;
    if (der > 0)
        double_layer_der_matrix *= 0.25;
    if (der > 1)
        double_layer_der2_matrix *= 0.25;
}

void GalerkinBuilder::assembleHypersingular(const std::complex<double>& k_in, double c_in, int der) {
    initialize_parameters(k_in, c_in);
    size_t i, j, ii, jj, II, JJ, dim_trial = data.getTrialSpaceDimension(), Qtrial = data.getQtrial(), numpanels = data.getNumPanels();
    hypersingular_matrix.setZero(dim_trial, dim_trial);
    if (der > 0)
        hypersingular_der_matrix.setZero(dim_trial, dim_trial);
    if (der > 1)
        hypersingular_der2_matrix.setZero(dim_trial, dim_trial);
    bool swap;
    // Panel oriented assembly
    for (i = 0; i < numpanels; ++i) {
        for (j = 0; j <= i; ++j) {
            auto tic = chrono::high_resolution_clock::now();
            if (i == j) {
                // coinciding panels
                compute_coinciding(i);
                hypersingular_coinciding(der);
            } else if (data.is_adjacent(i, j, swap)) {
                // adjacent panels
                compute_adjacent(i, j, swap);
                hypersingular_adjacent(swap, der);
            } else {
                // disjoint panels
                compute_general(i, j);
                hypersingular_general(der);
            }
            auto toc = chrono::high_resolution_clock::now();
            interaction_matrix_assembly_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
            // Local to global mapping of the elements in interaction matrix
            for (jj = 0; jj < Qtrial; ++jj) {
                for (ii = 0; ii < Qtrial; ++ii) {
                    II = data.trial_space.LocGlobMap(ii + 1, i + 1, dim_trial) - 1;
                    JJ = data.trial_space.LocGlobMap(jj + 1, j + 1, dim_trial) - 1;
                    hypersingular_matrix(II, JJ) += hypersingular_interaction_matrix(ii, jj);
                    if (!der) continue;
                    hypersingular_der_matrix(II, JJ) += hypersingular_der_interaction_matrix(ii, jj);
                    if (der > 1)
                        hypersingular_der2_matrix(II, JJ) += hypersingular_der2_interaction_matrix(ii, jj);
                }
            }
            if (i == j)
                continue;
            for (jj = 0; jj < Qtrial; ++jj) {
                for (ii = 0; ii < Qtrial; ++ii) {
                    II = data.trial_space.LocGlobMap(ii + 1, j + 1, dim_trial) - 1;
                    JJ = data.trial_space.LocGlobMap(jj + 1, i + 1, dim_trial) - 1;
                    hypersingular_matrix(II, JJ) += hypersingular_interaction_matrix(jj, ii);
                    if (!der) continue;
                    hypersingular_der_matrix(II, JJ) += hypersingular_der_interaction_matrix(jj, ii);
                    if (der > 1)
                        hypersingular_der2_matrix(II, JJ) += hypersingular_der2_interaction_matrix(jj, ii);
                }
            }
        }
    }
    hypersingular_matrix *= 0.25;
    if (der > 0)
        hypersingular_der_matrix *= 0.25;
    if (der > 1)
        hypersingular_der2_matrix *= 0.25;
}

void GalerkinBuilder::assembleSingleLayer(const std::complex<double>& k_in, double c_in, int der) {
    initialize_parameters(k_in, c_in);
    size_t i, j, ii, jj, II, JJ, dim_test = data.getTestSpaceDimension(), Qtest = data.getQtest(), numpanels = data.getNumPanels();
    single_layer_matrix.setZero(dim_test, dim_test);
    if (der > 0)
        single_layer_der_matrix.setZero(dim_test, dim_test);
    if (der > 1)
        single_layer_der2_matrix.setZero(dim_test, dim_test);
    bool swap;
    // Panel oriented assembly
    for (i = 0; i < numpanels; ++i) {
        for (j = 0; j <= i; ++j) {
            auto tic = chrono::high_resolution_clock::now();
            if (i == j) {
                // coinciding panels
                compute_coinciding(i);
                single_layer_coinciding(der);
            } else if (data.is_adjacent(i, j, swap)) {
                // adjacent panels
                compute_adjacent(i, j, swap);
                single_layer_adjacent(swap, der);
            } else {
                // disjoint panels
                compute_general(i, j);
                single_layer_general(der);
            }
            auto toc = chrono::high_resolution_clock::now();
            interaction_matrix_assembly_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
            // Local to global mapping of the elements in interaction matrix
            for (jj = 0; jj < Qtest; ++jj) {
                for (ii = 0; ii < Qtest; ++ii) {
                    II = data.test_space.LocGlobMap(ii + 1, i + 1, dim_test) - 1;
                    JJ = data.test_space.LocGlobMap(jj + 1, j + 1, dim_test) - 1;
                    single_layer_matrix(II, JJ) += single_layer_interaction_matrix(ii, jj);
                    if (!der) continue;
                    single_layer_der_matrix(II, JJ) += single_layer_der_interaction_matrix(ii, jj);
                    if (der > 1)
                        single_layer_der2_matrix(II, JJ) += single_layer_der2_interaction_matrix(ii, jj);
                }
            }
            if (i == j)
                continue;
            for (jj = 0; jj < Qtest; ++jj) {
                for (ii = 0; ii < Qtest; ++ii) {
                    II = data.test_space.LocGlobMap(ii + 1, j + 1, dim_test) - 1;
                    JJ = data.test_space.LocGlobMap(jj + 1, i + 1, dim_test) - 1;
                    single_layer_matrix(II, JJ) += single_layer_interaction_matrix(jj, ii);
                    if (!der) continue;
                    single_layer_der_matrix(II, JJ) += single_layer_der_interaction_matrix(jj, ii);
                    if (der > 1)
                        single_layer_der2_matrix(II, JJ) += single_layer_der2_interaction_matrix(jj, ii);
                }
            }
        }
    }
    single_layer_matrix *= 0.25;
    if (der > 0)
        single_layer_der_matrix *= 0.25;
    if (der > 1)
        single_layer_der2_matrix *= 0.25;
}

void GalerkinBuilder::assembleAll(const std::complex<double>& k_in, double c_in, int der) {
    if (!data.testTrialSpacesAreEqual())
        throw std::runtime_error("Trial and test spaces must be equal");
    initialize_parameters(k_in, c_in);
    size_t i, j, ii, jj, II, JJ, dim = data.getTestSpaceDimension(), Q = data.getQtest(), numpanels = data.getNumPanels();
    single_layer_matrix.setZero(dim, dim);
    double_layer_matrix.setZero(dim, dim);
    hypersingular_matrix.setZero(dim, dim);
    if (der > 0) {
        single_layer_der_matrix.setZero(dim, dim);
        double_layer_der_matrix.setZero(dim, dim);
        hypersingular_der_matrix.setZero(dim, dim);
    }
    if (der > 1) {
        single_layer_der2_matrix.setZero(dim, dim);
        double_layer_der2_matrix.setZero(dim, dim);
        hypersingular_der2_matrix.setZero(dim, dim);
    }
    bool swap, adj, nonnul;
    // Panel oriented assembly
    for (i = 0; i < numpanels; ++i) {
        for (j = 0; j <= i; ++j) {
            if (i == j) {
                // coinciding panels
                compute_coinciding(i);
                auto tic = chrono::high_resolution_clock::now();
                all_coinciding(der);
                auto toc = chrono::high_resolution_clock::now();
                interaction_matrix_assembly_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
            } else if ((adj = data.is_adjacent(i, j, swap))) {
                // adjacent panels
                compute_adjacent(i, j, swap);
                auto tic = chrono::high_resolution_clock::now();
                all_adjacent(swap, der);
                auto toc = chrono::high_resolution_clock::now();
                interaction_matrix_assembly_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
            } else {
                // disjoint panels
                compute_general(i, j);
                auto tic = chrono::high_resolution_clock::now();
                all_general(der);
                auto toc = chrono::high_resolution_clock::now();
                interaction_matrix_assembly_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
            }
            // Local to global mapping of the elements in interaction matrix
            nonnul = !double_layer_interaction_matrix.isZero();
            for (jj = 0; jj < Q; ++jj) {
                for (ii = 0; ii < Q; ++ii) {
                    II = data.test_space.LocGlobMap(ii + 1, i + 1, dim) - 1;
                    JJ = data.test_space.LocGlobMap(jj + 1, j + 1, dim) - 1;
                    if (nonnul)
                        double_layer_matrix(II, JJ) += double_layer_interaction_matrix(ii, jj);
                    hypersingular_matrix(II, JJ) += hypersingular_interaction_matrix(ii, jj);
                    single_layer_matrix(II, JJ) += single_layer_interaction_matrix(ii, jj);
                    if (!der) continue;
                    double_layer_der_matrix(II, JJ) += double_layer_der_interaction_matrix(ii, jj);
                    hypersingular_der_matrix(II, JJ) += hypersingular_der_interaction_matrix(ii, jj);
                    single_layer_der_matrix(II, JJ) += single_layer_der_interaction_matrix(ii, jj);
                    if (der > 1) {
                        double_layer_der2_matrix(II, JJ) += double_layer_der2_interaction_matrix(ii, jj);
                        hypersingular_der2_matrix(II, JJ) += hypersingular_der2_interaction_matrix(ii, jj);
                        single_layer_der2_matrix(II, JJ) += single_layer_der2_interaction_matrix(ii, jj);
                    }
                }
            }
            if (i == j)
                continue;
            // reuse data for the (j, i) case
            auto tic = chrono::high_resolution_clock::now();
            if (adj)
                double_layer_adjacent(!swap, der, true);
            else
                double_layer_general(der, true);
            auto toc = chrono::high_resolution_clock::now();
            interaction_matrix_assembly_time += chrono::duration_cast<chrono::microseconds>(toc - tic).count();
            nonnul = !double_layer_interaction_matrix.isZero();
            for (jj = 0; jj < Q; ++jj) {
                for (ii = 0; ii < Q; ++ii) {
                    II = data.test_space.LocGlobMap(ii + 1, j + 1, dim) - 1;
                    JJ = data.test_space.LocGlobMap(jj + 1, i + 1, dim) - 1;
                    if (nonnul)
                        double_layer_matrix(II, JJ) += double_layer_interaction_matrix(ii, jj);
                    hypersingular_matrix(II, JJ) += hypersingular_interaction_matrix(jj, ii);
                    single_layer_matrix(II, JJ) += single_layer_interaction_matrix(jj, ii);
                    if (!der) continue;
                    double_layer_der_matrix(II, JJ) += double_layer_der_interaction_matrix(ii, jj);
                    hypersingular_der_matrix(II, JJ) += hypersingular_der_interaction_matrix(jj, ii);
                    single_layer_der_matrix(II, JJ) += single_layer_der_interaction_matrix(jj, ii);
                    if (der > 1) {
                        double_layer_der2_matrix(II, JJ) += double_layer_der2_interaction_matrix(ii, jj);
                        hypersingular_der2_matrix(II, JJ) += hypersingular_der2_interaction_matrix(jj, ii);
                        single_layer_der2_matrix(II, JJ) += single_layer_der2_interaction_matrix(jj, ii);
                    }
                }
            }
        }
    }
    double_layer_matrix *= 0.25;
    hypersingular_matrix *= 0.25;
    single_layer_matrix *= 0.25;
    if (der > 0) {
        double_layer_der_matrix *= 0.25;
        hypersingular_der_matrix *= 0.25;
        single_layer_der_matrix *= 0.25;
    }
    if (der > 1) {
        double_layer_der2_matrix *= 0.25;
        hypersingular_der2_matrix *= 0.25;
        single_layer_der2_matrix *= 0.25;
    }
}

const Eigen::MatrixXcd & GalerkinBuilder::getDoubleLayer(int der) const {
    if (der == 0)
        return double_layer_matrix;
    if (der == 1)
        return double_layer_der_matrix;
    if (der == 2)
        return double_layer_der2_matrix;
    throw std::runtime_error("Invalid order of derivative");
}

const Eigen::MatrixXcd & GalerkinBuilder::getHypersingular(int der) const {
    if (der == 0)
        return hypersingular_matrix;
    if (der == 1)
        return hypersingular_der_matrix;
    if (der == 2)
        return hypersingular_der2_matrix;
    throw std::runtime_error("Invalid order of derivative");
}

const Eigen::MatrixXcd & GalerkinBuilder::getSingleLayer(int der) const {
    if (der == 0)
        return single_layer_matrix;
    if (der == 1)
        return single_layer_der_matrix;
    if (der == 2)
        return single_layer_der2_matrix;
    throw std::runtime_error("Invalid order of derivative");
}

unsigned int GalerkinBuilder::getInteractionMatrixAssemblyTime() {
    auto ret = interaction_matrix_assembly_time;
    interaction_matrix_assembly_time = 0;
    return ret;
}

unsigned int GalerkinBuilder::getHankelComputationTime() {
    auto ret = hankel_computation_time;
    hankel_computation_time = 0;
    return ret;
}

unsigned int GalerkinBuilder::getPanelInteractionDataTime() {
    auto ret = panel_interaction_data_time;
    panel_interaction_data_time = 0;
    return ret;
}

void GalerkinBuilder::initializeSparseAssembly(const std::complex<double>& k_in, double c_in) {
    initialize_parameters(k_in, c_in);
    dlp_sparse.clear();
    slp_sparse.clear();
    hyp_sparse.clear();
    computed_imat_indices.clear();
    assert(data.testTrialSpacesAreEqual() && data.getQtest() == 2);
}

std::complex<double> GalerkinBuilder::getDoubleLayerElement(size_t row, size_t col) {
    size_t i = row >= col ? row : col, j = row >= col ? col : row, ii, jj, II, JJ, r, c;
    size_t numpanels = data.getNumPanels();
    size_t ip = i == 0 ? numpanels - 1 : i - 1, jp = j == 0 ? numpanels - 1 : j - 1;
    bool adj, swap, nonnul;
    for (r = 0; r < 2; ++r) for (c = 0; c < 2; ++c) {
        i = (ip + r) % numpanels, j = (jp + c) % numpanels;
        if (i < j) std::swap(i, j);
        auto ins = computed_imat_indices.insert({i, j});
        if (!ins.second)
            continue;
        if (i == j) {
            // coinciding panels
            compute_coinciding(i);
            all_coinciding(0);
        } else if ((adj = data.is_adjacent(i, j, swap))) {
            // adjacent panels
            compute_adjacent(i, j, swap);
            all_adjacent(swap, 0);
        } else {
            // disjoint panels
            compute_general(i, j);
            all_general(0);
        }
        // Local to global mapping of the elements in interaction matrix
        if ((nonnul = !double_layer_interaction_matrix.isZero()))
            double_layer_interaction_matrix *= 0.25;
        single_layer_interaction_matrix *= 0.25;
        hypersingular_interaction_matrix *= 0.25;
        for (jj = 0; jj < 2; ++jj) {
            for (ii = 0; ii < 2; ++ii) {
                II = data.test_space.LocGlobMap(ii + 1, i + 1, numpanels) - 1;
                JJ = data.test_space.LocGlobMap(jj + 1, j + 1, numpanels) - 1;
                if (nonnul)
                    dlp_sparse[{II, JJ}] += double_layer_interaction_matrix(ii, jj);
                slp_sparse[{II, JJ}] += single_layer_interaction_matrix(ii, jj);
                hyp_sparse[{II, JJ}] += hypersingular_interaction_matrix(ii, jj);
            }
        }
        if (i == j)
            continue;
        // reuse data for the (j, i) case
        if (adj)
            double_layer_adjacent(!swap, 0, true);
        else
            double_layer_general(0, true);
        if ((nonnul = !double_layer_interaction_matrix.isZero()))
            double_layer_interaction_matrix *= 0.25;
        single_layer_interaction_matrix *= 0.25;
        hypersingular_interaction_matrix *= 0.25;
        for (jj = 0; jj < 2; ++jj) {
            for (ii = 0; ii < 2; ++ii) {
                II = data.test_space.LocGlobMap(ii + 1, j + 1, numpanels) - 1;
                JJ = data.test_space.LocGlobMap(jj + 1, i + 1, numpanels) - 1;
                if (nonnul)
                    dlp_sparse[{II, JJ}] += double_layer_interaction_matrix(ii, jj);
                slp_sparse[{II, JJ}] += single_layer_interaction_matrix(jj, ii);
                hyp_sparse[{II, JJ}] += hypersingular_interaction_matrix(jj, ii);
            }
        }
    }
    return dlp_sparse[{row, col}];
}

std::complex<double> GalerkinBuilder::getSingleLayerElement(size_t row, size_t col) {
    size_t i = row >= col ? row : col, j = row >= col ? col : row, ii, jj, II, JJ, r, c;
    size_t numpanels = data.getNumPanels();
    size_t ip = i == 0 ? numpanels - 1 : i - 1, jp = j == 0 ? numpanels - 1 : j - 1;
    bool adj, swap;
    for (r = 0; r < 2; ++r) for (c = 0; c < 2; ++c) {
        i = (ip + r) % numpanels, j = (jp + c) % numpanels;
        if (i < j) std::swap(i, j);
        auto ins = computed_imat_indices.insert({i, j});
        if (!ins.second)
            continue;
        if (i == j) {
            // coinciding panels
            compute_coinciding(i);
            single_layer_coinciding(0);
            hypersingular_coinciding(0);
        } else if ((adj = data.is_adjacent(i, j, swap))) {
            // adjacent panels
            compute_adjacent(i, j, swap);
            single_layer_adjacent(swap, 0);
            hypersingular_adjacent(swap, 0);
        } else {
            // disjoint panels
            compute_general(i, j);
            single_layer_general(0);
            hypersingular_general(0);
        }
        single_layer_interaction_matrix *= 0.25;
        hypersingular_interaction_matrix *= 0.25;
        // Local to global mapping of the elements in interaction matrix
        for (jj = 0; jj < 2; ++jj) {
            for (ii = 0; ii < 2; ++ii) {
                II = data.test_space.LocGlobMap(ii + 1, i + 1, numpanels) - 1;
                JJ = data.test_space.LocGlobMap(jj + 1, j + 1, numpanels) - 1;
                slp_sparse[{II, JJ}] += single_layer_interaction_matrix(ii, jj);
                hyp_sparse[{II, JJ}] += hypersingular_interaction_matrix(ii, jj);
                // reuse data for the (j, i) case
                II = data.test_space.LocGlobMap(ii + 1, j + 1, numpanels) - 1;
                JJ = data.test_space.LocGlobMap(jj + 1, i + 1, numpanels) - 1;
                slp_sparse[{II, JJ}] += single_layer_interaction_matrix(jj, ii);
                hyp_sparse[{II, JJ}] += hypersingular_interaction_matrix(jj, ii);
            }
        }
    }
    return slp_sparse[{row, col}];
}

std::complex<double> GalerkinBuilder::getHypersingularElement(size_t row, size_t col) {
        size_t i = row >= col ? row : col, j = row >= col ? col : row, ii, jj, II, JJ, r, c;
    size_t numpanels = data.getNumPanels();
    size_t ip = i == 0 ? numpanels - 1 : i - 1, jp = j == 0 ? numpanels - 1 : j - 1;
    bool adj, swap;
    for (r = 0; r < 2; ++r) for (c = 0; c < 2; ++c) {
        i = (ip + r) % numpanels, j = (jp + c) % numpanels;
        if (i < j) std::swap(i, j);
        auto ins = computed_imat_indices.insert({i, j});
        if (!ins.second)
            continue;
        if (i == j) {
            // coinciding panels
            compute_coinciding(i);
            hypersingular_coinciding(0);
        } else if ((adj = data.is_adjacent(i, j, swap))) {
            // adjacent panels
            compute_adjacent(i, j, swap);
            hypersingular_adjacent(swap, 0);
        } else {
            // disjoint panels
            compute_general(i, j);
            hypersingular_general(0);
        }
        hypersingular_interaction_matrix *= 0.25;
        // Local to global mapping of the elements in interaction matrix
        for (jj = 0; jj < 2; ++jj) {
            for (ii = 0; ii < 2; ++ii) {
                II = data.test_space.LocGlobMap(ii + 1, i + 1, numpanels) - 1;
                JJ = data.test_space.LocGlobMap(jj + 1, j + 1, numpanels) - 1;
                hyp_sparse[{II, JJ}] += hypersingular_interaction_matrix(ii, jj);
                // reuse data for the (j, i) case
                II = data.test_space.LocGlobMap(ii + 1, j + 1, numpanels) - 1;
                JJ = data.test_space.LocGlobMap(jj + 1, i + 1, numpanels) - 1;
                hyp_sparse[{II, JJ}] += hypersingular_interaction_matrix(jj, ii);
            }
        }
    }
    return slp_sparse[{row, col}];
}

