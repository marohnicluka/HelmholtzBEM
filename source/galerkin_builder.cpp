#include "galerkin_builder.hpp"
#include <numeric>
#include <iostream>
#include <chrono>
#include <set>
#ifdef PARALLELIZE_BUILDER
#include <execution>
#define EXEC_POLICY std::execution::par,
#else
#define EXEC_POLICY
#endif

//#define ENABLE_SPECIAL_CASES_FOR_PANEL_INTERACTION 1

static const double eps_mach = std::numeric_limits<double>::epsilon();

BuilderData::BuilderData(const ParametrizedMesh &mesh_in,
                         const AbstractBEMSpace &test_space_in,
                         const AbstractBEMSpace &trial_space_in,
                         unsigned order)
    : mesh(mesh_in), test_space(test_space_in), trial_space(trial_space_in)
{
    panels_are_lines = mesh_in.isPolygonal();
    GaussQR = getGaussQR(order, 0., 1.);
    CGaussQR = getCGaussQR(order);
    npanels = mesh.getNumPanels();
    dim_test = test_space.getSpaceDim(npanels);
    dim_trial = trial_space.getSpaceDim(npanels);
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
                        m_dlp_gen_fg(i * Ns + ii, j * Ns + jj) =
                            test_space.evaluateShapeFunction(i, s) * trial_space.evaluateShapeFunction(j, t);
                        m_dlp_gen_fg_t(i * Ns + ii, j * Ns + jj) =
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
            m_dlp_gen_fg.block(i * Ns, j * Ns, Ns, Ns) *= m_wg;
            m_dlp_gen_fg_t.block(i * Ns, j * Ns, Ns, Ns) *= m_wg;
        }
        for (size_t i = 0; i < Qtrial; ++i) {
            for (size_t jj = 0; jj < N; ++jj) {
                for (size_t ii = 0; ii < N; ++ii) {
                    double sc = m_sc(ii, jj), sa = m_sa(ii, jj), tc = m_tc(ii, jj), ta = m_ta(ii, jj);
                    m_hyp_cnc_fg(i * N + ii, j * N + jj) =
                        trial_space.evaluateShapeFunction(i, sc) * trial_space.evaluateShapeFunction(j, tc);
                    m_hyp_cnc_fg_arc(i * N + ii, j * N + jj) =
                        trial_space.evaluateShapeFunctionDot_01(i, sc) * trial_space.evaluateShapeFunctionDot_01(j, tc);
                    m_hyp_adj_fg(i * N + ii, j * N + jj) =
                        trial_space.evaluateShapeFunction_01_swapped(i, sa) * trial_space.evaluateShapeFunction(j, ta);
                    m_hyp_adj_fg_swap(i * N + ii, j * N + jj) =
                        trial_space.evaluateShapeFunction(i, sa) * trial_space.evaluateShapeFunction_01_swapped(j, ta);
                    m_hyp_adj_fg_arc(i * N + ii, j * N + jj) =
                        trial_space.evaluateShapeFunctionDot_01_swapped(i, sa) * trial_space.evaluateShapeFunctionDot_01(j, ta);
                    m_hyp_adj_fg_arc_swap(i * N + ii, j * N + jj) =
                        trial_space.evaluateShapeFunctionDot_01(i, sa) * trial_space.evaluateShapeFunctionDot_01_swapped(j, ta);
                    if (ii < Ns && jj < Ns) {
                        double s = m_sg(ii, jj), t = m_tg(ii, jj);
                        m_hyp_gen_fg(i * Ns + ii, j * Ns + jj) =
                            trial_space.evaluateShapeFunction(i, s) * trial_space.evaluateShapeFunction(j, t);
                        m_hyp_gen_fg_arc(i * Ns + ii, j * Ns + jj) =
                            trial_space.evaluateShapeFunctionDot_01(i, s) * trial_space.evaluateShapeFunctionDot_01(j, t);
                    }
                }
            }
            m_hyp_cnc_fg.block(i * N, j * N, N, N) *= m_wc;
            m_hyp_cnc_fg_arc.block(i * N, j * N, N, N) *= m_wc;
            m_hyp_adj_fg.block(i * N, j * N, N, N) *= m_wa;
            m_hyp_adj_fg_swap.block(i * N, j * N, N, N) *= m_wa;
            m_hyp_adj_fg_arc.block(i * N, j * N, N, N) *= m_wa;
            m_hyp_adj_fg_arc_swap.block(i * N, j * N, N, N) *= m_wa;
            m_hyp_gen_fg_arc.block(i * Ns, j * Ns, Ns, Ns) *= m_wg;
            m_hyp_gen_fg.block(i * Ns, j * Ns, Ns, Ns) *= m_wg;
        }
    }
    for (size_t i = 0; i < Qtest; ++i) {
        for (size_t j = 0; j < Qtest; ++j) {
            for (size_t jj = 0; jj < N; ++jj) {
                for (size_t ii = 0; ii < N; ++ii) {
                    double sc = m_sc(ii, jj), sa = m_sa(ii, jj), tc = m_tc(ii, jj), ta = m_ta(ii, jj);
                    m_slp_cnc_fg(i * N + ii, j * N + jj) =
                        test_space.evaluateShapeFunction(i, sc) * test_space.evaluateShapeFunction(j, tc);
                    m_slp_adj_fg(i * N + ii, j * N + jj) =
                        test_space.evaluateShapeFunction_01_swapped(i, sa) * test_space.evaluateShapeFunction(j, ta);
                    m_slp_adj_fg_swap(i * N + ii, j * N + jj) =
                        test_space.evaluateShapeFunction(i, sa) * test_space.evaluateShapeFunction_01_swapped(j, ta);
                    if (ii < Ns && jj < Ns) {
                        double s = m_sg(ii, jj), t = m_tg(ii, jj);
                        m_slp_gen_fg(i * Ns + ii, j * Ns + jj) =
                            test_space.evaluateShapeFunction(i, s) * test_space.evaluateShapeFunction(j, t);
                    }
                }
            }
            m_slp_cnc_fg.block(i * N, j * N, N, N) *= m_wc;
            m_slp_adj_fg.block(i * N, j * N, N, N) *= m_wa;
            m_slp_adj_fg_swap.block(i * N, j * N, N, N) *= m_wa;
            m_slp_gen_fg.block(i * Ns, j * Ns, Ns, Ns) *= m_wg;
        }
    }
    size_t Nsd = panels_are_lines ? 1 : Ns;
    Derivative_01_sg.resize(Nsd, Nsd * npanels);
    Derivative_01_tg.resize(Nsd, Nsd * npanels);
    Derivative_01_sg_n.resize(Nsd, Nsd * npanels);
    Derivative_01_tg_n.resize(Nsd, Nsd * npanels);
    op_sg.resize(Ns, Ns * npanels);
    op_tg.resize(Ns, Ns * npanels);
    Eigen::ArrayXXcd tmps;
    Eigen::ArrayXXd tmps_n;
    for (size_t i = 0; i < npanels; ++i) {
        const auto &p = panel(i);
        p.Derivative_01(m_sg.block(0, 0, Nsd, Nsd), tmps, tmps_n);
        Derivative_01_sg.block(0, Nsd * i, Nsd, Nsd) = tmps;
        Derivative_01_sg_n.block(0, Nsd * i, Nsd, Nsd) = tmps_n;
        p.Derivative_01(m_tg.block(0, 0, Nsd, Nsd), tmps, tmps_n);
        Derivative_01_tg.block(0, Nsd * i, Nsd, Nsd) = tmps;
        Derivative_01_tg_n.block(0, Nsd * i, Nsd, Nsd) = tmps_n;
        op_sg.block(0, i * Ns, Ns, Ns) = p[m_sg];
        op_tg.block(0, i * Ns, Ns, Ns) = p[m_tg];
    }
}

bool BuilderData::is_adjacent(size_t i, size_t j, bool &swap) const noexcept {
    const auto &p1 = panel(i), &p2 = panel(j);
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
    panels_are_lines = data.getPanelsAreLines();
#if defined(HANKEL_VECTORIZE) && !defined(PARALLELIZE_BUILDER)
    size_t N = data.getCGaussQROrder(), Ns = data.getGaussQROrder();
    H1R01.initialize(N, N);
    H1R01_s.initialize(Ns, Ns);
#endif
#ifdef PARALLELIZE_BUILDER
    Eigen::initParallel();
    Eigen::setNbThreads(1);
#endif
}

void replace_non_finite_elements_with_zeros(Eigen::ArrayXXcd &arr) {
    arr = arr.unaryExpr([](const std::complex<double>& val) -> std::complex<double> {
        if (!std::isfinite(val.real()) || !std::isfinite(val.imag()))
            return std::complex<double>(0.0, 0.0);
        return val;
    });
}

void GalerkinBuilder::H1_01_cplx(const Eigen::ArrayXXcd& x, Eigen::ArrayXXcd& h_0, Eigen::ArrayXXcd& h_1) {
    size_t nr = x.rows(), nc = x.cols(), i, j;
    h_0.resize(nr, nc);
    h_1.resize(nr, nc);
    for (j = 0; j < nc; ++j) for (i = 0; i < nr; ++i) {
        const auto &z = x(i, j);
        h_0(i, j) = complex_bessel::HankelH1(0, z);
        h_1(i, j) = complex_bessel::HankelH1(1, z);
    }
    replace_non_finite_elements_with_zeros(h_0);
    replace_non_finite_elements_with_zeros(h_1);
}

inline void GalerkinBuilder::compute_coinciding(workspace_t &ws) noexcept {
    assert(ws.i == ws.j);
    size_t N = data.getCGaussQROrder();
    const auto &p = data.panel(ws.i);
    if (panels_are_lines) {
        p.Derivative_01(data.m_tc.block(0, 0, 1, 1), ws.m_tangent_p[0], ws.m_tangent_p_norm[0]);
        p.Derivative_01(data.m_sc.block(0, 0, 1, 1), ws.m_tangent[0], ws.m_tangent_norm[0]);
    } else {
        p.Derivative_01(data.m_tc, ws.m_tangent_p[0], ws.m_tangent_p_norm[0]);
        p.Derivative_01(data.m_sc, ws.m_tangent[0], ws.m_tangent_norm[0]);
    }
    ws.m_v[0] = p[data.m_sc] - p[data.m_tc];
    ws.m_v_norm2[0] = ws.m_v[0].cwiseAbs2();
    ws.m_v_norm[0] = ws.m_v_norm2[0].cwiseSqrt();
    /** compute Hankel kernel data */
    if (k_rp) {
#ifdef HANKEL_VECTORIZE
#ifdef PARALLELIZE_BUILDER
        ws.H1R01.initialize(N, N);
        ws.
#endif
        H1R01.h1_01(ksqrtc.real() * ws.m_v_norm[0], ws.m_h0[0], ws.m_h1[0]);
#else
        ws.m_h0[0].resize(N, N), ws.m_h1[0].resize(N, N);
        for (size_t j = 0; j < N; ++j) for (size_t i = 0; i < N; ++i)
            complex_bessel::H1_01(ksqrtc.real() * ws.m_v_norm[0](i, j), ws.m_h0[0](i, j), ws.m_h1[0](i, j));
#endif
    } else H1_01_cplx(ksqrtc * ws.m_v_norm[0], ws.m_h0[0], ws.m_h1[0]);
}

inline void GalerkinBuilder::compute_adjacent(bool swap, workspace_t &ws) noexcept {
    size_t K, N = data.getCGaussQROrder();;
    const auto &p = data.panel(ws.i), &q = data.panel(ws.j);
    if (panels_are_lines) {
        p.Derivative_01(data.m_sa.block(0, 0, 1, 1), ws.m_tangent[0], ws.m_tangent_norm[0]);
        q.Derivative_01(data.m_ta.block(0, 0, 1, 1), ws.m_tangent_p[0], ws.m_tangent_p_norm[0]);
    } else if (swap) {
        q.Derivative_01_swapped(data.m_ta, ws.m_tangent_p[0], ws.m_tangent_p_norm[0], true);
        q.Derivative_01_swapped(data.m_sa, ws.m_tangent_p[1], ws.m_tangent_p_norm[1], true);
        p.Derivative_01(data.m_sa, ws.m_tangent[0], ws.m_tangent_norm[0]);
        p.Derivative_01(data.m_ta, ws.m_tangent[1], ws.m_tangent_norm[1]);
    } else {
        q.Derivative_01(data.m_ta, ws.m_tangent_p[0], ws.m_tangent_p_norm[0]);
        q.Derivative_01(data.m_sa, ws.m_tangent_p[1], ws.m_tangent_p_norm[1]);
        p.Derivative_01_swapped(data.m_sa, ws.m_tangent[0], ws.m_tangent_norm[0], true);
        p.Derivative_01_swapped(data.m_ta, ws.m_tangent[1], ws.m_tangent_norm[1], true);
    }
    ws.m_v[0] = swap ? p[data.m_sa] - q.swapped_op(data.m_ta) : p.swapped_op(data.m_sa) - q[data.m_ta];
    ws.m_v[1] = swap ? p[data.m_ta] - q.swapped_op(data.m_sa) : p.swapped_op(data.m_ta) - q[data.m_sa];
    ws.m_v_norm2[0] = ws.m_v[0].cwiseAbs2();
    ws.m_v_norm2[1] = ws.m_v[1].cwiseAbs2();
    ws.m_v_norm[0] = ws.m_v_norm2[0].cwiseSqrt();
    ws.m_v_norm[1] = ws.m_v_norm2[1].cwiseSqrt();
    /** compute Hankel kernel data */
    for (K = 0; K < 2; ++K) {
        if (k_rp) {
#ifdef HANKEL_VECTORIZE
#ifdef PARALLELIZE_BUILDER
            ws.H1R01.initialize(N, N);
            ws.
#endif
            H1R01.h1_01(ksqrtc.real() * ws.m_v_norm[K], ws.m_h0[K], ws.m_h1[K]);
#else
            ws.m_h0[K].resize(N, N), ws.m_h1[K].resize(N, N);
            for (size_t j = 0; j < N; ++j) for (size_t i = 0; i < N; ++i)
                complex_bessel::H1_01(ksqrtc.real() * ws.m_v_norm[K](i, j), ws.m_h0[K](i, j), ws.m_h1[K](i, j));
#endif
        } else H1_01_cplx(ksqrtc * ws.m_v_norm[K], ws.m_h0[K], ws.m_h1[K]);
    }
}

inline void GalerkinBuilder::compute_general(workspace_t &ws) noexcept {
    size_t N = data.getGaussQROrder();
    ws.m_v_s = data.op_sg.block(0, N * ws.i, N, N) - data.op_tg.block(0, N * ws.j, N, N);
    ws.m_v_norm2_s = ws.m_v_s.cwiseAbs2();
    ws.m_v_norm_s = ws.m_v_norm2_s.cwiseSqrt();
    /** compute Hankel kernel data */
    if (k_rp) {
#ifdef HANKEL_VECTORIZE
#ifdef PARALLELIZE_BUILDER
        ws.H1R01_s.initialize(N, N);
        ws.
#endif
        H1R01_s.h1_01(ksqrtc.real() * ws.m_v_norm_s, ws.m_h0_s, ws.m_h1_s);
#else
        ws.m_h0_s.resize(N, N), ws.m_h1_s.resize(N, N);
        for (size_t j = 0; j < N; ++j) for (size_t i = 0; i < N; ++i)
            complex_bessel::H1_01(ksqrtc.real() * ws.m_v_norm_s(i, j), ws.m_h0_s(i, j), ws.m_h1_s(i, j));
#endif
    } else H1_01_cplx(ksqrtc * ws.m_v_norm_s, ws.m_h0_s, ws.m_h1_s);
}

template<typename T>
bool GalerkinBuilder::isArrayZero(const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>& ar) noexcept {
    return (ar.cwiseAbs() < eps_mach).all();
}

GalerkinBuilder::IndexPair GalerkinBuilder::map_local_to_global(const IndexPair &ij, const IndexPair &kl, LayerType layer) const noexcept {
    IndexPair ret;
    size_t dim_test = data.getTestSpaceDimension(), dim_trial = data.getTrialSpaceDimension();
    if (layer == LayerType::HYPERSINGULAR)
        ret.first = data.trial_space.LocGlobMap(kl.first + 1, ij.first + 1, dim_trial) - 1;
    else ret.first = data.test_space.LocGlobMap(kl.first + 1, ij.first + 1, dim_test) - 1;
    if (layer == LayerType::SINGLE)
        ret.second = data.test_space.LocGlobMap(kl.second + 1, ij.second + 1, dim_test) - 1;
    else ret.second = data.trial_space.LocGlobMap(kl.second + 1, ij.second + 1, dim_trial) - 1;
    return ret;
}

void GalerkinBuilder::double_layer_coinciding(size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept {
    size_t i, j, K, N = data.getCGaussQROrder(), Qtest = data.getQtest(), Qtrial = data.getQtrial();
    assert(res.rows() == Qtest && res.cols() >= Qtrial * (d + 1));
    res.setZero();
    const auto &v = ws.m_v[0];
    const auto &v_norm = ws.m_v_norm[0];
    const auto &h1 = ws.m_h1[0], &h0 = ws.m_h0[0];
    ws.m_1[0] = (v_norm <= 0).select(Eigen::ArrayXXcd::Zero(N, N), ksqrtc * h1 / v_norm);
#ifdef ENABLE_SPECIAL_CASES_FOR_PANEL_INTERACTION
    if (Qtest == 2 && Qtrial == 2 && !d) {
        const auto &tangent_p = ws.m_tangent_p[0], &tangent = ws.m_tangent[0];
        if (panels_are_lines) {
            ws.m_1[1] = (v.imag() * tangent(0, 0).real() - v.real() * tangent(0, 0).imag()) * ws.m_1[0] * ws.m_tangent_p_norm[0](0, 0);
            ws.m_1[0] *= -(v.imag() * tangent_p(0, 0).real() - v.real() * tangent_p(0, 0).imag()) * ws.m_tangent_norm[0](0, 0);
        } else {
            ws.m_1[1] = (v.imag() * tangent.real() - v.real() * tangent.imag()) * ws.m_1[0] * ws.m_tangent_p_norm[0];
            ws.m_1[0] *= -(v.imag() * tangent_p.real() - v.real() * tangent_p.imag()) * ws.m_tangent_norm[0];
        }
        if (!isArrayZero(ws.m_1[0]) || !isArrayZero(ws.m_1[1]))
            res.block(0, 0, 2, 2) <<
                (data.m_dlp_cnc_fg.block(0, 0, N, N) * ws.m_1[0] + data.m_dlp_cnc_fg_t.block(0, 0, N, N) * ws.m_1[1]).sum(),
                (data.m_dlp_cnc_fg.block(0, N, N, N) * ws.m_1[0] + data.m_dlp_cnc_fg_t.block(0, N, N, N) * ws.m_1[1]).sum(),
                (data.m_dlp_cnc_fg.block(N, 0, N, N) * ws.m_1[0] + data.m_dlp_cnc_fg_t.block(N, 0, N, N) * ws.m_1[1]).sum(),
                (data.m_dlp_cnc_fg.block(N, N, N, N) * ws.m_1[0] + data.m_dlp_cnc_fg_t.block(N, N, N, N) * ws.m_1[1]).sum();
        return;
    }
#endif
    if (d > 1) {
        ws.m_3[0] = h0 - h1 * ksqrtc * v_norm;
    }
    for (K = 0; K < 2; ++K) {
        const auto &tangent = (K == 0 ? ws.m_tangent_p : ws.m_tangent)[0];
        const auto &tangent_norm = (K == 0 ? ws.m_tangent_norm : ws.m_tangent_p_norm)[0];
        if (panels_are_lines)
            ws.vdotn = (K * 2 - 1.) * tangent_norm(0, 0) * (v.imag() * tangent(0, 0).real() - v.real() * tangent(0, 0).imag());
        else ws.vdotn = (K * 2 - 1.) * tangent_norm * (v.imag() * tangent.real() - v.real() * tangent.imag());
        if (isArrayZero(ws.vdotn))
            continue;
        for (j = 0; j < Qtrial; ++j) {
            for (i = 0; i < Qtest; ++i) {
                ws.cfr = ws.vdotn * (K == 0 ? data.m_dlp_cnc_fg : data.m_dlp_cnc_fg_t).block(i * N, j * N, N, N);
                res(i, j) += (ws.cfr * ws.m_1[0]).sum();
                if (!d) continue;
                res(i, j + Qtrial) += (ws.cfr * h0).sum();
                if (d > 1)
                    res(i, j + 2 * Qtrial) += (ws.cfr * ws.m_3[0]).sum();
            }
        }
    }
    if (!d) return;
    res.block(0, Qtrial, Qtest, Qtrial) *= k * c;
    if (d > 1)
        res.block(0, 2 * Qtrial, Qtest, Qtrial) *= c;
}

void GalerkinBuilder::double_layer_adjacent(bool swap, size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept {
    size_t i, j, K, N = data.getCGaussQROrder(), Qtest = data.getQtest(), Qtrial = data.getQtrial();
    size_t step = ws.compute_transposed ? 2 : 1;
    assert(res.rows() == Qtest && res.cols() == step * Qtrial * (d + 1));
    res.setZero();
    for (K = 0; K < 2; ++K) {
        const auto &tangent = ws.m_tangent_p[panels_are_lines ? 0 : K];
        const auto &tangent_t = ws.m_tangent[panels_are_lines ? 0 : K];
        const auto &tangent_norm = ws.m_tangent_norm[panels_are_lines ? 0 : K];
        const auto &tangent_norm_t = ws.m_tangent_p_norm[panels_are_lines ? 0 : K];
        const auto &v = ws.m_v[K];
        const auto &v_norm = ws.m_v_norm[K];
        const auto &h0 = ws.m_h0[K], &h1 = ws.m_h1[K];
        ws.m_1[K] = (v_norm <= 0).select(Eigen::ArrayXXcd::Zero(N, N), ksqrtc * h1 / v_norm);
        if (d > 1)
            ws.m_3[K] = h0 - ksqrtc * v_norm * h1;
        if (panels_are_lines) {
            ws.vdotn = -(v.imag() * tangent(0, 0).real() - v.real() * tangent(0, 0).imag()) * tangent_norm(0, 0);
            if (ws.compute_transposed)
                ws.vdotn_t = (v.imag() * tangent_t(0, 0).real() - v.real() * tangent_t(0, 0).imag()) * tangent_norm_t(0, 0);
        } else {
            ws.vdotn = -(v.imag() * tangent.real() - v.real() * tangent.imag()) * tangent_norm;
            if (ws.compute_transposed)
                ws.vdotn_t = (v.imag() * tangent_t.real() - v.real() * tangent_t.imag()) * tangent_norm_t;
        }
        if (!isArrayZero(ws.vdotn)) {
            const auto &fg = swap ? (K == 1 ? data.m_dlp_adj_fg_swap_t : data.m_dlp_adj_fg_swap)
                                  : (K == 1 ? data.m_dlp_adj_fg_t : data.m_dlp_adj_fg);
            for (j = 0; j < Qtrial; ++j) {
                for (i = 0; i < Qtest; ++i) {
                    ws.cfr = ws.vdotn * fg.block(i * N, j * N, N, N);
                    res(i, j) += (ws.cfr * ws.m_1[K]).sum();
                    if (!d) continue;
                    res(i, step * Qtrial + j) += (ws.cfr * h0).sum();
                    if (d > 1)
                        res(i, 2 * step * Qtrial + j) += (ws.cfr * ws.m_3[K]).sum();
                }
            }
        }
        if (ws.compute_transposed && !isArrayZero(ws.vdotn_t)) {
            const auto &fg = swap ? (K == 0 ? data.m_dlp_adj_fg_t : data.m_dlp_adj_fg)
                                  : (K == 0 ? data.m_dlp_adj_fg_swap_t : data.m_dlp_adj_fg_swap);
            for (j = 0; j < Qtrial; ++j) {
                for (i = 0; i < Qtest; ++i) {
                    ws.cfr = ws.vdotn_t * fg.block(i * N, j * N, N, N);
                    res(i, Qtrial + j) += (ws.cfr * ws.m_1[K]).sum();
                    if (!d) continue;
                    res(i, 3 * Qtrial + j) += (ws.cfr * h0).sum();
                    if (d > 1)
                        res(i, 5 * Qtrial + j) += (ws.cfr * ws.m_3[K]).sum();
                }
            }
        }
    }
    if (!d) return;
    res.block(0, step * Qtrial, Qtest, step * Qtrial) *= k * c;
    if (d > 1)
        res.block(0, 2 * step * Qtrial, Qtest, step * Qtrial) *= c;
}

void GalerkinBuilder::double_layer_general(size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept {
    size_t i, j, N = data.getGaussQROrder(), Ns = panels_are_lines ? 1 : N, Qtest = data.getQtest(), Qtrial = data.getQtrial();
    size_t step = ws.compute_transposed ? 2 : 1;
    assert(res.rows() == Qtest && res.cols() == step * Qtrial * (d + 1));
    res.setZero();
    const auto tangent_s = data.Derivative_01_sg.block(0, Ns * ws.i, Ns, Ns);
    const auto tangent_norm_s = data.Derivative_01_sg_n.block(0, Ns * ws.i, Ns, Ns);
    const auto tangent_p_s = data.Derivative_01_tg.block(0, Ns * ws.j, Ns, Ns);
    const auto tangent_p_norm_s = data.Derivative_01_tg_n.block(0, Ns * ws.j, Ns, Ns);
    bool zr = false, zr_t = false;
    if (panels_are_lines) {
        ws.vdotn_s = ws.m_v_s.real() * tangent_p_s(0, 0).imag() - ws.m_v_s.imag() * tangent_p_s(0, 0).real();
        if (isArrayZero(ws.vdotn_s))
            zr = true;
        else ws.vdotn_s *= tangent_norm_s(0, 0);
        if (ws.compute_transposed) {
            ws.vdotn_t_s = ws.m_v_s.imag() * tangent_s(0, 0).real() - ws.m_v_s.real() * tangent_s(0, 0).imag();
            if (isArrayZero(ws.vdotn_t_s))
                zr_t = true;
            else ws.vdotn_t_s *= tangent_p_norm_s(0, 0);
        }
    } else {
        ws.vdotn_s = ws.m_v_s.real() * tangent_p_s.imag() - ws.m_v_s.imag() * tangent_p_s.real();
        if (isArrayZero(ws.vdotn_s))
            zr = true;
        else ws.vdotn_s *= tangent_norm_s;
        if (ws.compute_transposed) {
            ws.vdotn_t_s = ws.m_v_s.imag() * tangent_s.real() - ws.m_v_s.real() * tangent_s.imag();
            if (isArrayZero(ws.vdotn_t_s))
                zr_t = true;
            else ws.vdotn_t_s *= tangent_p_norm_s;
        }
    }
    if (zr && (!ws.compute_transposed || zr_t))
        return;
    ws.m_1_s = (ws.m_v_norm_s <= 0).select(Eigen::ArrayXXcd::Zero(N, N), ksqrtc * ws.m_h1_s / ws.m_v_norm_s);
    if (d > 1) ws.m_3_s = ws.m_h0_s - ws.m_1_s * ws.m_v_norm2_s;
#ifdef ENABLE_SPECIAL_CASES_FOR_PANEL_INTERACTION
    if (Qtrial == 2 && Qtest == 2 && d == 0) {
        if (!zr) {
            const auto &fg = data.m_dlp_gen_fg;
            ws.m_1_hg_s = ws.m_1_s * ws.vdotn_s;
            res.block(0, 0, 2, 2) << (fg.block(0, 0, N, N) * ws.m_1_hg_s).sum(), (fg.block(0, N, N, N) * ws.m_1_hg_s).sum(),
                                     (fg.block(N, 0, N, N) * ws.m_1_hg_s).sum(), (fg.block(N, N, N, N) * ws.m_1_hg_s).sum();
        }
        if (ws.compute_transposed && !zr_t) {
            const auto &fg = data.m_dlp_gen_fg_t;
            ws.m_1_hg_s = ws.m_1_s * ws.vdotn_t_s;
            res.block(0, 2, 2, 2) << (fg.block(0, 0, N, N) * ws.m_1_hg_s).sum(), (fg.block(0, N, N, N) * ws.m_1_hg_s).sum(),
                                     (fg.block(N, 0, N, N) * ws.m_1_hg_s).sum(), (fg.block(N, N, N, N) * ws.m_1_hg_s).sum();
        }
    }
#endif
    if (!zr) {
        const auto &fg = data.m_dlp_gen_fg;
        for (j = 0; j < Qtrial; ++j) {
            for (i = 0; i < Qtest; ++i) {
                ws.cfr_s = ws.vdotn_s * fg.block(i * N, j * N, N, N);
                res(i, j) = (ws.cfr_s * ws.m_1_s).sum();
                if (!d) continue;
                res(i, step * Qtrial + j) = (ws.cfr_s * ws.m_h0_s).sum();
                if (d > 1) res(i, 2 * step * Qtrial + j) = (ws.cfr_s * ws.m_3_s).sum();
            }
        }
    }
    if (ws.compute_transposed && !zr_t) {
        const auto &fg = data.m_dlp_gen_fg_t;
        for (j = 0; j < Qtrial; ++j) {
            for (i = 0; i < Qtest; ++i) {
                ws.cfr_s = ws.vdotn_t_s * fg.block(i * N, j * N, N, N);
                res(i, Qtrial + j) = (ws.cfr_s * ws.m_1_s).sum();
                if (!d) continue;
                res(i, 3 * Qtrial + j) = (ws.cfr_s * ws.m_h0_s).sum();
                if (d > 1) res(i, 5 * Qtrial + j) = (ws.cfr_s * ws.m_3_s).sum();
            }
        }
    }
    if (!d) return;
    res.block(0, step * Qtrial, Qtest, step * Qtrial) *= k * c;
    if (d > 1) res.block(0, 2 * step * Qtrial, Qtest, step * Qtrial) *= c;
}

void GalerkinBuilder::hypersingular_coinciding(size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept {
    size_t i, j, K, l, N = data.getCGaussQROrder(), Qtrial = data.getQtrial();
    assert(res.rows() == Qtrial && res.cols() == Qtrial * (d + 1));
    res.setZero();
    const auto &h0 = ws.m_h0[0];
    const auto &v_norm2 = ws.m_v_norm2[0];
    compute_tdottp(0, ws);
#ifdef ENABLE_SPECIAL_CASES_FOR_PANEL_INTERACTION
    if (Qtrial == 2 && k_rp && !d) {
        if (ws.tdottp_zero) {
            res(0, 0) = 2.0 * (data.m_hyp_cnc_fg_arc.block(0, 0, N, N) * h0).sum();
            res(1, 1) = 2.0 * (data.m_hyp_cnc_fg_arc.block(N, N, N, N) * h0).sum();
            res(1, 0) = res(0, 1) = ((data.m_hyp_cnc_fg_arc.block(N, 0, N, N) + data.m_hyp_cnc_fg_arc.block(0, N, N, N)) * h0).sum();
        } else {
            ws.tdottp *= k2ch.real();
            res(0, 0) = 2.0 * ((data.m_hyp_cnc_fg_arc.block(0, 0, N, N) - data.m_hyp_cnc_fg.block(0, 0, N, N) * ws.tdottp) * h0).sum();
            res(1, 1) = 2.0 * ((data.m_hyp_cnc_fg_arc.block(N, N, N, N) - data.m_hyp_cnc_fg.block(N, N, N, N) * ws.tdottp) * h0).sum();
            res(1, 0) = res(0, 1) = ((data.m_hyp_cnc_fg_arc.block(N, 0, N, N) + data.m_hyp_cnc_fg_arc.block(0, N, N, N) -
                                     (data.m_hyp_cnc_fg.block(N, 0, N, N) + data.m_hyp_cnc_fg.block(0, N, N, N)) * ws.tdottp) * h0).sum();
        }
        return;
    }
#endif
    if (d > 0)
        ws.h1_vnorm = ws.m_h1[0] * ws.m_v_norm[0];
    for (K = 0; K < 2; ++K) {
        for (j = 0; j < Qtrial; ++j) {
            for (i = 0; i < Qtrial; ++i) {
                if (i < j) { /** symmetry */
                    for (l = 0; l <= d; ++l) res(i, l * Qtrial + j) = res(j, l * Qtrial + i);
                    continue;
                }
                ws.cfr = data.m_hyp_cnc_fg_arc.block((K > 0 ? j : i) * N, (K > 0 ? i : j) * N, N, N);
                if (!ws.tdottp_zero) {
                    ws.temp = data.m_hyp_cnc_fg.block((K > 0 ? j : i) * N, (K > 0 ? i : j) * N, N, N) * ws.tdottp;
                    if (k_rp)
                        ws.cfr -= k2ch.real() * ws.temp;
                    else ws.cf = ws.cfr - k2ch * ws.temp;
                }
                if (ws.tdottp_zero || k_rp) {
                    res(i, j) += (ws.cfr * h0).sum();
                    if (!d) continue;
                    if (ws.tdottp_zero)
                        res(i, Qtrial + j) += (ws.h1_vnorm * ws.cfr).sum();
                    else res(i, Qtrial + j) += (ws.h1_vnorm * ws.cfr + ksqrtc * h0 * ws.temp).sum();
                    if (d > 1) {
                        if (ws.tdottp_zero)
                            res(i, 2 * Qtrial + j) += ((h0 * v_norm2 - ws.h1_vnorm * ksqrtc_inv) * ws.cfr).sum();
                        else res(i, 2 * Qtrial + j) += ((h0 * v_norm2 - ws.h1_vnorm * ksqrtc_inv) * ws.cfr - ws.temp * (ksqrtc_two * ws.h1_vnorm - h0)).sum();
                    }
                } else {
                    res(i, j) += (ws.cf * h0).sum();
                    if (!d) continue;
                    res(i, Qtrial + j) += (ws.h1_vnorm * ws.cf + ksqrtc * h0 * ws.temp).sum();
                    if (d > 1)
                        res(i, 2 * Qtrial + j) += ((h0 * v_norm2 - ws.h1_vnorm * ksqrtc_inv) * ws.cf - ws.temp * (ksqrtc_two * ws.h1_vnorm - h0)).sum();
                }
            }
        }
    }
    if (!d) return;
    res.block(0, Qtrial, Qtrial, Qtrial) *= -sqrtc;
    if (d > 1)
        res.block(0, 2 * Qtrial, Qtrial, Qtrial) *= -c;
}

void GalerkinBuilder::hypersingular_adjacent(bool swap, size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept {
    size_t i, j, K, N = data.getCGaussQROrder(), Qtrial = data.getQtrial();
    assert(res.rows() == Qtrial && res.cols() == Qtrial * (d + 1));
    res.setZero();
    for (K = 0; K < 2; ++K) {
        const auto &h0 = ws.m_h0[K];
        const auto &v_norm2 = ws.m_v_norm2[K];
        if (!panels_are_lines || K == 0)
            compute_tdottp(K, ws);
        if (d > 0)
            ws.h1_vnorm = ws.m_h1[K] * ws.m_v_norm[K];
        for (j = 0; j < Qtrial; ++j) {
            for (i = 0; i < Qtrial; ++i) {
                ws.cfr = swap ? (K > 0 ? data.m_hyp_adj_fg_arc.block(j * N, i * N, N, N)
                                       : data.m_hyp_adj_fg_arc_swap.block(i * N, j * N, N, N))
                              : (K > 0 ? data.m_hyp_adj_fg_arc_swap.block(j * N, i * N, N, N)
                                       : data.m_hyp_adj_fg_arc.block(i * N, j * N, N, N));
                if (!ws.tdottp_zero) {
                    ws.temp = (swap ? (K > 0 ? data.m_hyp_adj_fg.block(j * N, i * N, N, N)
                                             : data.m_hyp_adj_fg_swap.block(i * N, j * N, N, N))
                                    : (K > 0 ? data.m_hyp_adj_fg_swap.block(j * N, i * N, N, N)
                                             : data.m_hyp_adj_fg.block(i * N, j * N, N, N))) * ws.tdottp;
                    if (k_rp)
                        ws.cfr -= k2ch.real() * ws.temp;
                    else ws.cf = ws.cfr - k2ch * ws.temp;
                }
                if (ws.tdottp_zero || k_rp) {
                    res(i, j) += (ws.cfr * h0).sum();
                    if (!d) continue;
                    if (ws.tdottp_zero)
                        res(i, Qtrial + j) += (ws.h1_vnorm * ws.cfr).sum();
                    else res(i, Qtrial + j) += (ws.h1_vnorm * ws.cfr + ksqrtc * h0 * ws.temp).sum();
                    if (d > 1) {
                        if (ws.tdottp_zero)
                            res(i, 2 * Qtrial + j) += ((h0 * v_norm2 - ws.h1_vnorm * ksqrtc_inv) * ws.cfr).sum();
                        else res(i, 2 * Qtrial + j) += ((h0 * v_norm2 - ws.h1_vnorm * ksqrtc_inv) * ws.cfr - ws.temp * (ksqrtc_two * ws.h1_vnorm - h0)).sum();
                    }
                } else {
                    res(i, j) += (ws.cf * h0).sum();
                    if (!d) continue;
                    res(i, Qtrial + j) += (ws.h1_vnorm * ws.cf + ksqrtc * h0 * ws.temp).sum();
                    if (d > 1)
                        res(i, 2 * Qtrial + j) += ((h0 * v_norm2 - ws.h1_vnorm * ksqrtc_inv) * ws.cf - ws.temp * (ksqrtc_two * ws.h1_vnorm - h0)).sum();
                }
            }
        }
    }
    if (!d) return;
    res.block(0, Qtrial, Qtrial, Qtrial) *= -sqrtc;
    if (d > 1)
        res.block(0, 2 * Qtrial, Qtrial, Qtrial) *= -c;
}

void GalerkinBuilder::hypersingular_general(size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept {
    size_t i, j, N = data.getGaussQROrder(), Qtrial = data.getQtrial();
    assert(res.rows() == Qtrial && res.cols() == Qtrial * (d + 1));
    compute_tdottp_s(ws);
#ifdef ENABLE_SPECIAL_CASES_FOR_PANEL_INTERACTION
    if (Qtrial == 2 && k_rp && !d) {
        if (ws.tdottp_zero) {
            res << (data.m_hyp_gen_fg_arc.block(0, 0, N, N) * ws.m_h0_s).sum(), (data.m_hyp_gen_fg_arc.block(0, N, N, N) * ws.m_h0_s).sum(),
                   (data.m_hyp_gen_fg_arc.block(N, 0, N, N) * ws.m_h0_s).sum(), (data.m_hyp_gen_fg_arc.block(N, N, N, N) * ws.m_h0_s).sum();
        } else {
            ws.tdottp_s *= k2ch.real();
            res << ((data.m_hyp_gen_fg_arc.block(0, 0, N, N) - data.m_hyp_gen_fg.block(0, 0, N, N) * ws.tdottp_s) * ws.m_h0_s).sum(),
                   ((data.m_hyp_gen_fg_arc.block(0, N, N, N) - data.m_hyp_gen_fg.block(0, N, N, N) * ws.tdottp_s) * ws.m_h0_s).sum(),
                   ((data.m_hyp_gen_fg_arc.block(N, 0, N, N) - data.m_hyp_gen_fg.block(N, 0, N, N) * ws.tdottp_s) * ws.m_h0_s).sum(),
                   ((data.m_hyp_gen_fg_arc.block(N, N, N, N) - data.m_hyp_gen_fg.block(N, N, N, N) * ws.tdottp_s) * ws.m_h0_s).sum();
        }
        return;
    }
#endif
    if (d > 0)
        ws.h1_vnorm = ws.m_h1_s * ws.m_v_norm_s;
    for (j = 0; j < Qtrial; ++j) {
        for (i = 0; i < Qtrial; ++i) {
            ws.cfr_s = data.m_hyp_gen_fg_arc.block(i * N, j * N, N, N);
            if (!ws.tdottp_zero) {
                ws.temp_s = data.m_hyp_gen_fg.block(i * N, j * N, N, N) * ws.tdottp_s;
                if (k_rp)
                    ws.cfr_s -= k2ch.real() * ws.temp_s;
                else ws.cf_s = ws.cfr_s - k2ch * ws.temp_s;
            }
            if (ws.tdottp_zero || k_rp) {
                res(i, j) = (ws.cfr_s * ws.m_h0_s).sum();
                if (!d) continue;
                if (ws.tdottp_zero)
                    res(i, Qtrial + j) = (ws.h1_vnorm * ws.cfr_s).sum();
                else res(i, Qtrial + j) = (ws.h1_vnorm * ws.cfr_s + ksqrtc * ws.m_h0_s * ws.temp_s).sum();
                if (d > 1) {
                    if (ws.tdottp_zero)
                        res(i, 2 * Qtrial + j) = ((ws.m_h0_s * ws.m_v_norm2_s - ws.h1_vnorm * ksqrtc_inv) * ws.cfr_s).sum();
                    else res(i, 2 * Qtrial + j) =
                        ((ws.m_h0_s * ws.m_v_norm2_s - ws.h1_vnorm * ksqrtc_inv) * ws.cfr_s - ws.temp_s * (ksqrtc_two * ws.h1_vnorm - ws.m_h0_s)).sum();
                }
            } else {
                res(i, j) = (ws.cf_s * ws.m_h0_s).sum();
                if (!d) continue;
                res(i, Qtrial + j) = (ws.h1_vnorm * ws.cf_s + ksqrtc * ws.m_h0_s * ws.temp_s).sum();
                if (d > 1)
                    res(i, 2 * Qtrial + j) =
                        ((ws.m_h0_s * ws.m_v_norm2_s - ws.h1_vnorm * ksqrtc_inv) * ws.cf_s - ws.temp_s * (ksqrtc_two * ws.h1_vnorm - ws.m_h0_s)).sum();
            }
        }
    }
    if (!d) return;
    res.block(0, Qtrial, Qtrial, Qtrial) *= -sqrtc;
    if (d > 1)
        res.block(0, 2 * Qtrial, Qtrial, Qtrial) *= -c;
}

void GalerkinBuilder::single_layer_coinciding(size_t der, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept {
    size_t i, j, K, l, N = data.getCGaussQROrder(), Qtest = data.getQtest();
    assert(res.rows() == Qtest && res.cols() == Qtest * (der + 1));
    const auto &h0 = ws.m_h0[0];
    if (panels_are_lines)
        ws.ttp_norm.setConstant(N, N, ws.m_tangent_norm[0](0, 0) * ws.m_tangent_p_norm[0](0, 0));
    else ws.ttp_norm = ws.m_tangent_norm[0] * ws.m_tangent_p_norm[0];
#ifdef ENABLE_SPECIAL_CASES_FOR_PANEL_INTERACTION
    if (Qtest == 2 && !der) {
        ws.m_1_hg = h0 * ws.ttp_norm;
        res(0, 0) = 2.0 * (data.m_slp_cnc_fg.block(0, 0, N, N) * ws.m_1_hg).sum();
        res(1, 1) = 2.0 * (data.m_slp_cnc_fg.block(N, N, N, N) * ws.m_1_hg).sum();
        res(1, 0) = res(0, 1) = ((data.m_slp_cnc_fg.block(N, 0, N, N) + data.m_slp_cnc_fg.block(0, N, N, N)) * ws.m_1_hg).sum();
        return;
    }
#endif
    res.setZero();
    if (der > 0)
        ws.m_2[0] = ws.m_h1[0] * ws.m_v_norm[0];
    if (der > 1)
        ws.m_3[0] = ws.m_2[0] * ksqrtc_inv - h0 * ws.m_v_norm2[0];
    for (K = 0; K < 2; ++K) {
        for (j = 0; j < Qtest; ++j) {
            for (i = 0; i < Qtest; ++i) {
                if (i < j) { /** symmetry */
                    for (l = 0; l <= der; ++l) res(i, l * Qtest + j) = res(j, i);
                    continue;
                }
                ws.cfr = ws.ttp_norm * data.m_slp_cnc_fg.block((K > 0 ? j : i) * N, (K > 0 ? i : j) * N, N, N);
                res(i, j) += (ws.cfr * h0).sum();
                if (!der) continue;
                res(i, Qtest + j) += (ws.cfr * ws.m_2[0]).sum();
                if (der > 1)
                    res(i, 2 * Qtest + j) += (ws.cfr * ws.m_3[0]).sum();
            }
        }
    }
    if (!der) return;
    res.block(0, Qtest, Qtest, Qtest) *= -sqrtc;
    if (der > 1)
        res.block(0, 2 * Qtest, Qtest, Qtest) *= c;
}

void GalerkinBuilder::single_layer_adjacent(bool swap, size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept {
    size_t i, j, K, N = data.getCGaussQROrder(), Qtest = data.getQtest();
    assert(res.rows() == Qtest && res.cols() == Qtest * (d + 1));
    res.setZero();
    for (K = 0; K < 2; ++K) {
        const auto &h0 = ws.m_h0[K];
        if (d > 0)
            ws.m_2[K] = ws.m_h1[K] * ws.m_v_norm[K];
        if (d > 1)
            ws.m_3[K] = ws.m_2[K] * ksqrtc_inv - h0 * ws.m_v_norm2[K];
        if (panels_are_lines)
            ws.ttp_norm.setConstant(N, N, ws.m_tangent_norm[panels_are_lines ? 0 : K](0, 0) * ws.m_tangent_p_norm[panels_are_lines ? 0 : K](0, 0));
        else ws.ttp_norm = ws.m_tangent_norm[K] * ws.m_tangent_p_norm[K];
        for (j = 0; j < Qtest; ++j) {
            for (i = 0; i < Qtest; ++i) {
                ws.cfr = ws.ttp_norm * (swap ? (K > 0 ? data.m_slp_adj_fg.block(j * N, i * N, N, N)
                                                      : data.m_slp_adj_fg_swap.block(i * N, j * N, N, N))
                                             : (K > 0 ? data.m_slp_adj_fg_swap.block(j * N, i * N, N, N)
                                                      : data.m_slp_adj_fg.block(i * N, j * N, N, N)));
                res(i, j) += (ws.cfr * h0).sum();
                if (!d) continue;
                res(i, Qtest + j) += (ws.cfr * ws.m_2[K]).sum();
                if (d > 1)
                    res(i, 2 * Qtest + j) += (ws.cfr * ws.m_3[K]).sum();
            }
        }
    }
    if (!d) return;
    res.block(0, Qtest, Qtest, Qtest) *= -sqrtc;
    if (d > 1)
        res.block(0, 2 * Qtest, Qtest, Qtest) *= c;
}

void GalerkinBuilder::single_layer_general(size_t d, workspace_t &ws, Eigen::Ref<Eigen::ArrayXXcd> res) const noexcept {
    size_t i, j, N = data.getGaussQROrder(), Ns = panels_are_lines ? 1 : N, Qtest = data.getQtest();
    assert(res.rows() == Qtest && res.cols() == Qtest * (d + 1));
    const auto tangent_p_norm_s = data.Derivative_01_tg_n.block(0, Ns * ws.j, Ns, Ns);
    const auto tangent_norm_s = data.Derivative_01_sg_n.block(0, Ns * ws.i, Ns, Ns);
#ifdef ENABLE_SPECIAL_CASES_FOR_PANEL_INTERACTION
    if (Qtest == 2 && !d) {
        if (panels_are_lines)
            ws.m_1_hg_s = ws.m_h0_s * (tangent_norm_s(0, 0) * tangent_p_norm_s(0, 0));
        else ws.m_1_hg_s = ws.m_h0_s * tangent_norm_s * tangent_p_norm_s;
        res << (data.m_slp_gen_fg.block(0, 0, N, N) * ws.m_1_hg_s).sum(), (data.m_slp_gen_fg.block(0, N, N, N) * ws.m_1_hg_s).sum(),
               (data.m_slp_gen_fg.block(N, 0, N, N) * ws.m_1_hg_s).sum(), (data.m_slp_gen_fg.block(N, N, N, N) * ws.m_1_hg_s).sum();
        return;
    }
#endif
    if (panels_are_lines)
        ws.ttp_norm_s.setConstant(N, N, tangent_norm_s(0, 0) * tangent_p_norm_s(0, 0));
    else ws.ttp_norm_s = tangent_norm_s * tangent_p_norm_s;
    if (d > 0)
        ws.m_2_s = ws.m_h1_s * ws.m_v_norm_s;
    if (d > 1)
        ws.m_3_s = ws.m_2_s * ksqrtc_inv - ws.m_h0_s * ws.m_v_norm2_s;
    for (j = 0; j < Qtest; ++j) {
        for (i = 0; i < Qtest; ++i) {
            ws.cfr_s = ws.ttp_norm_s * data.m_slp_gen_fg.block(i * N, j * N, N, N);
            res(i, j) = (ws.cfr_s * ws.m_h0_s).sum();
            if (!d) continue;
            res(i, Qtest + j) = (ws.cfr_s * ws.m_2_s).sum();
            if (d > 1)
                res(i, 2 * Qtest + j) = (ws.cfr_s * ws.m_3_s).sum();
        }
    }
    if (!d) return;
    res.block(0, Qtest, Qtest, Qtest) *= -sqrtc;
    if (d > 1)
        res.block(0, 2 * Qtest, Qtest, Qtest) *= c;
}

void GalerkinBuilder::compute_tdottp(size_t K, workspace_t &ws) const noexcept {
    double v;
    size_t N = data.getCGaussQROrder();
    if (panels_are_lines) {
        const auto &z = ws.m_tangent[K](0, 0), &zp = ws.m_tangent_p[K](0, 0);
        v = z.imag() * zp.imag() + z.real() * zp.real();
        ws.tdottp_zero = std::abs(v) < eps_mach;
    } else {
        ws.tdottp = ws.m_tangent[K].imag() * ws.m_tangent_p[K].imag() + ws.m_tangent[K].real() * ws.m_tangent_p[K].real();
        ws.tdottp_zero = isArrayZero(ws.tdottp);
    }
    if (!ws.tdottp_zero) {
        if (panels_are_lines)
            ws.tdottp.setConstant(N, N, 2. * v);
        else ws.tdottp *= 2.0;
    }
}

void GalerkinBuilder::compute_tdottp_s(workspace_t &ws) const noexcept {
    double v;
    size_t N = data.getGaussQROrder(), Ns = panels_are_lines ? 1 : N;
    const auto tangent_p_s = data.Derivative_01_tg.block(0, Ns * ws.j, Ns, Ns);
    const auto tangent_s = data.Derivative_01_sg.block(0, Ns * ws.i, Ns, Ns);
    if (panels_are_lines) {
        const auto &z = tangent_s(0, 0), &zp = tangent_p_s(0, 0);
        v = z.imag() * zp.imag() + z.real() * zp.real();
        ws.tdottp_zero = std::abs(v) < eps_mach;
    } else {
        ws.tdottp_s = tangent_s.imag() * tangent_p_s.imag() + tangent_s.real() * tangent_p_s.real();
        ws.tdottp_zero = isArrayZero(ws.tdottp_s);
    }
    if (!ws.tdottp_zero) {
        if (panels_are_lines)
            ws.tdottp_s.setConstant(N, N, 2. * v);
        else ws.tdottp_s *= 2.0;
    }
}

void GalerkinBuilder::panel_interaction(size_t d, workspace_t& ws, LayerType layers, bool comp_hyp_sl_ncoinc) noexcept {
    bool swap;
    if (ws.i == ws.j) {
        /** coinciding panels */
        compute_coinciding(ws);
        if (layers & LayerType::DOUBLE) double_layer_coinciding(d, ws, ws.interaction_matrix[LayerType::DOUBLE]);
        if (layers & LayerType::HYPERSINGULAR) hypersingular_coinciding(d, ws, ws.interaction_matrix[LayerType::HYPERSINGULAR]);
        if (layers & LayerType::SINGLE) single_layer_coinciding(d, ws, ws.interaction_matrix[LayerType::SINGLE]);
    } else if (data.is_adjacent(ws.i, ws.j, swap)) {
        /** adjacent panels */
        compute_adjacent(swap, ws);
        if (layers & LayerType::DOUBLE) double_layer_adjacent(swap, d, ws, ws.interaction_matrix[LayerType::DOUBLE]);
        if (comp_hyp_sl_ncoinc) {
            if (layers & LayerType::HYPERSINGULAR) hypersingular_adjacent(swap, d, ws, ws.interaction_matrix[LayerType::HYPERSINGULAR]);
            if (layers & LayerType::SINGLE) single_layer_adjacent(swap, d, ws, ws.interaction_matrix[LayerType::SINGLE]);
        }
    } else {
        /** disjoint panels */
        compute_general(ws);
        if (layers & LayerType::DOUBLE) double_layer_general(d, ws, ws.interaction_matrix[LayerType::DOUBLE]);
        if (comp_hyp_sl_ncoinc) {
            if (layers & LayerType::HYPERSINGULAR) hypersingular_general(d, ws, ws.interaction_matrix[LayerType::HYPERSINGULAR]);
            if (layers & LayerType::SINGLE) single_layer_general(d, ws, ws.interaction_matrix[LayerType::SINGLE]);
        }
    }
}

void GalerkinBuilder::initialize_constants(const std::complex<double>& k_in, double c_in) {
    /** ensure that refraction index is valid (it is assumed that c_out = 1.0) */
    if (c_in < 1.)
        throw std::runtime_error("Refraction index must not be smaller than 1");
    /** ensure that wavenumber k is valid (i.e. that we have a unique solution),
      * see http://dx.doi.org/10.1016/j.apnum.2011.05.003, Theorem 2.1, p. 1019 */
    if (k_in.imag() < 0.0)
        throw std::runtime_error("Wavenumber does not meet conditions for uniqueness of the solution (Im(k) >= 0)");
    k = k_in;
    c = c_in;
    sqrtc = sqrt(c);
    ksqrtc = k * sqrtc;
    ksqrtc_inv = 1.0 / ksqrtc;
    ksqrtc_two = 2.0 * ksqrtc;
    ksqrtca = std::abs(ksqrtc);
    k2ch = ksqrtc * ksqrtc * .5;
    k_rp = k.imag() == 0 && k.real() > 0;
}

void GalerkinBuilder::assembleDense(const std::complex<double>& k_in, double c_in, LayerType layers, size_t d,
                                    size_t row, size_t col, size_t rows, size_t cols) {
    size_t s, npanels = data.getNumPanels();
    if (rows == 0) rows = npanels;
    if (cols == 0) cols = npanels;
    bool is_full = rows == npanels && cols == npanels;
    size_t dim_test = data.getTestSpaceDimension(rows), dim_trial = data.getTrialSpaceDimension(cols);
    size_t m0 = data.getTestSpaceDimension(row), n0 = data.getTrialSpaceDimension(col);
    size_t q_test = data.getQtest(), q_trial = data.getQtrial();
    size_t diag_start = std::max(row, col), diag_end = std::min(row + rows - 1, col + cols - 1);
    size_t ndiag = diag_end - diag_start + 1;
    bool is_square = data.testTrialSpacesAreEqual();
    /** indices of panel pairs */
    IndexPair inits[] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    std::vector<IndexPair> ind;
    ind.reserve(1 + (rows * cols - (is_square ? (ndiag * (ndiag - 1)) / 2 : 0)) / 4);
    /** initialization */
    initialize_constants(k_in, c_in);
    std::map<LayerType, IndexPair> layer_list;
    if (layers & LayerType::DOUBLE) {
        layer_list[LayerType::DOUBLE] = {q_test, (is_square ? 2 : 1) * (d + 1) * q_trial};
        for (size_t t = 0; t <= d; ++t) dense_layer_matrix[LayerType::DOUBLE][t].setZero(dim_test, dim_trial);
    } else is_square = true;
    if (layers & LayerType::HYPERSINGULAR) {
        layer_list[LayerType::HYPERSINGULAR] = {q_trial, (d + 1) * q_trial};
        for (size_t t = 0; t <= d; ++t) dense_layer_matrix[LayerType::HYPERSINGULAR][t].setZero(dim_trial, dim_trial);
    }
    if (layers & LayerType::SINGLE) {
        layer_list[LayerType::SINGLE] = {q_test, (d + 1) * q_test};
        for (size_t t = 0; t <= d; ++t) dense_layer_matrix[LayerType::SINGLE][t].setZero(dim_test, dim_test);
    }
    auto is_within_submatrix = [&](size_t &m, size_t &n) {
        if (is_full) return true;
        bool ret = m >= m0 && n >= n0 && m < m0 + dim_test && n < n0 + dim_trial;
        if (ret) { m -= m0, n -= n0; }
        return ret;
    };
    auto is_required = [&](size_t i, size_t j) {
        return i < diag_start || j < diag_start || i > diag_end || j > diag_end || j <= i;
    };
#ifndef PARALLELIZE_BUILDER
    /** storage */
    workspace_t ws;
    for (auto layer : layer_list)
        ws.interaction_matrix[layer.first].resize(layer.second.first, layer.second.second);
#endif
#ifdef PARALLELIZE_BUILDER
    /** work with indices split in four parts to avoid simultaneous writing within the for_each loop */
    for (s = 0; s < 4; ++s) {
        ind.clear();
        for (size_t i = row + inits[s].first; i < row + rows; i += 2) {
            for (size_t j = col + inits[s].second; j < col + cols; j += 2) {
#else
        for (size_t i = row; i < row + rows; ++i) {
            for (size_t j = col; j < col + cols; ++j) {
#endif
                if (!is_square)
                    ind.push_back({i, j});
                else if (is_required(i, j))
                    ind.push_back({std::max(i, j), std::min(i, j)});
            }
        }
        std::for_each(EXEC_POLICY ind.cbegin(), ind.cend(), [&](const auto &index) {
            size_t i = index.first, j = index.second, k, l, m, n, step = i != j && is_square ? 2 : 1;
#ifdef PARALLELIZE_BUILDER
            /** storage */
            workspace_t ws;
            for (auto layer_info : layer_list)
                ws.interaction_matrix[layer_info.first].resize(layer_info.second.first, layer_info.second.second);
#endif
            ws.i = i, ws.j = j, ws.compute_transposed = is_square;
            /** compute interaction matrices for panel pair (i, j) */
            panel_interaction(d, ws, layers, is_square || is_required(i, j));
            /** local to global mapping of the elements in interaction matrices */
            for (auto layer_info : layer_list) {
                auto layer = layer_info.first;
                size_t nr = layer == LayerType::HYPERSINGULAR ? q_trial : q_test, nc = layer == LayerType::SINGLE ? q_test : q_trial;
                bool ld = layer == LayerType::DOUBLE, write_transposed = i != j && (!ld || is_square);
                size_t alpha = ld ? step : 1, offset = ld ? 1 : 0;
                for (l = 0; l < nc; ++l) {
                    for (k = 0; k < nr; ++k) {
                        std::tie(m, n) = map_local_to_global({i, j}, {k, l}, layer);
                        if (is_within_submatrix(m, n)) for (size_t t = 0; t <= d; ++t)
                            dense_layer_matrix[layer][t](m, n) += ws.interaction_matrix[layer](k, t * alpha * nc + l);
                        if (write_transposed) {
                            std::tie(m, n) = map_local_to_global({j, i}, {k, l}, layer);
                            if (is_within_submatrix(m, n)) for (size_t t = 0; t <= d; ++t)
                                dense_layer_matrix[layer][t](m, n) += ws.interaction_matrix[layer](ld ? k : l, (t * alpha + offset) * nc + (ld ? l : k));
                        }
                    }
                }
            }
        });
#ifdef PARALLELIZE_BUILDER
    }
#endif
    /** Apply the factor i/4 from the Helmholtz 2D kernel function */
    for (auto layer_info : layer_list)
        for (size_t t = 0; t <= d; ++t)
            dense_layer_matrix[layer_info.first][t] *= 0.25i;
}

void GalerkinBuilder::assembleSparse(const std::complex<double> &k_in, double c_in,
                                     std::map<LayerType, Eigen::SparseMatrix<std::complex<double> > > &layers,
                                     size_t row, size_t col, size_t rows, size_t cols,
                                     std::vector<size_t> &ai, std::vector<size_t> &aj,
                                     size_t i, size_t j,
                                     workspace_t &ws) {
    assert(data.testTrialSpacesAreEqual()); /** only square BIOs are supported */
    auto is_available = [](const std::vector<size_t> &a, size_t b) {
        auto it = std::lower_bound(a.cbegin(), a.cend(), b);
        return it != a.end() && *it == b;
    };
    size_t q_test = data.getQtest(), q_trial = data.getQtrial();
    auto map_interaction_matrix = [&](size_t p, size_t q) {
        ws.i = p, ws.j = q;
        if (p == q) {
            compute_coinciding(ws);
        }
    };
    /** get row */
    if (i == 0 || i + 1 == rows || is_available(ai, i - 1) || is_available(ai, i + 1)) {

    }
    /** get column */
    if (j == 0 || j + 1 == cols || is_available(aj, j - 1) || is_available(aj, j + 1)) {

    }
    size_t npanels = data.getNumPanels();

}

const Eigen::MatrixXcd &GalerkinBuilder::get_dense_layer_matrix_safe(LayerType layer, size_t d) const {
    if (d > 2)
        throw std::runtime_error("Invalid order of derivative");
    if (dense_layer_matrix.find(layer) == dense_layer_matrix.end())
        throw std::runtime_error("Requested layer is not computed, so cannot be retrieved");
    return dense_layer_matrix.at(layer)[d];
}

const Eigen::MatrixXcd & GalerkinBuilder::getDoubleLayer(size_t d) const {
    return get_dense_layer_matrix_safe(LayerType::DOUBLE, d);
}

const Eigen::MatrixXcd & GalerkinBuilder::getHypersingular(size_t d) const {
    return get_dense_layer_matrix_safe(LayerType::HYPERSINGULAR, d);
}

const Eigen::MatrixXcd & GalerkinBuilder::getSingleLayer(size_t d) const {
    return get_dense_layer_matrix_safe(LayerType::SINGLE, d);
}
