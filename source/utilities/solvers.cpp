#include <iostream>
#include <execution>
#include <mutex>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include "solvers.hpp"
#include "mass_matrix.hpp"
#include "continuous_space.hpp"
#include "discontinuous_space.hpp"
#include "parametrized_mesh.hpp"
#include "parametrized_line.hpp"
#include "cbessel.hpp"
#include "cspline.hpp"
#include "scatterer.hpp"

#define USE_LAPACK_SOLVER 1
#ifdef USE_LAPACK_SOLVER
#include "lapacke.h"
#include "complex.h"
#endif

typedef std::complex<double> complex_t;

template<int p = Eigen::Dynamic>
Eigen::Matrix<complex_t,-1,p> complex_LU_solve(Eigen::MatrixXcd &A, Eigen::Matrix<complex_t,-1,p> &B, int *ws = NULL) {
#ifdef USE_LAPACK_SOLVER
            lapack_complex_double *a, *b;
            lapack_int n = A.rows(), *ipiv = ws == NULL ? new lapack_int[n] : ws;
            lapack_int info, lda = n, ldb = n, nrhs = p < 0 ? B.cols() : p;
            a = reinterpret_cast<lapack_complex_double*>(A.data());
            b = reinterpret_cast<lapack_complex_double*>(B.data());
            LAPACK_zgesv(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
            if (ws == NULL)
                delete[] ipiv;
            assert(info >= 0);
            if (info > 0)
                throw std::runtime_error("zgesv: LU is singular");
            Eigen::Matrix<complex_t,-1,p> res;
            res.resize(n, nrhs);
            for (unsigned j = 0; j < nrhs; ++j) for (unsigned i = 0; i < n; ++i) {
                auto bij = b[j * n + i];
                res(i, j) = complex_t(creal(bij), cimag(bij));
            }
            return res;
#else
            Eigen::PartialPivLU<Eigen::Ref<Eigen::MatrixXcd> > lu(A);
            return lu.solve(rhs);
#endif
}

namespace bvp {
    namespace direct_first_kind {
        Eigen::VectorXcd solve_dirichlet(const ParametrizedMesh &mesh,
                                         const std::function<complex_t(double, double)> u_dir,
                                         const unsigned order,
                                         const double k) {
            QuadRule GaussQR = getGaussQR(order, 0., 1.);
            // compute FEM-spaces of lowest order
            ContinuousSpace<1> cont_space;
            DiscontinuousSpace<0> discont_space;
            // compute interpolation coefficients for dirichlet data
            Eigen::VectorXcd u_dir_N = discont_space.Interpolate_helmholtz(u_dir, mesh);
            // compute operators for first kind direct Dirichlet problem BIE
            Eigen::MatrixXcd M = mass_matrix::GalerkinMatrix(mesh, discont_space, cont_space, GaussQR);
            BuilderData builder_data(mesh, cont_space, discont_space, order);
            GalerkinBuilder builder(builder_data);
            builder.assembleDense(k, 1., LayerType::DOUBLE | LayerType::SINGLE);
            Eigen::MatrixXcd K = builder.getDoubleLayer();
            Eigen::MatrixXcd V = builder.getSingleLayer();
            // build rhs for solving
            Eigen::VectorXcd rhs = (0.5*M-K)*u_dir_N;
            // solve for coefficients
            Eigen::HouseholderQR<Eigen::MatrixXcd> dec(-V);
            Eigen::VectorXcd sol = dec.solve(rhs);
            return sol;
        }

        Eigen::VectorXcd solve_neumann(const ParametrizedMesh &mesh,
                                       const std::function<complex_t(double, double)> u_neu,
                                       const unsigned int order,
                                       const double k) {
            QuadRule GaussQR = getGaussQR(order, 0., 1.);
            // compute FEM-spaces of lowest order
            ContinuousSpace<1> cont_space;
            DiscontinuousSpace<0> discont_space;
            // compute interpolation coefficients of Neumann data
            Eigen::VectorXcd u_neu_N = discont_space.Interpolate_helmholtz(u_neu, mesh);
            // compute operators for first kind direct Neumann problem BIE
            Eigen::MatrixXcd M = mass_matrix::GalerkinMatrix(mesh, cont_space, discont_space, GaussQR);
            BuilderData builder_data(mesh, discont_space, cont_space, order);
            GalerkinBuilder builder(builder_data);
            builder.assembleDense(k, 1., LayerType::DOUBLE | LayerType::HYPERSINGULAR);
            Eigen::MatrixXcd K = builder.getDoubleLayer();
            Eigen::MatrixXcd W = builder.getHypersingular();
            // build rhs for solving
            Eigen::VectorXcd rhs = (0.5*M+K.transpose())*u_neu_N;
            // solve for coefficients
            Eigen::HouseholderQR<Eigen::MatrixXcd> dec(-W);
            Eigen::VectorXcd sol = dec.solve(rhs);
            return sol;
        }
    } // namespace direct_first_kind
} // namespace bvp
namespace tp {
    namespace direct_second_kind {
        void A_rhs(const ParametrizedMesh &mesh,
                   const std::function<complex_t(double, double)> u_inc,
                   const std::function<Eigen::Vector2cd (double, double)> u_inc_del,
                   const std::complex<double> &k,
                   const double c_o,
                   const double c_i,
                   SolutionsOperator &sol_op,
                   Eigen::MatrixXcd &A,
                   Eigen::VectorXcd &rhs) {
            sol_op.gen_sol_op(k, c_o, c_i, A);
            Eigen::MatrixXcd M_cont = sol_op.mass_matrix();
            /** build operator for right hand side */
            const auto &K_o = sol_op.K_ext(), &V_o = sol_op.V_ext(), &W_o = sol_op.W_ext();
            Eigen::MatrixXcd A_o(K_o.rows() + W_o.rows(), K_o.cols() + V_o.cols());
            A_o.block(0, 0, K_o.rows(), K_o.cols()) = -K_o + 0.5*M_cont;
            A_o.block(0, K_o.cols(), V_o.rows(), V_o.cols()) = V_o;
            A_o.block(K_o.rows(), 0, W_o.rows(), W_o.cols()) = W_o;
            A_o.block(K_o.rows(), K_o.cols(), K_o.cols(), K_o.rows()) = K_o.transpose()+0.5*M_cont;
            /** build vector from incoming wave data for right hand side */
            const auto &bd = sol_op.getBuilderData();
            Eigen::VectorXcd u_inc_N(bd.getTestSpaceDimension() + bd.getTrialSpaceDimension());
            u_inc_N << bd.test_space.Interpolate_helmholtz(u_inc, mesh),
                       bd.trial_space.Interpolate_helmholtz_neu(u_inc_del, mesh);
            /** compute the right hand side */
            rhs = A_o * u_inc_N;
        }

        Eigen::VectorXcd solve(const ParametrizedMesh &mesh,
                               const std::function<complex_t(double, double)> u_inc,
                               const std::function<Eigen::Vector2cd (double, double)> u_inc_del,
                               const unsigned order,
                               const std::complex<double> &k,
                               const double c_o,
                               const double c_i) {
            ContinuousSpace<1> test_space;
            BuilderData builder_data(mesh, test_space, test_space, order);
            auto sol_op = SolutionsOperator(builder_data, false);
            Eigen::MatrixXcd so;
            Eigen::VectorXcd rhs;
            A_rhs(mesh, u_inc, u_inc_del, k, c_o, c_i, sol_op, so, rhs);
            if (!rhs.real().allFinite() || !rhs.imag().allFinite())
                throw std::runtime_error("Failed to solve for traces, right-hand side is undefined");
            if (!so.real().allFinite() || !so.imag().allFinite())
                throw std::runtime_error("Failed to solve for traces, solutions operator is undefined");
            // Solving for coefficients
            return complex_LU_solve<1>(so, rhs);
        }

        void in_solve(const ParametrizedMesh &mesh,
                      const std::function<std::complex<double> (double, double)> u_inc,
                      const std::function<Eigen::Vector2cd (double, double)> u_inc_del,
                      const ComplexSpline &trace_dir,
                      const ComplexSpline &trace_neu,
                      const unsigned order,
                      const std::complex<double>& k,
                      const double c_o,
                      const double c_i,
                      const Eigen::Vector2d &lower_left_corner,
                      const Eigen::Vector2d &upper_right_corner,
                      const unsigned grid_size_x,
                      const unsigned grid_size_y,
                      bool add_u_inc,
                      Eigen::ArrayXXd &grid_X,
                      Eigen::ArrayXXd &grid_Y,
                      Eigen::ArrayXXcd &S) {
            S.resize(grid_size_x, grid_size_y);
            grid_X.resize(grid_size_x, grid_size_y);
            grid_Y.resize(grid_size_x, grid_size_y);
            QuadRule qr = getGaussQR(order, 0., 1.);
            size_t n = qr.n;
            Eigen::ArrayXXcd Y, Tr_i_D, Tr_i_N, Tr_o_D, Tr_o_N, N;
            Eigen::ArrayXXd D;
            unsigned npanels = mesh.getNumPanels();
            Y.resize(npanels, n);
            Tr_i_D.resize(npanels, n);
            Tr_i_N.resize(npanels, n);
            Tr_o_D.resize(npanels, n);
            Tr_o_N.resize(npanels, n);
            D.resize(npanels, n);
            N.resize(npanels, n);
            double t0 = 0., t1, width = upper_right_corner(0) - lower_left_corner(0);
            double height = upper_right_corner(1) - lower_left_corner(1);
            for (unsigned i = 0; i < npanels; ++i) {
                const auto &p = *mesh.getPanels()[i];
                double plen = p.length();
                for (unsigned j = 0; j < n; ++j) {
                    Eigen::Vector2d y = p[qr.x(j)], tangent = p.Derivative_01(qr.x(j)), normal_o;
                    Y(i, j) = -y(0) - 1i * y(1);
                    D(i, j) = tangent.norm() * qr.w(j);
                    // normal vector
                    normal_o << tangent(1), -tangent(0);
                    normal_o.normalize();
                    N(i, j) = normal_o(0) + 1i * normal_o(1);
                    t1 = t0 + qr.x(j) * plen;
                    Tr_i_D(i, j) = trace_dir.eval(t1);
                    Tr_i_N(i, j) = trace_neu.eval(t1);
                    Tr_o_D(i, j) = Tr_i_D(i, j) - u_inc(y(0), y(1));
                    Tr_o_N(i, j) = Tr_i_N(i, j) - normal_o.dot(u_inc_del(y(0), y(1)));
                }
                t0 += plen;
            }
            // compute solution using the Green second identity
            double x_step = width / (grid_size_x - 1.);
            double y_step = height / (grid_size_y - 1.);
            complex_t k_sqrt_ci = k * std::sqrt(c_i), k_sqrt_co = k * std::sqrt(c_o);
            bool k_rp = k.imag() == 0 && k.real() > 0;
            unsigned done = 0, progress, old_progress = 0;
            std::vector<std::pair<size_t,size_t> > ind(npanels * n), array_ind(grid_size_x * grid_size_y);
            std::iota(ind.begin(), ind.end(), PairInc<size_t>(0, 0, npanels));
            std::iota(array_ind.begin(), array_ind.end(), PairInc<size_t>(0, 0, grid_size_y));
            std::vector<size_t> indp(npanels);
            std::iota(indp.begin(), indp.end(), 0);
            gsl_set_error_handler_off();
            auto arr0 = Eigen::ArrayXXcd::Zero(npanels, n);
            std::mutex mtx;
            std::transform(std::execution::par, array_ind.cbegin(), array_ind.cend(), S.data(), [&](const auto &II_JJ) {
                size_t II = II_JJ.first, JJ = II_JJ.second;
                complex_t retval;
                {
                    std::unique_lock lock(mtx);
                    progress = (100 * (++done)) / array_ind.size();
#ifdef CMDL
                    if (progress > old_progress) {
                        auto pstr = std::to_string(progress);
                        std::cout << "\rLifting solution from traces... " << pstr << "%" << std::string(3 - pstr.length(), ' ');
                        std::flush(std::cout);
                        old_progress = progress;
                    }
#endif
                }
                Eigen::Vector2d x;
                x << lower_left_corner(0) + II * x_step, lower_left_corner(1) + JJ * y_step;
                grid_X(II, JJ) = x(0);
                grid_Y(II, JJ) = x(1);
                bool inside = scatterer::inside_mesh(mesh, x);
                complex_t scale = (inside ? 1. : -1.) * 0.25i, kappa = inside ? k_sqrt_ci : k_sqrt_co, excess(0, 0);
                std::map<size_t,double> np;
                for (size_t i : indp) {
                    const auto &panel = *mesh.getPanels()[i];
                    double t, d = scatterer::panel_distance(panel, x, &t);
                    if (d < 1e2 * std::numeric_limits<double>::epsilon()) /** x is on the boundary */
                        return trace_dir.eval((i > 0 ? mesh.getCSum(i - 1) : 0.) + t * panel.length());
                    if (d * std::abs(kappa) < 1e-2)
                        np[i] = t;
                }
                Eigen::ArrayXXcd Z, H0, H1;
                Eigen::ArrayXXd X;
                H0.resize(npanels, n);
                H1.resize(npanels, n);
                auto kernel = complex_bessel::Hankel1Real01(npanels, n);
                Z = Y + x(0) + 1i * x(1);
                X = Z.cwiseAbs();
                if (k_rp)
                    kernel.h1_01(kappa.real() * X, H0, H1);
                else for_each(std::execution::seq, ind.cbegin(), ind.cend(), [&](const auto &ij) {
                    size_t ii = ij.first, jj = ij.second;
                    H0(ii, jj) = complex_bessel::HankelH1(0, kappa * X(ii, jj));
                    H1(ii, jj) = complex_bessel::HankelH1(1, kappa * X(ii, jj));
                });
                Eigen::ArrayXXcd P = (X > 0).select(H0 * (inside ? Tr_i_N : Tr_o_N) -
                    kappa * H1 * (inside ? Tr_i_D : Tr_o_D) * (Z.real() * N.real() + Z.imag() * N.imag()) / X, arr0) * D;
                /** improve integration accuracy for near panels */
                for (const auto &npi : np) {
                    size_t i = npi.first;
                    double t_min = npi.second;
                    const auto &panel = *mesh.getPanels()[i];
                    std::function<complex_t(double)> func = [&](double t) {
                        Eigen::Vector2d y = panel[t], tangent = panel.Derivative_01(t), normal_o;
                        double d = (x - y).norm(), len = (i > 0 ? mesh.getCSum(i - 1) : 0.) + t * panel.length();
                        if (d == 0) return complex_t(0, 0);
                        complex_t h0, h1, tD, tN;
                        complex_bessel::H1_01(kappa * d, h0, h1, k_rp);
                        tD = trace_dir.eval(len);
                        tN = trace_neu.eval(len);
                        normal_o << tangent(1), -tangent(0);
                        normal_o.normalize();
                        if (!inside) {
                            tD -= u_inc(y(0), y(1));
                            tN -= normal_o.dot(u_inc_del(y(0), y(1)));
                        }
                        return (h0 * tN - kappa * h1 * tD * normal_o.dot(x - y) / d) * tangent.norm();
                    };
                    size_t lim = 100;
                    auto w = gsl_integration_workspace_alloc(lim);
                    double error, res_real, res_imag, gsl_tol = 1e-8;
                    int rc_real, rc_imag;
                    gsl_function Freal = {
                        [](double d, void* vf) -> double {
                            auto& f = *static_cast<std::function<complex_t(double)>*>(vf);
                            return f(d).real();
                        },
                        &func
                    };
                    gsl_function Fimag = {
                        [](double d, void* vf) -> double {
                            auto& f = *static_cast<std::function<complex_t(double)>*>(vf);
                            return f(d).imag();
                        },
                        &func
                    };
                    double pts[3] = {0, t_min, 1};
                    rc_real = gsl_integration_qagp(&Freal, pts, 3, 0, gsl_tol, lim, w, &res_real, &error);
                    rc_imag = gsl_integration_qagp(&Fimag, pts, 3, 0, gsl_tol, lim, w, &res_imag, &error);
#ifdef CMDL
                    if (rc_real || rc_imag)
                        std::cout << "Warning: QAGP failed to converge, last value: " << complex_t(res_real, res_imag) << std::endl;
#endif
                    complex_t old_value = P.row(i).sum(), improved_value = complex_t(res_real, res_imag);
                    gsl_integration_workspace_free(w);
                    excess += improved_value - old_value;
                }
                retval = scale * (P.sum() + excess);
                if (add_u_inc && !inside)
                    retval += u_inc(x(0), x(1));
                return retval;
            });
#ifdef CMDL
            std::cout << std::endl;
#endif
        }

        Eigen::ArrayXXcd solve_in_rectangle(const ParametrizedMesh& mesh,
                                            const std::function<std::complex<double> (double, double)> u_inc,
                                            const std::function<Eigen::Vector2cd (double, double)> u_inc_del,
                                            const unsigned order,
                                            const std::complex<double>& k,
                                            const double c_o,
                                            const double c_i,
                                            const Eigen::Vector2d& lower_left_corner,
                                            const Eigen::Vector2d& upper_right_corner,
                                            const unsigned grid_size_x,
                                            const unsigned grid_size_y,
                                            Eigen::ArrayXXd &grid_X,
                                            Eigen::ArrayXXd &grid_Y,
                                            bool total_field,
                                            SolutionsOperator &sol_op,
                                            Eigen::MatrixXcd &so) {
#ifdef CMDL
            std::cout << "Computing boundary traces..." << std::endl;
#endif
            Eigen::VectorXcd rhs;
            A_rhs(mesh, u_inc, u_inc_del, k, c_o, c_i, sol_op, so, rhs);
            // Solving for coefficients
            Eigen::VectorXcd traces = complex_LU_solve<1>(so, rhs);
            unsigned numpanels = mesh.getNumPanels();
            ComplexSpline trace_dir(mesh, traces.head(numpanels));
            ComplexSpline trace_neu(mesh, traces.tail(numpanels));
            Eigen::ArrayXXcd S;
            in_solve(mesh, u_inc, u_inc_del, trace_dir, trace_neu, order, k, c_o, c_i,
                     lower_left_corner, upper_right_corner, grid_size_x, grid_size_y, total_field, grid_X, grid_Y, S);
            return S;
        }

        Eigen::ArrayXXcd solve_in_rectangle(const ParametrizedMesh& mesh,
                                            const std::function<std::complex<double> (double, double)> u_inc,
                                            const std::function<Eigen::Vector2cd (double, double)> u_inc_del,
                                            const unsigned order_galerkin,
                                            const unsigned order_green,
                                            const std::complex<double>& k,
                                            const double c_o,
                                            const double c_i,
                                            const Eigen::Vector2d& lower_left_corner,
                                            const Eigen::Vector2d& upper_right_corner,
                                            const unsigned grid_size_x,
                                            const unsigned grid_size_y,
                                            Eigen::ArrayXXd &grid_X,
                                            Eigen::ArrayXXd &grid_Y,
                                            bool total_field) {
            unsigned numpanels = mesh.getNumPanels();
#ifdef CMDL
            std::cout << "Computing boundary traces..." << std::endl;
#endif
            auto traces = solve(mesh, u_inc, u_inc_del, order_galerkin, k, c_o, c_i);
            if (!traces.real().allFinite() || !traces.imag().allFinite())
                throw std::runtime_error("Traces are undefined, aborting solution lifting");
            ComplexSpline trace_dir(mesh, traces.head(numpanels));
            ComplexSpline trace_neu(mesh, traces.tail(numpanels));
            Eigen::ArrayXXcd S;
            in_solve(mesh, u_inc, u_inc_del, trace_dir, trace_neu, order_green, k, c_o, c_i,
                     lower_left_corner, upper_right_corner, grid_size_x, grid_size_y, total_field, grid_X, grid_Y, S);
            return S;
        }

    } // namespace direct_second_kind

} // namespace tsp
