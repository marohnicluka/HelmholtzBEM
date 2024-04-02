#include <iostream>
#include "gen_sol_op.hpp"
#include "solvers.hpp"
#include "mass_matrix.hpp"
#include "continuous_space.hpp"
#include "discontinuous_space.hpp"
#include "parametrized_mesh.hpp"
#include "parametrized_line.hpp"
#include "cbessel.hpp"
#include "cspline.hpp"
#include "scatterer.hpp"

typedef std::complex<double> complex_t;
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
            GalerkinMatrixBuilder builder(builder_data);
            builder.assembleDoubleLayer(k, 1.);
            Eigen::MatrixXcd K = builder.getDoubleLayer();
            builder.assembleSingleLayer(k, 1.);
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
            GalerkinMatrixBuilder builder(builder_data);
            builder.assembleDoubleLayer(k, 1.);
            Eigen::MatrixXcd K = builder.getDoubleLayer();
            builder.assembleHypersingular(k, 1.);
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
        Eigen::VectorXcd in_traces(const ParametrizedMesh &mesh,
                                   const std::function<complex_t(double, double)> u_inc,
                                   const std::function<Eigen::Vector2cd (double, double)> u_inc_del,
                                   const std::complex<double> &k,
                                   const double c_o,
                                   const double c_i,
                                   GalerkinMatrixBuilder &builder,
                                   Eigen::MatrixXcd &A) {
            builder.assembleAll(k, c_o);
            Eigen::MatrixXcd K_o = builder.getDoubleLayer();
            Eigen::MatrixXcd W_o = builder.getHypersingular();
            Eigen::MatrixXcd V_o = builder.getSingleLayer();
            builder.assembleAll(k, c_i);
            Eigen::MatrixXcd K_i = builder.getDoubleLayer();
            Eigen::MatrixXcd W_i = builder.getHypersingular();
            Eigen::MatrixXcd V_i = builder.getSingleLayer();
            // Build matrices for solving linear system of equations
            const auto &bd = builder.getData();
            Eigen::MatrixXcd M_cont = mass_matrix::GalerkinMatrix(mesh, bd.test_space, bd.trial_space, bd.GaussQR);
            A.resize(K_o.rows() + W_o.rows(), K_o.cols() + V_o.cols());
            A.block(0, 0, K_o.rows(), K_o.cols()) = (-K_o + K_i)+M_cont;
            A.block(0, K_o.cols(), V_o.rows(), V_o.cols()) = (V_o-V_i);
            A.block(K_o.rows(), 0, W_o.rows(), W_o.cols()) = W_o-W_i;
            A.block(K_o.rows(), K_o.cols(), K_o.cols(), K_o.rows()) =
                    (K_o-K_i).transpose()+M_cont;
            // Build matrices for right hand side
            Eigen::MatrixXcd A_o(K_o.rows() + W_o.rows(), K_o.cols() + V_o.cols());
            A_o.block(0, 0, K_o.rows(), K_o.cols()) = -K_o + 0.5*M_cont;
            A_o.block(0, K_o.cols(), V_o.rows(), V_o.cols()) = V_o;
            A_o.block(K_o.rows(), 0, W_o.rows(), W_o.cols()) = W_o;
            A_o.block(K_o.rows(), K_o.cols(), K_o.cols(), K_o.rows()) =
                    K_o.transpose()+0.5*M_cont;
            // Build vectors from incoming wave data for right hand side
            Eigen::VectorXcd u_inc_dir_N = bd.test_space.Interpolate_helmholtz(u_inc, mesh);
            Eigen::VectorXcd u_inc_neu_N = bd.trial_space.Interpolate_helmholtz_neu(u_inc_del, mesh);
            Eigen::VectorXcd u_inc_N(bd.getTestSpaceDimension() + bd.getTrialSpaceDimension());
            u_inc_N << u_inc_dir_N, u_inc_neu_N;
            // compute right hand side
            Eigen::VectorXcd rhs = (A_o * u_inc_N);
            // Solving for coefficients
            Eigen::HouseholderQR<Eigen::MatrixXcd> dec(A);
            return dec.solve(rhs);
        }

        Eigen::VectorXcd solve(const ParametrizedMesh &mesh,
                               const std::function<complex_t(double, double)> u_inc,
                               const std::function<Eigen::Vector2cd (double, double)> u_inc_del,
                               const unsigned order,
                               const std::complex<double> &k,
                               const double c_o,
                               const double c_i) {
            QuadRule GaussQR = getGaussQR(order, 0., 1.);
            ContinuousSpace<1> cont_space;
            BuilderData builder_data(mesh, cont_space, cont_space, order);
            GalerkinMatrixBuilder builder(builder_data);
            Eigen::MatrixXcd so;
            return in_traces(mesh, u_inc, u_inc_del, k, c_o, c_i, builder, so);
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
            Eigen::ArrayXXcd Y, Z, H0, H1, Trace_i_D, Trace_i_N, Trace_o_D, Trace_o_N, N, G;
            Eigen::ArrayXXd X, D;
            unsigned numpanels = mesh.getNumPanels();
            X.resize(numpanels, n);
            Y.resize(numpanels, n);
            Z.resize(numpanels, n);
            H0.resize(numpanels, n);
            H1.resize(numpanels, n);
            Trace_i_D.resize(numpanels, n);
            Trace_i_N.resize(numpanels, n);
            Trace_o_D.resize(numpanels, n);
            Trace_o_N.resize(numpanels, n);
            D.resize(numpanels, n);
            N.resize(numpanels, n);
            G.resize(numpanels, n);
            double t0 = 0., t1;
            for (unsigned i = 0; i < numpanels; ++i) {
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
                    Trace_i_D(i, j) = trace_dir.eval(t1);
                    Trace_i_N(i, j) = trace_neu.eval(t1);
                    Trace_o_D(i, j) = Trace_i_D(i, j) - u_inc(y(0), y(1));
                    Trace_o_N(i, j) = Trace_i_N(i, j) - normal_o.dot(u_inc_del(y(0), y(1)));
                }
                t0 += plen;
            }
            // compute solution using the Green second identity
            double x_step = (upper_right_corner(0) - lower_left_corner(0)) / (grid_size_x - 1.);
            double y_step = (upper_right_corner(1) - lower_left_corner(1)) / (grid_size_y - 1.);
            double excess;
            complex_t k_sqrt_ci = k * std::sqrt(c_i), k_sqrt_co = k * std::sqrt(c_o), kappa;
            unsigned ind;
            int pos;
            bool k_real_positive = k.imag() == 0 && k.real() > 0;
            for (unsigned I = 0; I < grid_size_x; ++I) {
                for (unsigned J = 0; J < grid_size_y; ++J) {
                    Eigen::Vector2d x;
                    x << lower_left_corner(0) + I * x_step, lower_left_corner(1) + J * y_step;
                    grid_X(I, J) = x(0);
                    grid_Y(I, J) = x(1);
                    pos = ppoly(mesh.getPanels(), x, ind, excess);
                    if (pos == 0) { // on the boundary
                        t0 = 0.;
                        for (unsigned i = 0; i < ind; ++i)
                            t0 += mesh.getPanels()[i]->length();
                        t0 += excess * mesh.getPanels()[ind]->length();
                        S(I, J) = trace_dir.eval(t0);
                    } else { // not on the boundary (pos = 1 if inside, else pos = -1)
                        kappa = (pos == 1 ? k_sqrt_ci : k_sqrt_co);
                        Z = Y + x(0) + 1i * x(1);
                        X = Z.cwiseAbs();
                        if (k_real_positive)
                            complex_bessel::H1_01(kappa.real() * X, H0, H1);
                        else
                            complex_bessel::H1_01_cplx(kappa * X, H0, H1);
                        G = (H0 * (pos == 1 ? Trace_i_N : Trace_o_N)
                            - kappa * H1 * (pos == 1 ? Trace_i_D : Trace_o_D) * (Z.real() * N.real() + Z.imag() * N.imag()) / X) * D;
                        S(I, J) = double(pos) * 1i * 0.25 * G.sum();
                        if (add_u_inc && pos == -1)
                            S(I, J) += u_inc(x(0), x(1));
                    }
                }
            }
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
                                            GalerkinMatrixBuilder &builder,
                                            Eigen::MatrixXcd &so) {
            auto traces = in_traces(mesh, u_inc, u_inc_del, k, c_o, c_i, builder, so);
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
            auto traces = solve(mesh, u_inc, u_inc_del, order_galerkin, k, c_o, c_i);
            ComplexSpline trace_dir(mesh, traces.head(numpanels));
            ComplexSpline trace_neu(mesh, traces.tail(numpanels));
            Eigen::ArrayXXcd S;
            in_solve(mesh, u_inc, u_inc_del, trace_dir, trace_neu, order_green, k, c_o, c_i,
                     lower_left_corner, upper_right_corner, grid_size_x, grid_size_y, total_field, grid_X, grid_Y, S);
            return S;
        }

    } // namespace direct_second_kind
} // namespace tsp
