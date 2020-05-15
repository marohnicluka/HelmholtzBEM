/**
 * \file single_layer.cpp
 * \brief This file declares the functions to evaluate the entries of
 *        Galerkin matrices based on the bilinear form induced by the
 *        Single Layer BIO, using the transformations given in
 *        \f$\ref{ss:quadapprox}\f$ in the Lecture Notes for Advanced Numerical
 *        Methods for CSE.
 *
 * This File is a part of the 2D-Parametric BEM package
 */

#include "single_layer_der2.hpp"
#include "/usr/include/complex_bessel.h"

namespace parametricbem2d {
    namespace single_layer_helmholtz_der2 {

        typedef std::complex<double> complex_t;
        complex_t ii = complex_t(0.0,1.0);
        double epsilon = std::numeric_limits<double>::epsilon();

        Eigen::MatrixXcd InteractionMatrix(const AbstractParametrizedCurve &pi,
                                           const AbstractParametrizedCurve &pi_p,
                                           const AbstractBEMSpace &space,
                                           const QuadRule &GaussQR,
                                           const QuadRule &CGaussQR,
                                           const double c,
                                           const complex_t k) {
            if (&pi == &pi_p) { // Same Panels case
                return ComputeIntegralCoinciding(pi, pi_p, space, CGaussQR, c, k);
            }
            else if ((pi(1) - pi_p(-1)).norm() / 100. < epsilon ||
                     (pi(-1) - pi_p(1)).norm() / 100. < epsilon) {// Adjacent Panels case
                return ComputeIntegralAdjacent(pi, pi_p, space, CGaussQR, c, k);
            }
            else {// Disjoint panels case
                return ComputeIntegralGeneral(pi, pi_p, space, GaussQR, c, k);
            }
        }

        Eigen::MatrixXcd ComputeIntegralAdjacent(const AbstractParametrizedCurve &pi,
                                                 const AbstractParametrizedCurve &pi_p,
                                                 const AbstractBEMSpace &space,
                                                 const QuadRule &GaussQR,
                                                 const double c,
                                                 const complex_t k) {
            unsigned N = GaussQR.n; // Quadrature order for the GaussQR object.
            // The number of Reference Shape Functions in trial space
            int Qtrial = space.getQ();
            // The number of Reference Shape Functions in test space
            int Qtest = space.getQ();
            // Interaction matrix with size Qtest x Qtrial
            Eigen::MatrixXcd interaction_matrix(Qtest, Qtrial);
            // Computing the (i,j)th matrix entry
            bool swap = (pi(1) - pi_p(-1)).norm() / 100. > epsilon;
            for (int i = 0; i < Qtest; ++i) {
                for (int j = 0; j < Qtrial; ++j) {
                    // Lambda expression for functions F and G in \f$\eqref{eq:Kidp}\f$
                    auto F = [&](double t) {
                        if (swap) {
                            return space.evaluateShapeFunction_01_swapped(j, t) * pi_p.Derivative_01_swapped(t).norm();
                        } else {
                            return space.evaluateShapeFunction_01(j, t) * pi_p.Derivative_01(t).norm();
                        }
                    };
                    auto G = [&](double s) {
                        if (swap) {
                            return space.evaluateShapeFunction_01(i, s) * pi.Derivative_01(s).norm();
                        } else {
                            return space.evaluateShapeFunction_01_swapped(i, s) * pi.Derivative_01_swapped(s).norm();
                        }
                    };
                    // Lambda expression for the integrand in \f$\eqref{eq:Kidp}\f$
                    auto integrand = [&](double s, double t) {
                        complex_t result = complex_t(0.,0.);
                        if (swap) {
                            if ( abs(k)*(pi[s]-pi_p.swapped_op(t)).norm() > epsilon) {
                                result = sp_bessel::hankelH1p(1,k*(pi[s]-pi_p.swapped_op(t)).norm())
                                                           *(pi[s]-pi_p.swapped_op(t)).norm();
                            };
                        } else {
                            if ( abs(k)*(pi.swapped_op(s)-pi_p[t]).norm() > epsilon) {
                                result = sp_bessel::hankelH1p(1,k*(pi.swapped_op(s)-pi_p[t]).norm())
                                         *(pi.swapped_op(s)-pi_p[t]).squaredNorm();
                            };
                        }
                        return result * F(t) * G(s);
                    };
                    complex_t integral = complex_t(0.,0.);
                    // Tensor product quadrature for double integral in \f$\eqref{eq:Kidp}\f$
                    for (unsigned int k = 0; k < N; ++k) {
                        for (unsigned int l = 0; l < N; ++l) {
                            double s = GaussQR.x(k)*GaussQR.x(l);
                            double t = GaussQR.x(k);
                            double w = GaussQR.x(k)*GaussQR.w(k)*GaussQR.w(l);
                            integral += w*integrand(s,t);
                            integral += w*integrand(t,s);
                        }
                    }
                    // Filling the matrix entry
                    interaction_matrix(i, j) = -ii*c*c*integral/4.;
                }
            }
            return interaction_matrix;
        }

        Eigen::MatrixXcd ComputeIntegralCoinciding(const AbstractParametrizedCurve &pi,
                                                   const AbstractParametrizedCurve &pi_p,
                                                   const AbstractBEMSpace &space,
                                                   const QuadRule &GaussQR,
                                                   const double c,
                                                   const complex_t k) {
            unsigned N = GaussQR.n; // Quadrature order for the GaussQR object.
            // Calculating the quadrature order for stable evaluation of integrands for
            // disjoint panels as mentioned in \f$\ref{par:distpan}\f$
            // No. of Reference Shape Functions in trial/test space
            int Q = space.getQ();
            // Interaction matrix with size Q x Q
            Eigen::MatrixXcd interaction_matrix(Q, Q);
            // Computing the (i,j)th matrix entry
            for (int i = 0; i < Q; ++i) {
                for (int j = 0; j < Q; ++j) {
                    // Lambda expression for functions F and G in \f$\eqref{eq:titg}\f$ for
                    // Single Layer BIO
                    auto F = [&](double t) { // Function associated with panel pi_p
                        return space.evaluateShapeFunction_01(j, t) * pi_p.Derivative_01(t).norm();
                    };
                    auto G = [&](double s) { // Function associated with panel pi
                        return space.evaluateShapeFunction_01(i, s) * pi.Derivative_01(s).norm();
                    };
                    auto integrand = [&](double s, double t) {
                        complex_t result = complex_t(0.0,0.0);
                        if ( abs(k)*(pi[s]-pi_p[t]).norm() > epsilon) {
                            result = sp_bessel::hankelH1p(1,k*(pi[s]-pi_p[t]).norm())
                                     *(pi[s]-pi_p[t]).squaredNorm();
                        }
                        return result*F(t)*G(s);
                    };
                    complex_t integral = complex_t(0.,0.);
                    // Tensor product quadrature rule
                    for (unsigned int k = 0; k < N; ++k) {
                        for (unsigned int l = 0; l < N; ++l) {
                            double s = GaussQR.x(l)*(1.-GaussQR.x(k));
                            double t = GaussQR.x(l);
                            double w = GaussQR.x(l)*GaussQR.w(k)*GaussQR.w(l);
                            integral += w*integrand(s,t);
                            integral += w*integrand(t,s);
                        }
                    }
                    // Filling up the matrix entry
                    interaction_matrix(i, j) = -ii*c*c*integral/4.;
                }
            }
            return interaction_matrix;
        }

        Eigen::MatrixXcd ComputeIntegralGeneral(const AbstractParametrizedCurve &pi,
                                                const AbstractParametrizedCurve &pi_p,
                                                const AbstractBEMSpace &space,
                                                const QuadRule &GaussQR,
                                                const double c,
                                                const complex_t k) {
            unsigned N = GaussQR.n; // Quadrature order for the GaussQR object.
            // Calculating the quadrature order for stable evaluation of integrands for
            // disjoint panels as mentioned in \f$\ref{par:distpan}\f$
            // No. of Reference Shape Functions in trial/test space
            int Q = space.getQ();
            // Interaction matrix with size Q x Q
            Eigen::MatrixXcd interaction_matrix(Q, Q);
            // Computing the (i,j)th matrix entry
            for (int i = 0; i < Q; ++i) {
                for (int j = 0; j < Q; ++j) {
                    // Lambda expression for functions F and G in \f$\eqref{eq:titg}\f$ for
                    // Single Layer BIO
                    auto F = [&](double t) { // Function associated with panel pi_p
                        return space.evaluateShapeFunction_01(j, t) * pi_p.Derivative_01(t).norm();
                    };
                    auto G = [&](double s) { // Function associated with panel pi
                        return space.evaluateShapeFunction_01(i, s) * pi.Derivative_01(s).norm();
                    };
                    auto integrand = [&](double s, double t) {
                        complex_t result = complex_t(0.0,0.0);
                        if ( abs(k)*(pi[s]-pi_p[t]).norm() > epsilon) {
                            result = sp_bessel::hankelH1p(1,k*(pi[s]-pi_p[t]).norm())
                                     *(pi[s]-pi_p[t]).squaredNorm();
                        }
                        return result*F(t)*G(s);
                    };
                    complex_t integral = complex_t(0.,0.);
                    // Tensor product quadrature rule
                    for (unsigned int k = 0; k < N; ++k) {
                        for (unsigned int l = 0; l < N; ++l) {
                            double s = GaussQR.x(k);
                            double t = GaussQR.x(l);
                            double w = GaussQR.w(k)*GaussQR.w(l);
                            integral += w*integrand(s,t);
                        }
                    }
                    // Filling up the matrix entry
                    interaction_matrix(i, j) = -ii*c*c*integral/4.;
                }
            }
            return interaction_matrix;
        }

        Eigen::MatrixXcd GalerkinMatrix(const ParametrizedMesh mesh,
                                        const AbstractBEMSpace &space,
                                        const unsigned int &N,
                                        const double c,
                                        const complex_t k) {
            // Getting the number of panels in the mesh
            unsigned int numpanels = mesh.getNumPanels();
            // Getting dimensions of trial/test space
            unsigned int dims = space.getSpaceDim(numpanels);
            // Getting the panels from the mesh
            PanelVector panels = mesh.getPanels();
            // Getting the number of local shape functions in the trial/test space
            unsigned int Q = space.getQ();
            // Initializing the Galerkin matrix with zeros
            Eigen::MatrixXcd output = Eigen::MatrixXcd::Zero(dims, dims);
            // Panel oriented assembly \f$\ref{pc:ass}\f$
            QuadRule GaussQR = getGaussQR(N,0.,1.);
            QuadRule CGaussQR = getCGaussQR(N);
            for (unsigned int i = 0; i < numpanels; ++i) {
                for (unsigned int j = 0; j < numpanels; ++j) {
                    // Getting the interaction matrix for the pair of panels i and j
                    Eigen::MatrixXcd interaction_matrix =
                            InteractionMatrix(*panels[i], *panels[j], space, GaussQR, CGaussQR, c, k);
                    // Local to global mapping of the elements in interaction matrix
                    for (unsigned int I = 0; I < Q; ++I) {
                        for (unsigned int J = 0; J < Q; ++J) {
                            int II = space.LocGlobMap(I + 1, i + 1, numpanels) - 1;
                            int JJ = space.LocGlobMap(J + 1, j + 1, numpanels) - 1;
                            // Filling the Galerkin matrix entries
                            output(II, JJ) += interaction_matrix(I, J);
                        }
                    }
                }
            }
            return output;
        }

    } //namespace single_layer_helmholtz
} // namespace parametricbem2d
