#include "double_layer_der.hpp"
#include "cbessel.hpp"

    namespace double_layer_helmholtz_der {

        typedef std::complex<double> complex_t;
        complex_t ii = complex_t(0.0,1.0);
        double epsilon = std::numeric_limits<double>::epsilon();

        Eigen::MatrixXcd InteractionMatrix(const AbstractParametrizedCurve &pi,
                                           const AbstractParametrizedCurve &pi_p,
                                           const AbstractBEMSpace &trial_space,
                                           const AbstractBEMSpace &test_space,
                                           const QuadRule &GaussQR,
                                           const QuadRule &CGaussQR,
                                           const complex_t k,
                                           const double c) {
            if (&pi == &pi_p) { // Same Panels case
                return ComputeIntegralCoinciding(pi, pi_p, trial_space, test_space, CGaussQR, k, c);
            }
            else if ((pi(1) - pi_p(-1)).norm() / 100. < epsilon ||
                     (pi(-1) - pi_p(1)).norm() / 100. < epsilon) {// Adjacent Panels case
                return ComputeIntegralAdjacent(pi, pi_p, trial_space, test_space, CGaussQR, k, c);
            } else { //Disjoint panels case
                return ComputeIntegralGeneral(pi, pi_p, trial_space, test_space, GaussQR, k, c);
            }
        }

        Eigen::MatrixXcd ComputeIntegralCoinciding(const AbstractParametrizedCurve &pi,
                                                   const AbstractParametrizedCurve &pi_p,
                                                   const AbstractBEMSpace &trial_space,
                                                   const AbstractBEMSpace &test_space,
                                                   const QuadRule &GaussQR,
                                                   const complex_t k,
                                                   const double c) {
            unsigned N = GaussQR.n; // quadrature order for the GaussQR object.
            // The number of Reference Shape Functions in trial space
            int Qtrial = trial_space.getQ();
            // The number of Reference Shape Functions in test space
            int Qtest = test_space.getQ();
            // Interaction matrix with size Qtest x Qtrial
            Eigen::MatrixXcd interaction_matrix(Qtest, Qtrial);
            // Computing the (i,j)th matrix entry
            for (int i = 0; i < Qtest; ++i) {
                for (int j = 0; j < Qtrial; ++j) {
                    // Lambda expression for functions F and G
                    auto F = [&](double t) {
                        return trial_space.evaluateShapeFunction(j, t) * pi_p.Derivative_01(t).norm();
                    };
                    auto G = [&](double s) {
                        return test_space.evaluateShapeFunction(i, s) * pi.Derivative_01(s).norm();
                    };
                    // Lambda expression for the integrand
                    auto integrand = [&](double s, double t) {
                        complex_t result = complex_t(0.,0.);
                        // Finding the tangent of pi_p to get its normal
                        Eigen::Vector2d tangent = pi_p.Derivative_01(t);
                        Eigen::Vector2d normal;
                        // Outward normal vector
                        normal << tangent(1), -tangent(0);
                        // Normalizing the normal vector
                        normal = normal / normal.norm();
                        if ( abs(k*sqrt(c))*(pi[s]-pi_p[t]).norm() > epsilon ) { // Away from singularity
                            result = (complex_bessel::HankelH1p(1,k*sqrt(c)*(pi[s]-pi_p[t]).norm())*(pi[s]-pi_p[t]).norm()*k*sqrt(c)
                                         + complex_bessel::HankelH1(1,k*sqrt(c)*(pi[s]-pi_p[t]).norm()))
                                     *(pi[s]-pi_p[t]).normalized().dot(normal);
                            //std::cout << complex_bessel::H1p(1,k*sqrt(c)*(pi[s]-pi[t]).norm()) << std::endl;
                        }
                        return result*F(t)*G(s);
                    };
                    complex_t integral = complex_t(0.,0.);
                    // Tensor product quadrature for double integral
                    for (unsigned int k = 0; k < N; ++k) {
                        for (unsigned int l = 0; l < N; ++l) {
                            double s = GaussQR.x(l)*(1.-GaussQR.x(k));
                            double t = GaussQR.x(l);
                            double w = GaussQR.x(l)*GaussQR.w(k)*GaussQR.w(l);
                            integral += w*integrand(s,t);
                            integral += w*integrand(t,s);
                        }
                    }
                    // Filling the matrix entry
                    interaction_matrix(i, j) = ii*sqrt(c)*integral/4.;
                }
            }
            return interaction_matrix;
        }

        Eigen::MatrixXcd ComputeIntegralAdjacent(const AbstractParametrizedCurve &pi,
                                                 const AbstractParametrizedCurve &pi_p,
                                                 const AbstractBEMSpace &trial_space,
                                                 const AbstractBEMSpace &test_space,
                                                 const QuadRule &GaussQR,
                                                 const complex_t k,
                                                 const double c) {
            unsigned N = GaussQR.n; // quadrature order for the GaussQR object.
            // The number of Reference Shape Functions in trial space
            int Qtrial = trial_space.getQ();
            // The number of Reference Shape Functions in test space
            int Qtest = test_space.getQ();
            // Interaction matrix with size Qtest x Qtrial
            Eigen::MatrixXcd interaction_matrix(Qtest, Qtrial);
            // Computing the (i,j)th matrix entry
            bool swap = ((pi(1) - pi_p(-1)).norm() / 100. > sqrt(epsilon));
            for (int i = 0; i < Qtest; ++i) {
                for (int j = 0; j < Qtrial; ++j) {
                    // Lambda expression for functions F and G
                    auto F = [&](double t) {
                        if (swap) {
                            return trial_space.evaluateShapeFunction_01_swapped(j, t) * pi_p.Derivative_01_swapped(t).norm();
                        } else {
                            return trial_space.evaluateShapeFunction(j, t) * pi_p.Derivative_01(t).norm();
                        }
                    };
                    auto G = [&](double s) {
                        if (swap) {
                            return test_space.evaluateShapeFunction(i, s) * pi.Derivative_01(s).norm();
                        } else {
                            return test_space.evaluateShapeFunction_01_swapped(i, s) * pi.Derivative_01_swapped(s).norm();
                        }
                    };
                    // Lambda expression for the integrand
                    auto integrand = [&](double s, double t) {
                        complex_t result = complex_t(0.,0.);
                        // Finding the tangent of pi_p to get its normal
                        Eigen::Vector2d tangent = swap ? pi_p.Derivative_01_swapped(t) : pi_p.Derivative_01(t);
                        Eigen::Vector2d normal;
                        // Outward normal vector
                        if (swap) {
                            normal << -tangent(1), tangent(0);
                        } else {
                            normal << tangent(1), -tangent(0);
                        }
                        // Normalizing the normal vector
                        normal = normal / normal.norm();
                        if (swap) {
                            if ( abs(k*sqrt(c))*(pi[s]-pi_p.swapped_op(t)).norm() > epsilon ) { // Away from singularity
                                result = (complex_bessel::HankelH1p(1,k*sqrt(c)*(pi[s]-pi_p.swapped_op(t)).norm())*(pi[s]-pi_p.swapped_op(t)).norm()*k*sqrt(c)
                                             +complex_bessel::HankelH1(1,k*sqrt(c)*(pi[s]-pi_p.swapped_op(t)).norm()))
                                         *(pi[s]-pi_p.swapped_op(t)).normalized().dot(normal);
                            }
                        } else {
                            if ( abs(k*sqrt(c))*(pi.swapped_op(s)-pi_p[t]).norm() > epsilon ) { // Away from singularity
                                result = (complex_bessel::HankelH1p(1,k*sqrt(c)*(pi.swapped_op(s)-pi_p[t]).norm())*(pi.swapped_op(s)-pi_p[t]).norm()*k*sqrt(c)
                                             +complex_bessel::HankelH1(1,k*sqrt(c)*(pi.swapped_op(s)-pi_p[t]).norm()))
                                         *(pi.swapped_op(s)-pi_p[t]).normalized().dot(normal);
                            }
                        }
                        return result * F(t) * G(s);
                    };
                    complex_t integral = complex_t(0.,0.);
                    // Tensor product quadrature for double integral
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
                    interaction_matrix(i, j) = ii*sqrt(c)*integral/4.;
                }
            }
            return interaction_matrix;
        }

        Eigen::MatrixXcd ComputeIntegralGeneral(const AbstractParametrizedCurve &pi,
                                                const AbstractParametrizedCurve &pi_p,
                                                const AbstractBEMSpace &trial_space,
                                                const AbstractBEMSpace &test_space,
                                                const QuadRule &GaussQR,
                                                const complex_t k,
                                                const double c) {
            unsigned N = GaussQR.n; // quadrature order for the GaussQR object.
            // The number of Reference Shape Functions in space
            int Qtrial = trial_space.getQ();
            // The number of Reference Shape Functions in space
            int Qtest = test_space.getQ();
            // Interaction matrix with size Qtest x Qtrial
            Eigen::MatrixXcd interaction_matrix(Qtest, Qtrial);
            // Computing the (i,j)th matrix entry
            for (int i = 0; i < Qtest; ++i) {
                for (int j = 0; j < Qtrial; ++j) {
                    // Lambda expression for functions F and G
                    // Double Layer BIO
                    auto F = [&](double t) { // Function associated with panel pi_p
                        return trial_space.evaluateShapeFunction(j, t) *
                               pi_p.Derivative_01(t).norm();
                    };
                    auto G = [&](double s) { // Function associated with panel pi
                        return test_space.evaluateShapeFunction(i, s) *
                               pi.Derivative_01(s).norm();
                    };
                    // Lambda expression for \f$\hat{K}\f$
                    // Layer BIO
                    auto integrand = [&](double s, double t) {
                        complex_t result = complex_t(0.,0.);
                        // Finding the tangent of pi_p to get its normal
                        Eigen::Vector2d tangent = pi_p.Derivative_01(t);
                        Eigen::Vector2d normal;
                        // Outward normal vector
                        normal << tangent(1), -tangent(0);
                        // Normalizing the normal vector
                        normal = normal / normal.norm();
                        if ( abs(k*sqrt(c))*(pi[s]-pi_p[t]).norm() > epsilon ) { // Away from singularity
                            result = (complex_bessel::HankelH1p(1,k*sqrt(c)*(pi[s]-pi_p[t]).norm())*(pi[s]-pi_p[t]).norm()*k*sqrt(c)
                                        +complex_bessel::HankelH1(1,k*sqrt(c)*(pi[s]-pi_p[t]).norm()))
                                                *(pi[s]-pi_p[t]).normalized().dot(normal);
                        }
                        return result*F(t)*G(s);
                    };
                    complex_t integral = complex_t(0.,0.);
                    // Tensor product quadrature for double integral
                    for (unsigned int i = 0; i < N; ++i) {
                        for (unsigned int j = 0; j < N; ++j) {
                            double s = GaussQR.x(i);
                            double t = GaussQR.x(j);
                            double w = GaussQR.w(i) * GaussQR.w(j);
                            integral += w*integrand(s, t);
                        }
                    }
                    // Filling the matrix entry
                    interaction_matrix(i, j) = ii*sqrt(c)*integral/4.;
                }
            }
            return interaction_matrix;
        }

        Eigen::MatrixXcd GalerkinMatrix(const ParametrizedMesh mesh,
                                        const AbstractBEMSpace &trial_space,
                                        const AbstractBEMSpace &test_space,
                                        const unsigned int &N,
                                        const complex_t k,
                                        const double c) {
            // Getting number of panels in the mesh
            unsigned int numpanels = mesh.getNumPanels();
            // Getting dimensions for trial and test spaces
            unsigned int rows = test_space.getSpaceDim(numpanels);
            unsigned int cols = trial_space.getSpaceDim(numpanels);
            // Getting the panels from the mesh
            PanelVector panels = mesh.getPanels();
            // Getting the number of local shape functions in the trial and test spaces
            unsigned int Qtest = test_space.getQ();
            unsigned int Qtrial = trial_space.getQ();
            // Initializing the Galerkin matrix with zeros
            Eigen::MatrixXcd output = Eigen::MatrixXd::Zero(rows, cols);
            // Panel oriented assembly
            QuadRule GaussQR = getGaussQR(N,0.,1.);
            QuadRule CGaussQR = getCGaussQR(N);
            for (unsigned int i = 0; i < numpanels; ++i) {
                for (unsigned int j = 0; j < numpanels; ++j) {
                    // Getting the interaction matrix for the pair of panels i and j
                    Eigen::MatrixXcd interaction_matrix = InteractionMatrix(
                            *panels[i], *panels[j], trial_space, test_space, GaussQR, CGaussQR, k,c);
                    // Local to global mapping of the elements in interaction matrix
                    for (unsigned int ii = 0; ii < Qtest; ++ii) {
                        for (unsigned int jj = 0; jj < Qtrial; ++jj) {
                            int II = test_space.LocGlobMap(ii + 1, i + 1, numpanels) - 1;
                            int JJ = trial_space.LocGlobMap(jj + 1, j + 1, numpanels) - 1;
                            // Filling the Galerkin matrix entries
                            output(II, JJ) += interaction_matrix(ii, jj);
                        }
                    }
                }
            }
            return output;
        }

    } // namespace double_layer_helmholtz

