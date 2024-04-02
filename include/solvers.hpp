/**
 * \file solvers.hpp
 * \brief This file defines lowest order solvers for direct first kind BIEs
 * for the Dirichlet and the Neumann problem as well as a direct second kind
 * BIE for the Helmholtz transmission problem.
 *
 * This File is a part of the HelmholtzTransmissionBEM
 */

#ifndef SOLVERSHPP
#define SOLVERSHPP

#include "parametrized_mesh.hpp"
#include "galerkin_matrix_builder.hpp"
/**
 * \namespace bvp
 * \brief This namespace contains solvers for boundary value problems.
 */
namespace bvp {

/**
 * \namespace direct_first_kind
 * \brief This namespace contains solvers using direct first kind BIE
 * for the Dirichlet and the Neumann problem.
 * The solvers use the lowest order BEM spaces for computation.
 */
    namespace direct_first_kind {

        /**
         * This function solves the Dirichlet problem given a mesh \p mesh, the Dirichlet data of u
         * \p u_dir, the order of the quadrature rule used to compute the Galerkin matrix entries
         * \p order and the wavenumber \p k.
         * @param mesh mesh of the boundary on which to compute BIOs
         * @param u_dir Dirichlet data
         * @param order order of qudrature rule for matrix entries
         * @param k wavenumber
         * @return Neumann data of u
         */
        Eigen::VectorXcd solve_dirichlet(const ParametrizedMesh &mesh,
                                         const std::function<std::complex<double>(double, double)> u_dir,
                                         const unsigned order,
                                         const double k);

        /**
         * This function solves the Neumann problem given a mesh \p mesh, the Neumann data of u
         * \p u_dir, the order of the quadrature rule used to compute the Galerkin matrix entries
         * \p order and the wavenumber \p k.
         * @param mesh mesh of the boundary on which to compute BIOs
         * @param u_neu Dirichlet data
         * @param order order of qudrature rule for matrix entries
         * @param k wavenumber
         * @return Dirichlet data of u
         */
        Eigen::VectorXcd solve_neumann(const ParametrizedMesh &mesh,
                                       const std::function<std::complex<double>(double, double)> u_neu,
                                       const unsigned order,
                                       const double k);
    } // namespace direct_first_kind
} // namespace bvp

/**
 * \namespace tp
 * \brief This namespace contains the solver for the Helmholtz transmission problem.
 */
namespace tp {

    /**
	 * \namespace direct_second_kind
	 * \brief This namespace contains the solver using direct second kind BIEs
     * for the Helmholtz transmission problem.
     * The solver uses the lowest order BEM spaces for computation.
     */
    namespace direct_second_kind {
        /**
         * This function returns the solution to the Helmholtz transmission problem
         * on boundary given by \p mesh for an incoming wave defined by \p u_inc_dir and
         * \p u_inc_neu. The wavenumber is set by \p k and th refraction indeces by
         * \p c_o and \p c_i. The Galerkin matrix entries are compute with a quadrature rule
         * defined by the parameter \p order.
         * @param mesh mesh of the boundary on which to compute BIOs
         * @param u_inc_dir Dirichlet data of incoming wave
         * @param u_inc_neu Neumann data of incoming wave
         * @param order order of qudrature rule for matrix entries
         * @param k wavenumber
         * @param c_o refraction index outer domain
         * @param c_i refraction index on inner domain
         * @return Dirichlet and Neumann data of resulting wave
         */
        Eigen::VectorXcd solve(const ParametrizedMesh &mesh,
                               const std::function<std::complex<double>(double, double)> u_inc,
                               const std::function<Eigen::Vector2cd(double, double)> u_inc_del,
                               const unsigned order,
                               const std::complex<double> &k,
                               const double c_o,
                               const double c_i);

        /**
         * This function returns the solution to the Helmholtz transmission problem
         * in the rectangle [lower_left_corner, upper_right_corner] with a scatterer defined by
         * \p mesh for an incoming wave defined by \p u_inc_dir and
         * \p u_inc_neu. The wavenumber is set by \p k and th refraction indeces by
         * \p c_o and \p c_i. The Galerkin matrix entries are compute with a quadrature rule
         * defined by the parameter \p order.
         * The function returns the scattered/transmitted wave and initializes Galerkin builder
         * automatically.
         * @param mesh mesh of the boundary on which to compute BIOs
         * @param u_inc incoming wave
         * @param u_inc_del incoming wave gradient as 2d vector
         * @param order_galerkin order of qudrature rule for matrix entries in Galerkin matrices
         * @param order_green order of qudrature rule for Green identity
         * @param k wavenumber
         * @param c_o refraction index outer domain
         * @param c_i refraction index on inner domain
         * @param lower_left_corner coordinates of the lower left corner of the rectangle
         * @param lower_left_corner coordinates of the upper right corner of the rectangle
         * @param grid_size_x number of points in grid at x-axis
         * @param grid_size_y number of points in grid at y-axis
         * @param grid_X x-coordinates of grid points
         * @param grid_Y y-coordinates of grid points
         * @param total_field whether to add the incoming wave to the scattered wave
         * @return the resulting wave in the specified rectangle
         */
        Eigen::ArrayXXcd solve_in_rectangle(const ParametrizedMesh &mesh,
                               const std::function<std::complex<double>(double, double)> u_inc,
                               const std::function<Eigen::Vector2cd(double, double)> u_inc_del,
                               const unsigned order_galerkin,
                               const unsigned order_green,
                               const std::complex<double> &k,
                               const double c_o,
                               const double c_i,
                               const Eigen::Vector2d &lower_left_corner,
                               const Eigen::Vector2d &upper_right_corner,
                               const unsigned grid_size_x,
                               const unsigned grid_size_y,
                               Eigen::ArrayXXd &grid_X,
                               Eigen::ArrayXXd &grid_Y,
                               bool total_field);

        /**
         * This function returns the solution to the Helmholtz transmission problem
         * in the rectangle [lower_left_corner, upper_right_corner] with a scatterer defined by
         * \p mesh for an incoming wave defined by \p u_inc_dir and
         * \p u_inc_neu. The wavenumber is set by \p k and th refraction indeces by
         * \p c_o and \p c_i. The Galerkin matrix entries are compute with a quadrature rule
         * defined by the parameter \p order.
         * This function uses pre-existing builder and outputs the solutions
         * operator matrix.
         * @param mesh mesh of the boundary on which to compute BIOs
         * @param u_inc incoming wave
         * @param u_inc_del incoming wave gradient as 2d vector
         * @param order order of qudrature rule for Green identity
         * @param k wavenumber
         * @param c_o refraction index outer domain
         * @param c_i refraction index on inner domain
         * @param lower_left_corner coordinates of the lower left corner of the rectangle
         * @param lower_left_corner coordinates of the upper right corner of the rectangle
         * @param grid_size_x number of points in grid at x-axis
         * @param grid_size_y number of points in grid at y-axis
         * @param grid_X x-coordinates of grid points
         * @param grid_Y y-coordinates of grid points
         * @param total_field whether to add the incoming wave to the scattered wave
         * @param builder Galerkin matrix builder
         * @param so solutions operator matrix
         * @return the resulting wave in the specified rectangle
         */
        Eigen::ArrayXXcd solve_in_rectangle(const ParametrizedMesh &mesh,
                               const std::function<std::complex<double>(double, double)> u_inc,
                               const std::function<Eigen::Vector2cd(double, double)> u_inc_del,
                               const unsigned order,
                               const std::complex<double> &k,
                               const double c_o,
                               const double c_i,
                               const Eigen::Vector2d &lower_left_corner,
                               const Eigen::Vector2d &upper_right_corner,
                               const unsigned grid_size_x,
                               const unsigned grid_size_y,
                               Eigen::ArrayXXd &grid_X,
                               Eigen::ArrayXXd &grid_Y,
                               bool total_field,
                               GalerkinMatrixBuilder &builder,
                               Eigen::MatrixXcd &so);
    } // namespace direct_second_kind
} // namespace tp
#endif // DIRICHLETHPP
