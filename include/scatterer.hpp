/**
 * \file scatterer.hpp
 *
 * \brief This is a C++ library of utilities for
 * handling the scatterer object. The input file
 * syntax (CSV) is described below.
 * Each line of the input file specifies a vertex
 * of the scatterer. A vertex is specified as a
 * pair xk,yk. Vertices should be sorted in
 * counter-clockwise order.
 * Lines starting with '#' are ignored. Empty lines
 * and extra spaces are discarded.
 *
 * (c) 2024 Luka Marohnić
 */

#ifndef SCATTERER_HPP
#define SCATTERER_HPP

#include "parametrized_mesh.hpp"
#include <Eigen/Dense>

using namespace std;

namespace scatterer {

    /**
    * This function reads polygonal scatterer from
    * file and fills x and y with vertex coordinates.
    *
    * @param fname file name
    * @param x x-coordinates of vertices
    * @param y y-coordinates of vertices
    */
    bool read_polygon(const std::string& fname, Eigen::VectorXd& x, Eigen::VectorXd& y, double scale = 1.0);

    /**
    * This function computes the number of
    * panels automatically by setting panel
    * length to approximately f times the
    * length of the shortest side.
    *
    * @param x x-coordinates of vertices
    * @param y y-coordinates of vertices
    * @param f fraction of the shortest side
    */
    unsigned int auto_num_panels(const Eigen::VectorXd &x, const Eigen::VectorXd &y, double f);

    /**
    * This function returns the mesh of approximately
    * N panels of approximately equal lengths which
    * linearly interpolate vertices (x,y).
    *
    * @param x x-coordinates of vertices
    * @param y y-coordinates of vertices
    * @param N desired number of panels
    */
    ParametrizedMesh panelize(const Eigen::VectorXd &x, const Eigen::VectorXd &y, unsigned N, double refinement_factor);

    /**
    * This function returns true if p is inside polygon
    * formed by panels and false otherwise.
    *
    * @param mesh parametrized panel mesh
    * @param p point
    */
    bool inside_poly(const ParametrizedMesh &mesh, const Eigen::Vector2d &p);

    /**
    * This function returns the distance from p
    * to the i-th panel of the mesh.
    *
    * @param mesh parametrized panel mesh
    * @param i panel index
    * @param p point
    * @param t0 pointer to the position of the projection of p onto the i-th panel
    */
    double panel_distance(const ParametrizedMesh &mesh, unsigned i, const Eigen::Vector2d &p, double *t0 = NULL);

    double distance(const ParametrizedMesh &mesh, const Eigen::Vector2d &p, unsigned *i = NULL, double *t0 = NULL);
}
#endif // SCATTERER_HPP
