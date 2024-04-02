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

#include "parametrized_line.hpp"
#include <Eigen/Dense>

using namespace std;

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
 * This function returns the vector of approximately
 * N panels of approximately equal lengths which
 * linearly interpolate vertices (x,y).
 *
 * @param x x-coordinates of vertices
 * @param y y-coordinates of vertices
 * @param N desired number of panels
 */
PanelVector make_scatterer(const Eigen::VectorXd &x, const Eigen::VectorXd &y, unsigned N, double refinement_factor);

/**
 * This function returns 1 if p is inside polygon
 * formed by panels, -1 if p is outside, and 0
 * if p is on the boundary (with the specified tolerance).
 *
 * @param panels list of parametrized line panels
 * @param p point
 * @param ind the index of the nearest panel if x is on the boundary
 * @param t parameter value corresponding to the nearest point if x is on boundary
 * @param boundary_threshold boundary width, default 0.001
 */
int ppoly(const PanelVector &panels, const Eigen::Vector2d &p, unsigned &ind, double &t, double boundary_threshold = 1e-5);

#endif // SCATTERER_HPP
