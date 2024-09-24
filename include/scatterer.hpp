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
#include "parametrized_polynomial.hpp"
#include "cspline.hpp"
#include "gauleg.hpp"

using namespace std;

namespace scatterer {

    typedef std::vector<std::vector<std::string> > options;

    /**
    * This function reads polygonal scatterer from
    * file and fills x and y with vertex coordinates.
    *
    * @param fname file name
    * @param x x-coordinates of vertices
    * @param y y-coordinates of vertices
    */
    bool read(const std::string& fname, Eigen::VectorXd& x, Eigen::VectorXd& y, double scale = 1.0);

    /**
    * This function computes the polygon length
    * from its vertices (x,y).
    *
    * @param x x-coordinates of vertices
    * @param y y-coordinates of vertices
    * @return the length of the polygon
    */
    double length(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

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
    * This function returns a polygonal mesh of
    * approx N panels of near-equal lengths which
    * linearly interpolate vertices (x,y).
    *
    * @param x x-coordinates of vertices
    * @param y y-coordinates of vertices
    * @param N desired number of panels
    * @param refinement_factor the max allowed panel shrinkage
    */
    PanelVector make_polygonal_panels(const Eigen::VectorXd &x, const Eigen::VectorXd &y, unsigned N, double refinement_factor = 0.5);

    /**
    * This function returns a polygonal mesh of
    * exactly N panels of near-equal lengths which
    * linearly interpolate vertices (x,y).
    *
    * @param x x-coordinates of vertices
    * @param y y-coordinates of vertices
    * @param N desired number of panels
    */
    PanelVector make_N_polygonal_panels(const Eigen::VectorXd &x, const Eigen::VectorXd &y, unsigned N);

    /**
     * This function prints info on the list
     * of panels (min/max length, variability).
     *
     * @param panels list of panels
     */
    void print_panels_info(const PanelVector &panels);

    /**
    * This function returns true if p is inside
    * the polygonal mesh and false otherwise.
    *
    * @param mesh parametrized panel mesh
    * @param p point
    */
    bool inside_mesh(const ParametrizedMesh &mesh, const Eigen::Vector2d &p);

    /**
    * This function returns the distance from p
    * to the i-th panel of the polygonal mesh.
    *
    * @param mesh parametrized panel mesh
    * @param i panel index
    * @param p point
    * @param t0 pointer to the position of the projection of p onto the i-th panel
    */
    double panel_distance(const ParametrizedMesh &mesh, unsigned i, const Eigen::Vector2d &p, double *t0 = NULL);

    /**
    * This function returns the distance from p
    * to the polygonal mesh. If i and/or t0 are not
    * NULL, they will be filled with the index of
    * the nearest panel and the parameter value
    * corresponding to the nearest point on that
    * panel, respectively.
    *
    * @param mesh parametrized panel mesh
    * @param i panel index
    * @param p point
    * @param t0 pointer to the position of the projection of p onto the i-th panel
    */
    double distance(const ParametrizedMesh &mesh, const Eigen::Vector2d &p, unsigned *i = NULL, double *t0 = NULL);

    /**
     * This class defines a smooth scatterer obtained by
     * interpolating Akima spline through points read from file.
     */
    class SmoothScatterer {
    private:
        /**
         * The closed non-intersecting curve (x,y)(t) for t in [0,1].
         */
        ComplexSpline *spline;
        /**
         * The list of x and y coordinates of the vertices.
         * The first and last points are the same.
         */
        Eigen::VectorXd x, y;
        /**
         * List of (interpolated) vertices as 2D points.
         */
        std::vector<Eigen::Vector2d> vert;
        /**
         * The number of given points (n = x.size() - 1).
         */
        unsigned n;
        /**
         * Gaussian quadrature rule for composite integration.
         */
        QuadRule GaussQR;
        /**
         * The list of parameter values corresponding to vertices.
         */
        Eigen::VectorXd t_vert;
        /**
         * List of vertex indices.
         */
        std::vector<unsigned> ind;
        /**
         * Storage for distances from a point to vertices.
         */
        std::vector<double> dist;
        /**
         * Function used for computing the winding number.
         */
        std::function<double(double)> wf;
        /**
         * Function for computing distance from the scatterer
         * and the point the user provided.
         */
        std::function<double(double)> distf;
        Eigen::Vector2d user_point;
        /**
         * Function for computing the length of the curve.
         */
        std::function<double(double)> lf;
        /**
         * Tolerance for minima/root finding when computing distance.
         */
        double tol;
        /**
         * Scatterer arc-length.
         */
        double total_length_;
        /**
         * Helper functions.
         */
        Eigen::Vector2d curve_point(double t);
    public:
        /**
         * Contructor.
         *
         * @param fname name of the file containing the vertices
         * @param scale scale factor for vertex points
         */
        SmoothScatterer(const std::string& fname, double scale = 1.0);
        ~SmoothScatterer() { assert(spline != NULL); delete spline; }
        /**
         * This function creates a parametrized mesh
         * containing the given number of polynomial
         * panels of equal lengths.
         *
         * @param npanels number of panels
         * @return parametrized mesh
         */
        PanelVector panelize(unsigned npanels);
        /**
         * This function returns the distance from
         * the point p and the curve. If t is not NULL,
         * it is set to the value of the parameter
         * corresponding to the nearest point.
         *
         * @param p 2D point
         * @param t the parameter of the point on
         *          the curve which is nearest to p
         * @return the distance between p and the curve
         */
        double distance(const Eigen::Vector2d &p, double *t = NULL);
        /**
         * This function returns true iff the point p
         * is inside the curve. Otherwise, it returns
         * false. It computes winding number w for p and
         * the curve and returns true iff w > 0.5.
         *
         * @param p 2D point
         * @return true if p is inside, false if outside
         */
        bool is_inside(const Eigen::Vector2d &p);
        /**
         * This function returns the total arc-length of
         * the scatterer.
         *
         * @return scatterer length
         */
        double total_length();
        /**
         * This function samples the vertices on the
         * curve for drawing purposes.
         *
         * @param refsize reference size (diameter of the drawing area)
         * @param resolution number of points per refsize
         * @param vx x coordinates of the vertices
         * @param vy y coordinates of the vertices
         */
        void sample_vertices(double refsize, unsigned resolution, Eigen::VectorXd &vx, Eigen::VectorXd &vy);
    };
}
#endif // SCATTERER_HPP
