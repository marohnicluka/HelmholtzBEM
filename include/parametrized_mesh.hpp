/**
 * \file parametrized_mesh.hpp
 * \brief This file declares a class for representing a mesh comprising
 *        of panels in the form of parametrized curves.
 *
 *  This File is a part of the 2D-Parametric BEM package
 */

#ifndef PARAMETRIZEDMESHHPP
#define PARAMETRIZEDMESHHPP

#include "abstract_parametrized_curve.hpp"

typedef enum MeshType {
  MESH_TYPE_GENERAL,
  MESH_TYPE_POLYGONAL,
  MESH_TYPE_SMOOTH
} mesh_type;

/**
 * \class ParametrizedMesh
 * \brief This class represents a mesh which is comprised of panels in the
 *        form of parametrized curves. It is used in assembly of Galerkin
 *        matrices using the parametric BEM approach. It stores the panels
 *        using PanelVector and enforces additional constraints which
 *        require the end point of a panel to be the starting point of the
 *        next, such that the panels form a curved polygon.
 */
class ParametrizedMesh {
public:
  /**
   * Constructor using a PanelVector object which contains the component
   * panels of the mesh in the form of parametrized curves.
   */
  ParametrizedMesh(PanelVector panels, mesh_type meshtype = MESH_TYPE_GENERAL, void *data = NULL);

  /**
   * This function is used for retrieving a PanelVector containing all the
   * parametrized curve panels in the parametrized mesh.
   *
   * @return A PanelVector containing all the parametrized panels in the mesh
   */
  const PanelVector &getPanels() const;

  /**
   * This function is used for getting the number of panels in the
   * parametrized mesh
   *
   * @return number of panels in the mesh
   */
  unsigned int getNumPanels() const;

  /**
   * This function is used for getting the ith vertex in the parametrized mesh
   *
   * @return ith vertex in the mesh as Eigen::Vector2d
   */
  Eigen::Vector2d getVertex(unsigned int i) const;

  /**
   * This function is used for getting the total length of the
   * parametrized mesh
   *
   * @return the total length of the mesh
   */
  double getTotalLength() const;

  /**
   * This function is used for getting the length of the
   * largest panel in the mesh
   *
   * @return the length of the largest panel
   */
  double maxPanelLength() const;

  /**
   * This function is used for getting the panel at point s in [0,1]
   *
   * @param s length
   * @return the panel at s
   */
  const AbstractParametrizedCurve &getPanel(double s, double &t) const;

  double getCSum(unsigned int i) const { return c_sums_[i]; }

  /**
   * This function is used for getting the split value for the mesh. If split
   * is non zero, it indicates the position where the second boundary begins
   * in the mesh object; the domain is annular. A zero value indicates there is
   * only one boundary in the mesh object
   *
   * @return The position in the mesh where the second boundary begins
   */
  unsigned getSplit() const { return split_; }

  /**
   * This function returns true iff the mesh is polygonal.
   */
  bool isPolygonal() const;

  /**
   * This function returns true iff the mesh is smooth (cubic spline).
   */
  bool isSmooth() const { return type_ == MESH_TYPE_SMOOTH; }

  /**
   * This function returns the additional data.
   */
  void *getData() const { return data_; }

  //void addPanels(const PanelVector& panels) {
  //  panels_.insert()
  //}

private:
  /**
   * Private const field for the PanelVector of the mesh
   */
  const PanelVector panels_;
  /**
   * Private unsigned field used to distinguish one boundary from another in the
   * mesh (annular domain). Indicates the starting position of second boundary.
   */
  unsigned split_;
  /**
   * Private field containing the cummulative sums of panel lengths.
   */
  std::vector<double> c_sums_;
  /**
   * This private field is the mesh type.
   */
  mesh_type type_;
  /**
   * Private field containing pointer to additional data.
   */
  void *data_;
}; // class ParametrizedMesh

#endif // PARAMETRIZEDMESHHPP
