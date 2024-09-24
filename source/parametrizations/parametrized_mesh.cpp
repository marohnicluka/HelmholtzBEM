/**
 * \file parametrized_mesh.cpp
 * \brief This file defines the class for representing a mesh comprised
 *        of panels represented by parametrized curves.
 *
 *  This File is a part of the 2D-Parametric BEM package
 */

#include "parametrized_mesh.hpp"
#include <limits>
#include <Eigen/Dense>

ParametrizedMesh::ParametrizedMesh(PanelVector panels, mesh_type meshtype, void *data)
  : panels_(panels), type_(meshtype), data_(data)
{
  // std::cout << "ParametrizedMesh constructor called!" << std::endl;
  unsigned int N = getNumPanels();
  c_sums_.resize(N);
  // Determining the split value by looping over the panels in the mesh. Non
  // zero split indicates two distinct boundaries in the mesh object. Used when
  // dealing with an annular domain
  for (unsigned int i = 0; i < N; ++i) {
    c_sums_[i] = (i > 0 ? c_sums_[i - 1] : 0.) + panels[i]->length();
    if ((panels[0]->operator()(-1.) - panels[i]->operator()(1.)).norm() <
      10*std::numeric_limits<double>::epsilon()) {
      // std::cout << "Break in continuity at position "<< i << std::endl;
      split_ = (i + 1) % N;
      }
  }
  // std::cout << "split : " << split_ << std::endl;
}

const PanelVector &ParametrizedMesh::getPanels() const {
  // Returning a copy of the stored panels
  return panels_;
}

unsigned int ParametrizedMesh::getNumPanels() const {
  // Returning the size of the PanelVector panels_
  return panels_.size();
}

Eigen::Vector2d ParametrizedMesh::getVertex(unsigned int i) const {
  assert(i < getNumPanels());                               // Asserting requested index is within limits
  return panels_[i]->operator()(-1);
}

double ParametrizedMesh::getTotalLength() const {
  return c_sums_.back();
}

const AbstractParametrizedCurve & ParametrizedMesh::getPanel(double s, double &t) const {
  double len = s * c_sums_.back();
  unsigned int i = std::upper_bound(c_sums_.cbegin(), c_sums_.cend(), len) - c_sums_.cbegin();
  const auto &panel = *panels_[i % getNumPanels()];
  t = (len - (i > 0 ? c_sums_[i - 1] : 0.)) / panel.length();
  return panel;
}

double ParametrizedMesh::maxPanelLength() const {
  double ret = 0., L;
  for (const auto &p : panels_) {
    L = p->length();
    if (L > ret)
      ret = L;
  }
  return ret;
}

bool ParametrizedMesh::isPolygonal() const {
  if (type_ == MESH_TYPE_POLYGONAL)
    return true;
  else if (type_ == MESH_TYPE_GENERAL) {
    for (const auto &p : panels_) {
      if (!p->isLineSegment())
        return false;
    }
    return true;
  }
  return false;
}
