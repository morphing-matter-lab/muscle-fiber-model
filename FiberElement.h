// FiberElement.h
//
// Author: David Jourdan (djourdan@berkelely.edu)
// Created: 2/6/2025

#pragma once

#include "fsim/ElementBase.h"
#include "fsim/util/typedefs.h"

class FiberElement : public fsim::ElementBase<3>
{
public:
  /**
   * Constructor for the FiberElement class
   * @param V  n by 2 list of vertex positions (each row is a vertex)
   * @param face  list of 3 indices, one per vertex of the triangle
   */
  FiberElement(const Eigen::Ref<const fsim::Mat3<double>> V, const Eigen::Vector3i &face, const Eigen::VectorXd &phi, double thickness, double sigma_max);

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  energy of the triangle element for a given material model
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X, double e0, double e1) const;
  double energy(double e, double e0, double e1) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  gradient of the energy (9 by 1 vector), derivatives are stacked in the order of the triangle indices
   */
  LocalVector gradient(const Eigen::Ref<const Eigen::VectorXd> X, double e0, double e1) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian matrix of the energy w.r.t. all 9 degrees of freedom of the triangle
   */
  LocalMatrix hessian(const Eigen::Ref<const Eigen::VectorXd> X, double e0, double e1) const;

  /**
   * Computes the Cauchy-Green deformation tensor C = F^T F where F is the deformation gradient
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return Cauchy-Green deformation tensor
   */
  Eigen::Matrix<double, 3, 2> deformationGradient(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * Computes the Second Piola-Kirchhoff stress tensor S = \frac{\partial f}{\partial E} where E is the Green strain
   * Uses Voigt's notation to express it as a vector
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return second Piola-Kirchhoff stress
   */
  Eigen::Matrix2d stress(const Eigen::Matrix<double, 3, 2> &F, double e0, double e1) const;
  double stress(double e, double e0, double e1) const;

  double strain(const Eigen::Matrix<double, 3, 2> &F, const Eigen::Vector2d &u) const;

  Eigen::Matrix3d elasticityTensor(const Eigen::Matrix<double, 3, 2> &F, double e0, double e1) const;
  double stressDeriv(double e, double e0, double e1) const;

  double coeff;
  Eigen::Matrix2d R;
  Eigen::VectorXd phi;
};
