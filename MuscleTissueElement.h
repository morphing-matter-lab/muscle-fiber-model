// MuscleTissueElement.h
//
// Author: David Jourdan (djourdan@berkelely.edu)
// Created: 2/6/2025

#pragma once

#include "fsim/ElementBase.h"
#include "fsim/util/typedefs.h"


class MuscleTissueElement : public fsim::ElementBase<3>
{
public:
  /**
   * Constructor for the NeoHookeanMURI class
   * @param V  n by 2 list of vertex positions (each row is a vertex)
   * @param face  list of 3 indices, one per vertex of the triangle
   */
  MuscleTissueElement(const Eigen::Ref<const fsim::Mat2<double>> V, const Eigen::Vector3i &face, const Eigen::Matrix2d &phi);
  MuscleTissueElement(const Eigen::Ref<const fsim::Mat2<double>> V, const Eigen::Vector3i &face, double mean_theta, double concentration_eta, double phi);

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  energy of the triangle element for a given material model
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch, double sigma) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  gradient of the energy (9 by 1 vector), derivatives are stacked in the order of the triangle indices
   */
  fsim::Vec<double, 6> gradient(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch, double sigma) const;
  // fsim::Vec<double, 6> gradient_derivative_sensitivity(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian matrix of the energy w.r.t. all 9 degrees of freedom of the triangle
   */
  fsim::Mat<double, 6, 6> hessian(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch, double sigma) const;

  /**
   * Computes the deformation gradient F
   * @param X  a flat vector stacking all degrees of freedom
   * @return deformation gradient F
   */
  Eigen::Matrix2d deformationGradient(const Eigen::Ref<const Eigen::VectorXd> X, double stretch) const;

  /**
   * Computes the Second Piola-Kirchhoff stress tensor S = \frac{\partial f}{\partial E} where E is the Green strain
   * Uses Voigt's notation to express it as a vector
   * @param X  a flat vector stacking all degrees of freedom
   * @return second Piola-Kirchhoff stress
   */
  Eigen::Matrix2d stress(const Eigen::Matrix2d &F, double lambda, double mu, double stretch, double sigma) const;
  Eigen::Matrix3d elasticityTensor(const Eigen::Matrix2d &F, double lambda, double mu, double stretch, double sigma) const;

  double area;
  double coeff;
  Eigen::Matrix2d R;
  Eigen::Matrix2d Phi;
};