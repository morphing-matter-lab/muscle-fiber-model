// ActinBundle.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 01/19/20

#pragma once

#include "fsim/ModelBase.h"
#include "fsim/util/typedefs.h"
#include "FiberElement.h"

/**
 * template class for isotropic membrane models (e.g. StVK, neohookean...)
 */
class ActinBundle : public fsim::ModelBase<FiberElement>
{
public:
  /**
   * constructor for ActinBundle
   * @param V  nV by 2 list of vertex positions (initial position in the 2D plane)
   * @param F  nF by 3 list of face indices
   * @param Phi  nF by 6 list of F-actin directional volume fractions
   * @param thickness  membrane's thickness
   * @param e0
   * @param e1
   */
  ActinBundle(const Eigen::Ref<const fsim::Mat3<double>> V,
              const Eigen::Ref<const fsim::Mat3<int>> F,
              const Eigen::Ref<const Eigen::MatrixXd> Phi,
              double thickness,
              double sigma_max,
              double e0,
              double e1);

  /**
   * energy function of this material model   f : \R^n -> \R
   * @param X  a flat vector stacking all degrees of freedom
   * @return  the energy of this model evaluated at X
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * gradient of the energy  \nabla f : \R^n -> \R^n
   * @param X  a flat vector stacking all degrees of freedom
   * @param Y  gradient (or sum of gradients) vector in which we will add the gradient of energy evaluated at X
   * @return Y
   */
  void gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const;
  Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * hessian of the energy  \nabla^2 f : \R^n -> \R^{n \times n}
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian of the energy stored in a sparse matrix representation
   */
  Eigen::SparseMatrix<double> hessian(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * (row, column, value) triplets used to build the sparse hessian matrix
   * @param X  a flat vector stacking all degrees of freedom
   * @return  all the triplets needed to build the hessian
   */
  std::vector<Eigen::Triplet<double>> hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const;

  // number of degrees of freedom
  int nbDOFs() const { return 3 * nV; }

private:
  int nV, nF;
  double e0;
  double e1;
};
