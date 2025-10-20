// MuscleTissueModel.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 10/30/21

#pragma once

#include "fsim/ElementBase.h"
#include "fsim/util/typedefs.h"
#include "fsim/ModelBase.h"
#include "MuscleTissueElement.h"

class MuscleTissueModel
{
public:
  /**
   * constructor for MuscleTissueModel
   * @param V  nV by 2 list of vertex positions (initial position in the 2D plane)
   * @param F  nF by 3 list of face indices
   * @param young_modulus  membrane's Young's modulus, controls the resistance to bending
   * @param poisson_ratio  membrane's Poisson's ratio
   * @param stretch
   */
  MuscleTissueModel(const Eigen::Ref<const fsim::Mat2<double>> V,
                    const Eigen::Ref<const fsim::Mat3<int>> F,
                    const Eigen::Ref<const Eigen::MatrixXd> Phi,
                    double young_modulus,
                    double poisson_ratio,
                    double stretch,
                    double sigma);

  MuscleTissueModel(const Eigen::Ref<const fsim::Mat2<double>> V,
                    const Eigen::Ref<const fsim::Mat3<int>> F,
                    const Eigen::Ref<const Eigen::VectorXd> theta0,
                    const Eigen::Ref<const Eigen::VectorXd> eta,
                    const Eigen::Ref<const Eigen::VectorXd> phi,
                    double young_modulus,
                    double poisson_ratio,
                    double stretch,
                    double sigma);
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
  Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> X) const;
  void gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const;
  // Eigen::VectorXd gradient_derivative_sensitivity(const Eigen::Ref<const Eigen::VectorXd> X, double stretch) const;
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

  Eigen::VectorXd I5(const Eigen::Ref<const Eigen::VectorXd> X) const;

  void setStretch(double stretch)
  {
    _stretch = stretch;
  }

  Eigen::VectorXd updatePhi(const Eigen::Ref<const Eigen::VectorXd> X);

  void setPoissonRatio(double poisson_ratio)
  {
    _nu = poisson_ratio;
    _lambda = _E * _nu / (1 - std::pow(_nu, 2));
    _mu = 0.5 * _E / (1 + _nu);
  }

  void setYoungModulus(double young_modulus)
  {
    _E = young_modulus;
    _lambda = _E * _nu / (1 - std::pow(_nu, 2));
    _mu = 0.5 * _E / (1 + _nu);
  }

  double _stretch;
  int nV;
  double _E;
  double _nu;
  double _lambda;
  double _mu;
  double _sigma;
  std::vector<MuscleTissueElement> _elements;
};
