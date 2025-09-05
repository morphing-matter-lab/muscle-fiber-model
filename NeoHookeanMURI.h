// NeoHookeanMURI.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 10/30/21

#pragma once

#include "fsim/ElementBase.h"
#include "fsim/util/typedefs.h"
#include "fsim/ModelBase.h"

class NeoHookeanMURIElement : public fsim::ElementBase<3>
{
public:
  /**
   * Constructor for the NeoHookeanMURI class
   * @param V  n by 2 list of vertex positions (each row is a vertex)
   * @param face  list of 3 indices, one per vertex of the triangle
   */
  NeoHookeanMURIElement(const Eigen::Ref<const fsim::Mat3<double>> V, const Eigen::Vector3i &face, double thickness);

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  energy of the triangle element for a given material model
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch, double mass) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  gradient of the energy (9 by 1 vector), derivatives are stacked in the order of the triangle indices
   */
  LocalVector gradient(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch, double mass) const;
  LocalVector gradient_derivative_sensitivity(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian matrix of the energy w.r.t. all 9 degrees of freedom of the triangle
   */
  LocalMatrix hessian(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const;

  /**
   * Computes the deformation gradient F
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return deformation gradient F
   */
  Eigen::Matrix<double, 3, 2> deformationGradient(const Eigen::Ref<const Eigen::VectorXd> X, double stretch) const;

  /**
   * Computes the Second Piola-Kirchhoff stress tensor S = \frac{\partial f}{\partial E} where E is the Green strain
   * Uses Voigt's notation to express it as a vector
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return second Piola-Kirchhoff stress
   */
  Eigen::Matrix2d stress(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const;
  Eigen::Matrix2d stress(const Eigen::Matrix2d &Cinv, double lambda, double mu) const;
  Eigen::Matrix3d elasticityTensor(const Eigen::Matrix2d &Cinv, double lambda, double mu) const;

  double coeff;
  Eigen::Matrix<double, 2, 2> _R;
};

class NeoHookeanMURI : public fsim::ModelBase<NeoHookeanMURIElement>
{
public:
  /**
   * constructor for NeoHookeanMURI
   * @param V  nV by 2 list of vertex positions (initial position in the 2D plane)
   * @param F  nF by 3 list of face indices
   * @param thickness  membrane's thickness
   * @param young_modulus  membrane's Young's modulus, controls the resistance to bending
   * @param poisson_ratio  membrane's Poisson's ratio
   * @param mass  membrane's mass (defaults to 0 to disable gravity)
   */
  NeoHookeanMURI(const Eigen::Ref<const fsim::Mat3<double>> V,
                 const Eigen::Ref<const fsim::Mat3<int>> F,
                 double thickness,
                 double young_modulus,
                 double poisson_ratio,
                 double stretch = 1,
                 double mass = 0)
    : _E(young_modulus), _nu(poisson_ratio), _mass(mass), _stretch(stretch), nV(V.rows())
{
  _lambda = _E * _nu / (1 - std::pow(_nu, 2));
  _mu = 0.5 * _E / (1 + _nu);

  this->_elements.reserve(F.rows());
  for(int i = 0; i < F.rows(); ++i)
    this->_elements.emplace_back(V, F.row(i), thickness);
}


  /**
   * energy function of this material model   f : \R^n -> \R
   * @param X  a flat vector stacking all degrees of freedom
   * @return  the energy of this model evaluated at X
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    return fsim::ModelBase<NeoHookeanMURIElement>::energy(X, _lambda, _mu, _stretch, _mass);
  }

  /**
   * gradient of the energy  \nabla f : \R^n -> \R^n
   * @param X  a flat vector stacking all degrees of freedom
   * @param Y  gradient (or sum of gradients) vector in which we will add the gradient of energy evaluated at X
   * @return Y
   */
  Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    using namespace Eigen;
    VectorXd Y = VectorXd::Zero(X.size());
    gradient(X, Y);
    return Y;
  }

  void gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y, double stretch) const
  {
    fsim::ModelBase<NeoHookeanMURIElement>::gradient(X, Y, _lambda, _mu, stretch, _mass);
  }

  Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> X, double stretch) const
  {
    using namespace Eigen;
    VectorXd Y = VectorXd::Zero(X.size());
    gradient(X, Y, stretch);
    return Y;
  }

  void gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
  {
    fsim::ModelBase<NeoHookeanMURIElement>::gradient(X, Y, _lambda, _mu, _stretch, _mass);
  }

  Eigen::VectorXd gradient_derivative_sensitivity(const Eigen::Ref<const Eigen::VectorXd> X, double stretch) const
  {
    using namespace Eigen;
    VectorXd Y = VectorXd::Zero(X.size());
    for(auto &element: this->_elements)
    {
      auto grad = element.gradient_derivative_sensitivity(X, _lambda, _mu, stretch);

      for(int j = 0; j < 3; ++j)
        Y.segment<3>(3 * element.idx(j)) += grad.template segment<3>(3 * j);
    }
    return Y;
  }

  /**
   * hessian of the energy  \nabla^2 f : \R^n -> \R^{n \times n}
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian of the energy stored in a sparse matrix representation
   */
  Eigen::SparseMatrix<double> hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    return fsim::ModelBase<NeoHookeanMURIElement>::hessian(X, _lambda, _mu, _stretch);
  }

  /**
   * (row, column, value) triplets used to build the sparse hessian matrix
   * @param X  a flat vector stacking all degrees of freedom
   * @return  all the triplets needed to build the hessian
   */
  std::vector<Eigen::Triplet<double>> hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    return fsim::ModelBase<NeoHookeanMURIElement>::hessianTriplets(X, _lambda, _mu, _stretch);
  }

  void setStretch(double stretch)
  {
    _stretch = stretch;
  }

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

  void setMass(double mass)
  {
    _mass = mass;
  }

  double _stretch;
  int nV;
  double _E;
  double _nu;
  double _lambda;
  double _mu;
  double _mass;
};
