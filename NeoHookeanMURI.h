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
  NeoHookeanMURIElement(const Eigen::Ref<const fsim::Mat2<double>> V, const Eigen::Vector3i &face);

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  energy of the triangle element for a given material model
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  gradient of the energy (9 by 1 vector), derivatives are stacked in the order of the triangle indices
   */
  fsim::Vec<double, 6> gradient(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const;
  fsim::Vec<double, 6> gradient_derivative_sensitivity(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian matrix of the energy w.r.t. all 9 degrees of freedom of the triangle
   */
  fsim::Mat<double, 6, 6> hessian(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const;

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
  Eigen::Matrix2d stress(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const;
  Eigen::Matrix2d stress(const Eigen::Matrix2d &Cinv, double lambda, double mu) const;
  Eigen::Matrix3d elasticityTensor(const Eigen::Matrix2d &Cinv, double lambda, double mu) const;

  double coeff;
  Eigen::Matrix2d _R;
};

class NeoHookeanMURI
{
public:
  /**
   * constructor for NeoHookeanMURI
   * @param V  nV by 2 list of vertex positions (initial position in the 2D plane)
   * @param F  nF by 3 list of face indices
   * @param young_modulus  membrane's Young's modulus, controls the resistance to bending
   * @param poisson_ratio  membrane's Poisson's ratio
   * @param stretch
   */
  NeoHookeanMURI(const Eigen::Ref<const fsim::Mat2<double>> V,
                 const Eigen::Ref<const fsim::Mat3<int>> F,
                 double young_modulus,
                 double poisson_ratio,
                 double stretch = 1)
    : _E(young_modulus), _nu(poisson_ratio), _stretch(stretch), nV(V.rows())
  {
    _lambda = _E * _nu / (1 - std::pow(_nu, 2));
    _mu = 0.5 * _E / (1 + _nu);

    _elements.reserve(F.rows());
    for(int i = 0; i < F.rows(); ++i)
      _elements.emplace_back(V, F.row(i));
  }


  /**
   * energy function of this material model   f : \R^n -> \R
   * @param X  a flat vector stacking all degrees of freedom
   * @return  the energy of this model evaluated at X
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    double result = 0;
    for(auto &element: _elements)
    {
      result += element.energy(X, _lambda, _mu, _stretch);
    }
    return result;
  }

  /**
   * gradient of the energy  \nabla f : \R^n -> \R^n
   * @param X  a flat vector stacking all degrees of freedom
   * @param Y  gradient (or sum of gradients) vector in which we will add the gradient of energy evaluated at X
   * @return Y
   */
  Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> X, double stretch) const
  {
    using namespace Eigen;

    VectorXd Y = VectorXd::Zero(X.size());
    for(auto &element: _elements)
    {
      auto grad = element.gradient(X, _lambda, _mu, stretch);

      for(int j = 0; j < 3; ++j)
        Y.segment<2>(2 * element.idx(j)) += grad.template segment<2>(2 * j);
    }
    return Y;
  }

  Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    return gradient(X, _stretch);
  }

  Eigen::VectorXd gradient_derivative_sensitivity(const Eigen::Ref<const Eigen::VectorXd> X, double stretch) const
  {
    using namespace Eigen;
    VectorXd Y = VectorXd::Zero(X.size());
    for(auto &element: _elements)
    {
      auto grad = element.gradient_derivative_sensitivity(X, _lambda, _mu, stretch);

      for(int j = 0; j < 3; ++j)
        Y.segment<2>(2 * element.idx(j)) += grad.template segment<2>(2 * j);
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
    using namespace Eigen;

    std::vector<Triplet<double>> triplets = hessianTriplets(X);

    SparseMatrix<double> hess(X.size(), X.size());
    hess.setFromTriplets(triplets.begin(), triplets.end());
    return hess;
  }

  /**
   * (row, column, value) triplets used to build the sparse hessian matrix
   * @param X  a flat vector stacking all degrees of freedom
   * @return  all the triplets needed to build the hessian
   */
  std::vector<Eigen::Triplet<double>> hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
  {
    using namespace Eigen;

    constexpr int n = 6 * (6 + 3) / 2;
    std::vector<Triplet<double>> triplets(n * _elements.size());

  #pragma omp parallel for if(_elements.size() > 1000)
    for(int i = 0; i < _elements.size(); ++i)
    {
      auto &e = _elements[i];
      auto hess = e.hessian(X, _lambda, _mu, _stretch);

      int id = 0;
      for(int j = 0; j < 3; ++j)
        for(int k = 0; k < 3; ++k)
          if(e.idx(j) <= e.idx(k))
            for(int l = 0; l < 2; ++l)
              for(int m = 0; m < 2; ++m)
                triplets[n * i + id++] = Triplet<double>(2 * e.idx(j) + l, 2 * e.idx(k) + m, hess(2 * j + l, 2 * k + m));
    }

    return triplets;
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

  double _stretch;
  int nV;
  double _E;
  double _nu;
  double _lambda;
  double _mu;
  std::vector<NeoHookeanMURIElement> _elements;
};
