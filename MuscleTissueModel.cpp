// NeoHookeanMURI.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 10/30/21

#include "MuscleTissueModel.h"

#include <cmath>
#include <fsim/util/geometry.h>
#include <TinyAD/Utils/HessianProjection.hh>

using namespace fsim;

MuscleTissueModel::MuscleTissueModel(const Eigen::Ref<const fsim::Mat2<double>> V,
                                     const Eigen::Ref<const fsim::Mat3<int>> F,
                                     const Eigen::Ref<const Eigen::MatrixXd> Phi,
                                     double young_modulus,
                                     double poisson_ratio,
                                     double stretch,
                                     double sigma)
    : _E(young_modulus), _nu(poisson_ratio), _stretch(stretch), nV(V.rows()), _sigma(sigma)
{
  _lambda = _E * _nu / (1 - std::pow(_nu, 2));
  _mu = 0.5 * _E / (1 + _nu);

  _elements.reserve(F.rows());
  for (int i = 0; i < F.rows(); ++i)
    _elements.emplace_back(V, F.row(i), Phi.row(i));
}

MuscleTissueModel::MuscleTissueModel(const Eigen::Ref<const fsim::Mat2<double>> V,
                                     const Eigen::Ref<const fsim::Mat3<int>> F,
                                     const Eigen::Ref<const Eigen::VectorXd> theta0,
                                     const Eigen::Ref<const Eigen::VectorXd> eta,
                                     const Eigen::Ref<const Eigen::VectorXd> phi,
                                     double young_modulus,
                                     double poisson_ratio,
                                     double stretch,
                                     double sigma)
    : _E(young_modulus), _nu(poisson_ratio), _stretch(stretch), nV(V.rows()), _sigma(sigma)
{
  _lambda = _E * _nu / (1 - std::pow(_nu, 2));
  _mu = 0.5 * _E / (1 + _nu);

  _elements.reserve(F.rows());
  for (int i = 0; i < F.rows(); ++i)
    _elements.emplace_back(V, F.row(i), theta0(i), eta(i), phi(i));
}

Eigen::VectorXd MuscleTissueModel::updatePhi(const Eigen::Ref<const Eigen::VectorXd> X)
{
  using namespace Eigen;

  VectorXd thetas(_elements.size());

  for (int i = 0; i < _elements.size(); ++i)
  {
    Matrix2d F = _elements[i].deformationGradient(X, _stretch);
    Matrix2d R = F.jacobiSvd(ComputeFullU | ComputeFullV).matrixU() * F.jacobiSvd(ComputeFullU | ComputeFullV).matrixV().transpose();
    _elements[i].Phi = R.transpose() * _elements[i].Phi * R;

    thetas(i) = -std::atan2(R(1,0), R(0,0));
  }

  return thetas;
}

double MuscleTissueModel::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  double result = 0;
  for (auto &element : _elements)
  {
    result += element.energy(X, _lambda, _mu, _stretch, _sigma);
  }
  return result;
}

Eigen::VectorXd MuscleTissueModel::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  VectorXd Y = VectorXd::Zero(X.size());
  gradient(X, Y);
  return Y;
}

void MuscleTissueModel::gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
{
  for (auto &element : _elements)
  {
    fsim::Vec<double, 6> grad = element.gradient(X, _lambda, _mu, _stretch, _sigma);

    for (int j = 0; j < 3; ++j)
      Y.segment<2>(2 * element.idx(j)) += grad.segment<2>(2 * j);
  }
}

// Eigen::VectorXd MuscleTissueModel::gradient_derivative_sensitivity(const Eigen::Ref<const Eigen::VectorXd> X, double stretch) const
// {
//   using namespace Eigen;
//   VectorXd Y = VectorXd::Zero(X.size());
//   for(auto &element: _elements)
//   {
//     fsim::Vec<double, 6> grad = element.gradient_derivative_sensitivity(X, _lambda, _mu, stretch);

//     for(int j = 0; j < 3; ++j)
//       Y.segment<2>(2 * element.idx(j)) += grad.segment<2>(2 * j);
//   }
//   return Y;
// }

Eigen::SparseMatrix<double> MuscleTissueModel::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  std::vector<Triplet<double>> triplets = hessianTriplets(X);

  SparseMatrix<double> hess(X.size(), X.size());
  hess.setFromTriplets(triplets.begin(), triplets.end());
  return hess;
}

std::vector<Eigen::Triplet<double>> MuscleTissueModel::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  constexpr int n = 6 * (6 + 3) / 2;
  std::vector<Triplet<double>> triplets(n * _elements.size());

#pragma omp parallel for if (_elements.size() > 1000)
  for (int i = 0; i < _elements.size(); ++i)
  {
    auto &e = _elements[i];
    auto hess = e.hessian(X, _lambda, _mu, _stretch, _sigma);

    int id = 0;
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        if (e.idx(j) <= e.idx(k))
          for (int l = 0; l < 2; ++l)
            for (int m = 0; m < 2; ++m)
              triplets[n * i + id++] = Triplet<double>(2 * e.idx(j) + l, 2 * e.idx(k) + m, hess(2 * j + l, 2 * k + m));
  }

  return triplets;
}

Eigen::VectorXd  MuscleTissueModel::I5(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen; 
  VectorXd Y(_elements.size());
  for (int i = 0; i < _elements.size(); ++i)
  {
    Matrix2d F = _elements[i].deformationGradient(X, _stretch);
    Matrix2d C = F.transpose() * F;
    Y(i) = (C * _elements[i].Phi).trace();
  }

  return Y;
}