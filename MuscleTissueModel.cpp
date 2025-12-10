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
                                     const std::vector<int> &post_indices,
                                     double young_modulus,
                                     double poisson_ratio,
                                     double stretch,
                                     double sigma,
                                     double k_post)
    : _E(young_modulus), _nu(poisson_ratio), _stretch(stretch), nV(V.rows()), _sigma(sigma), _kpost(k_post)
{
  _lambda = _E * _nu / (1 - std::pow(_nu, 2));
  _mu = 0.5 * _E / (1 + _nu);

  _elements.reserve(F.rows());
  for (int i = 0; i < F.rows(); ++i)
    _elements.emplace_back(V, F.row(i), Phi.block<2, 2>(2 * i, 0));

  for (int k : post_indices)
    _post_anchors.emplace_back(k, V(k, 0), V(k, 1));
}

MuscleTissueModel::MuscleTissueModel(const Eigen::Ref<const fsim::Mat2<double>> V,
                                     const Eigen::Ref<const fsim::Mat3<int>> F,
                                     const Eigen::Ref<const Eigen::VectorXd> theta0,
                                     const Eigen::Ref<const Eigen::VectorXd> eta,
                                     const Eigen::Ref<const Eigen::VectorXd> phi,
                                     const std::vector<int> &post_indices,
                                     double young_modulus,
                                     double poisson_ratio,
                                     double stretch,
                                     double sigma,
                                     double k_post)
    : _E(young_modulus), _nu(poisson_ratio), _stretch(stretch), nV(V.rows()), _sigma(sigma), _kpost(k_post)
{
  _lambda = _E * _nu / (1 - std::pow(_nu, 2));
  _mu = 0.5 * _E / (1 + _nu);

  _elements.reserve(F.rows());
  for (int i = 0; i < F.rows(); ++i)
    _elements.emplace_back(V, F.row(i), theta0(i), eta(i), phi(i));

  for (int j : post_indices)
    _post_anchors.emplace_back(j, V(j, 0), V(j, 1));
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

    thetas(i) = -std::atan2(R(1, 0), R(0, 0));
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
  for (auto [j, x, y] : _post_anchors)
  {
    result += 0.5 * _kpost * ((X(2 * j) - x) * (X(2 * j) - x) + (X(2 * j + 1) - y) * (X(2 *  + 1) - y));
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
  for (auto [j, x, y] : _post_anchors)
  {
    Y(2 * j) += _kpost * (X(2 * j) - x);
    Y(2 * j + 1) += _kpost * (X(2 * j + 1) - y);
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

  for (auto [j, x, y] : _post_anchors)
  {
    triplets.emplace_back(2 * j, 2 * j, _kpost);
    triplets.emplace_back(2 * j + 1, 2 * j + 1, _kpost);
  }

  return triplets;
}

Eigen::VectorXd MuscleTissueModel::I5(const Eigen::Ref<const Eigen::VectorXd> X) const
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

Eigen::MatrixXd MuscleTissueModel::phi_ODE(const Eigen::Ref<const Eigen::VectorXd> X, double k0, double k1, double kd, double dt, int n)
{
  using namespace Eigen;

  const double phi_f = 0.7;

#pragma omp parallel for
  for (int i = 0; i < _elements.size(); ++i)
  {
    Matrix2d F = _elements[i].deformationGradient(X, _stretch);
    Matrix2d C = F.transpose() * F;
    for (int k = 0; k < n; ++k)
    {
      Matrix2d Phi = _elements[i].Phi;
      double phi_m = 0.05 - Phi.trace();

      // compute stress
      double I5 = (C * Phi).trace();
      Matrix2d stress;

      if (I5 < 1e-8)
        stress = Matrix2d::Zero();
      else
        stress = _sigma * (1 - std::sqrt(Phi.trace() / I5)) * F * Phi * F.transpose();

      Matrix2d dPhi_dt = phi_m / phi_f * (k0 / 2 * Matrix2d::Identity() + k1 * stress) - kd * Phi;

      _elements[i].Phi += dt * dPhi_dt;
    }
  }

  MatrixXd Phis(2 * _elements.size(), 2);
  for (int i = 0; i < _elements.size(); ++i)
    Phis.block<2, 2>(2 * i, 0) = _elements[i].Phi;

  return Phis;
}

Eigen::MatrixXd MuscleTissueModel::theta0(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  MatrixXd dirs(_elements.size(), 2);
  VectorXd thetas(_elements.size());
  for (int i = 0; i < _elements.size(); ++i)
  {
    Matrix2d F = _elements[i].deformationGradient(X, _stretch);
    // Matrix2d C = F.transpose() * F;
    // Matrix2d V = C.jacobiSvd(ComputeFullU | ComputeFullV).matrixV();
    // Matrix2d U = C.jacobiSvd(ComputeFullU | ComputeFullV).matrixU();
    // std::cout << V.determinant() << " " << U.determinant() << " " << (U * V).determinant() << "\n";
    // dirs.row(i) = (U * V.transpose()).col(0);

    double y1 = F(1, 0) + F(0, 1);
    double x1 = F(0, 0) - F(1, 1);
    double y2 = F(1, 0) - F(0, 1);
    double x2 = F(0, 0) + F(1, 1);

    if (std::abs(y1) < std::numeric_limits<double>::min() && std::abs(x1) < std::numeric_limits<double>::min())
    {
      // U = Matrix2d::Identity();
      // double s0 = std::hypot(F(0, 0), F(0, 1));
      // double s1 = std::hypot(F(1, 0), F(1, 1));
      // if(std::abs(s0) < std::numeric_limits<double>::min())
      //     V = Matrix2d::Identity();
      // else
      //     V = DiagonalMatrix<double, 2>(1 / s0, 1 / s1) * F;
      thetas(i) = 0;
      dirs.row(i) << 1, 0;
    }
    else if (std::abs(y2) < std::numeric_limits<double>::min() && std::abs(x2) < std::numeric_limits<double>::min())
    {
      // U = Matrix2d::Identity();
      // double s0 = std::hypot(F(0, 0), F(0, 1));
      // double s1 = -std::hypot(F(1, 0), F(1, 1));
      // V = DiagonalMatrix<double, 2>(1 / s0, 1 / s1) * F;
      thetas(i) = 0;
      dirs.row(i) << 1, 0;
    }
    else
    {
      double a1 = std::atan2(y1, x1);
      double a2 = std::atan2(y2, x2);

      // U = Rotation2D<double>((a2 + a1) / 2);
      // V = Rotation2D<double>((a2 - a1) / 2);
      thetas(i) = (a2 - a1) / 2;
      dirs.row(i) << std::cos(thetas(i)), std::sin(thetas(i));
    }
  }

  return dirs;
}