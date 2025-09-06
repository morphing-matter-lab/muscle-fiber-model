// NeoHookeanMURI.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 10/30/21

#include "NeoHookeanMURI.h"

#include <cmath>
#include <fsim/util/geometry.h>
#include <TinyAD/Utils/HessianProjection.hh>

using namespace fsim;

NeoHookeanMURIElement::NeoHookeanMURIElement(const Eigen::Ref<const fsim::Mat2<double>> V, const Eigen::Vector3i &E)
{
  using namespace Eigen;

  idx = E;

  Vector2d e1 = V.row(E(0)) - V.row(E(2));
  Vector2d e2 = V.row(E(1)) - V.row(E(2));

  _R.col(0) << e1;
  _R.col(1) << e2;
  _R = _R.inverse().eval();

  coeff = 0.5 * (e1(0) * e2(1) - e2(0) * e1(1));
}

Eigen::Matrix2d NeoHookeanMURIElement::deformationGradient(const Eigen::Ref<const Eigen::VectorXd> X, double stretch) const
{
  using namespace Eigen;

  Matrix2d Ds;
  Ds.col(0) = X.segment<2>(2 * idx(0)) - X.segment<2>(2 * idx(2));
  Ds.col(1) = X.segment<2>(2 * idx(1)) - X.segment<2>(2 * idx(2));
  Matrix2d F = stretch * Ds * _R;

  return F;
}

Eigen::Matrix2d NeoHookeanMURIElement::stress(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const
{
  using namespace Eigen;

  Matrix2d F = deformationGradient(X, stretch);
  Matrix2d C = F.transpose() * F;
  double lnJ = log(C.determinant()) / 2;

  return mu * Matrix2d::Identity() + (lambda * lnJ - mu) * C.inverse();
}

Eigen::Matrix2d NeoHookeanMURIElement::stress(const Eigen::Matrix2d &Cinv, double lambda, double mu) const
{
  using namespace Eigen;

  double lnJ = -log(Cinv.determinant()) / 2;
  return mu * Matrix2d::Identity() + (lambda * lnJ - mu) * Cinv;
}

Eigen::Matrix3d NeoHookeanMURIElement::elasticityTensor(const Eigen::Matrix2d &Cinv, double lambda, double mu) const
{
  using namespace Eigen;

  double lnJ = -log(Cinv.determinant()) / 2;

  Matrix3d _C;
  _C << Cinv(0, 0) * Cinv(0, 0), Cinv(0, 1) * Cinv(0, 1), Cinv(0, 0) * Cinv(0, 1), 
        Cinv(1, 0) * Cinv(1, 0), Cinv(1, 1) * Cinv(1, 1), Cinv(0, 1) * Cinv(1, 1), 
        Cinv(0, 1) * Cinv(0, 0), Cinv(0, 1) * Cinv(1, 1), (Cinv(0, 0) * Cinv(1, 1) + Cinv(0, 1) * Cinv(0, 1)) / 2;
  _C *= 2 * (mu - lambda * lnJ);
  _C += lambda * Vector3d(Cinv(0, 0), Cinv(1, 1), Cinv(0, 1)) * RowVector3d(Cinv(0, 0), Cinv(1, 1), Cinv(0, 1));

  return _C;
}

double NeoHookeanMURIElement::energy(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const
{
  using namespace Eigen;

  Matrix2d F = deformationGradient(X, stretch);
  Matrix2d C = F.transpose() * F;
  double lnJ = log(C.determinant()) / 2;

  return coeff * (mu / 2 * (C.trace() - 2 - 2 * lnJ) + lambda / 2 * pow(lnJ, 2));
}

fsim::Vec<double, 6>
NeoHookeanMURIElement::gradient(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const
{
  using namespace Eigen;

  Matrix2d F = deformationGradient(X, stretch);
  Matrix2d Cinv = (F.transpose() * F).inverse();
  Matrix2d S = stress(Cinv, lambda, mu);

  Matrix2d H = stretch * coeff * F * S * _R.transpose();

  Vec<double, 6> grad;
  grad.segment<2>(0) = H.col(0);
  grad.segment<2>(2) = H.col(1);
  grad.segment<2>(4) = -H.col(0) - H.col(1);

  return grad;
}

fsim::Vec<double, 6>
NeoHookeanMURIElement::gradient_derivative_sensitivity(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const
{
  using namespace Eigen;

  Matrix2d F = deformationGradient(X, stretch);
  Matrix2d Cinv = (F.transpose() * F).inverse();
  Matrix2d S = stress(Cinv, lambda, mu);

  Matrix2d H = coeff * (2 * mu * F + 3 * lambda / stretch * F * Cinv) * _R.transpose();

  fsim::Vec<double, 6> res;
  res.segment<2>(0) = H.col(0);
  res.segment<2>(2) = H.col(1);
  res.segment<2>(4) = -H.col(0) - H.col(1);

  return res;
}

fsim::Mat<double, 6, 6>
NeoHookeanMURIElement::hessian(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const
{
  using namespace Eigen;

  Matrix2d F = deformationGradient(X, stretch);

  Matrix2d Cinv = (F.transpose() * F).inverse();
  Matrix2d S = stress(Cinv, lambda, mu);
  Matrix3d _C = elasticityTensor(Cinv, lambda, mu);

  Matrix2d A = _C(0, 0) * F.col(0) * F.col(0).transpose() + S(0, 0) * Matrix2d::Identity() +
               _C(2, 2) * F.col(1) * F.col(1).transpose() + 2 * _C(0, 2) * sym(F.col(0) * F.col(1).transpose());
  Matrix2d B = _C(1, 1) * F.col(1) * F.col(1).transpose() + S(1, 1) * Matrix2d::Identity() +
               _C(2, 2) * F.col(0) * F.col(0).transpose() + 2 * _C(1, 2) * sym(F.col(0) * F.col(1).transpose());
  Matrix2d C = _C(0, 1) * F.col(0) * F.col(1).transpose() + S(0, 1) * Matrix2d::Identity() +
               _C(2, 2) * F.col(1) * F.col(0).transpose() + _C(0, 2) * F.col(0) * F.col(0).transpose() +
               _C(1, 2) * F.col(1) * F.col(1).transpose();

  Matrix<double, 6, 6> hess;

  for(int i = 0; i < 2; ++i)
    for(int j = i; j < 2; ++j)
      hess.block<2, 2>(2 * i, 2 * j) =
          _R(i, 0) * _R(j, 0) * A + _R(i, 1) * _R(j, 1) * B + _R(i, 0) * _R(j, 1) * C + _R(i, 1) * _R(j, 0) * C.transpose();

  hess.block<2, 2>(2, 0) = hess.block<2, 2>(0, 2).transpose();
  hess.block<4, 2>(0, 4) = -hess.block<4, 2>(0, 0) - hess.block<4, 2>(0, 2);
  hess.block<2, 4>(4, 0) = hess.block<4, 2>(0, 4).transpose();
  hess.block<2, 2>(4, 4) = -hess.block<2, 2>(0, 4) - hess.block<2, 2>(2, 4);

  TinyAD::project_positive_definite<6, double>(hess, 1e-9);

  return coeff * stretch * stretch * hess;
}

