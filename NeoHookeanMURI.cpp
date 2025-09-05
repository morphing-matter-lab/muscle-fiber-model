// NeoHookeanMURI.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 10/30/21

#include "NeoHookeanMURI.h"
#include "fsim/util/geometry.h"

#include <cmath>

using namespace fsim;

NeoHookeanMURIElement::NeoHookeanMURIElement(const Eigen::Ref<const fsim::Mat3<double>> V, const Eigen::Vector3i &E, double thickness)
{
  using namespace Eigen;

  idx = E;

  Vector3d e1 = V.row(E(0)) - V.row(E(2));
  Vector3d e2 = V.row(E(1)) - V.row(E(2));

  _R.col(0) << e1.squaredNorm(), 0;
  _R.col(1) << e2.dot(e1), e2.cross(e1).norm();
  _R /= e1.norm();
  _R = _R.inverse().eval();

  coeff = thickness / 2 * e1.cross(e2).norm();
}

Eigen::Matrix<double, 3, 2> NeoHookeanMURIElement::deformationGradient(const Eigen::Ref<const Eigen::VectorXd> X, double stretch) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> Ds;
  Ds.col(0) = X.segment<3>(3 * idx(0)) - X.segment<3>(3 * idx(2));
  Ds.col(1) = X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = stretch * Ds * _R;

  return F;
}

Eigen::Matrix2d NeoHookeanMURIElement::stress(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> F = deformationGradient(X, stretch);
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

double NeoHookeanMURIElement::energy(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch, double mass) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> F = deformationGradient(X, stretch);
  Matrix2d C = F.transpose() * F;
  double lnJ = log(C.determinant()) / 2;

  return coeff * (mu / 2 * (C.trace() - 2 - 2 * lnJ) + lambda / 2 * pow(lnJ, 2) +
                  9.8 * mass * (X(3 * idx(0) + 2) + X(3 * idx(1) + 2) + X(3 * idx(2) + 2)) / 3);
}

fsim::Vec<double, 9>
NeoHookeanMURIElement::gradient(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch, double mass) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> F = deformationGradient(X, stretch);
  Matrix2d Cinv = (F.transpose() * F).inverse();
  Matrix2d S = stress(Cinv, lambda, mu);

  Matrix<double, 3, 2> H = stretch * coeff * F * (S * _R.transpose());

  Vec<double, 9> grad;
  grad.segment<3>(0) = H.col(0);
  grad.segment<3>(3) = H.col(1);
  grad.segment<3>(6) = -H.col(0) - H.col(1);

  grad(2) += 9.8 * coeff / 3 * mass;
  grad(5) += 9.8 * coeff / 3 * mass;
  grad(8) += 9.8 * coeff / 3 * mass;

  return grad;
}

fsim::Vec<double, 9>
NeoHookeanMURIElement::gradient_derivative_sensitivity(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> F = deformationGradient(X, stretch);
  Matrix2d Cinv = (F.transpose() * F).inverse();
  Matrix2d S = stress(Cinv, lambda, mu);

  Matrix<double, 3, 2> H = -coeff * (2 * mu * F + 3 * lambda / stretch * F * Cinv) * _R.transpose();

  fsim::Vec<double, 9> res;
  res.segment<3>(0) = H.col(0);
  res.segment<3>(3) = H.col(1);
  res.segment<3>(6) = -H.col(0) - H.col(1);

  return res;
}

fsim::Mat<double, 9, 9>
NeoHookeanMURIElement::hessian(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> F = deformationGradient(X, stretch);

  Matrix2d Cinv = (F.transpose() * F).inverse();
  Matrix2d S = stress(Cinv, lambda, mu);
  Matrix3d _C = elasticityTensor(Cinv, lambda, mu);

  Matrix3d A = _C(0, 0) * F.col(0) * F.col(0).transpose() + S(0, 0) * Matrix3d::Identity() +
               _C(2, 2) * F.col(1) * F.col(1).transpose() + 2 * _C(0, 2) * sym(F.col(0) * F.col(1).transpose());
  Matrix3d B = _C(1, 1) * F.col(1) * F.col(1).transpose() + S(1, 1) * Matrix3d::Identity() +
               _C(2, 2) * F.col(0) * F.col(0).transpose() + 2 * _C(1, 2) * sym(F.col(0) * F.col(1).transpose());
  Matrix3d C = _C(0, 1) * F.col(0) * F.col(1).transpose() + S(0, 1) * Matrix3d::Identity() +
               _C(2, 2) * F.col(1) * F.col(0).transpose() + _C(0, 2) * F.col(0) * F.col(0).transpose() +
               _C(1, 2) * F.col(1) * F.col(1).transpose();

  Matrix<double, 9, 9> hess;

  for(int i = 0; i < 2; ++i)
    for(int j = i; j < 2; ++j)
      hess.block<3, 3>(3 * i, 3 * j) =
          _R(i, 0) * _R(j, 0) * A + _R(i, 1) * _R(j, 1) * B + _R(i, 0) * _R(j, 1) * C + _R(i, 1) * _R(j, 0) * C.transpose();

  hess.block<3, 3>(3, 0) = hess.block<3, 3>(0, 3).transpose();
  hess.block<6, 3>(0, 6) = -hess.block<6, 3>(0, 0) - hess.block<6, 3>(0, 3);
  hess.block<3, 6>(6, 0) = hess.block<6, 3>(0, 6).transpose();
  hess.block<3, 3>(6, 6) = -hess.block<3, 3>(0, 6) - hess.block<3, 3>(3, 6);
  return coeff * stretch * stretch * hess;
}

