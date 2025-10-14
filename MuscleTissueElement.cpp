// MuscleTissueElement.cpp
//
// Author: David Jourdan (djourdan@berkelely.edu)
// Created: 3/24/2025

#include <cmath>
#include <numbers>
#include "fsim/util/geometry.h"
#include "MuscleTissueElement.h"

MuscleTissueElement::MuscleTissueElement(const Eigen::Ref<const fsim::Mat2<double>> V, const Eigen::Vector3i &E, const Eigen::VectorXd &_phi)
{
  using namespace Eigen;
  using namespace std::numbers;

  idx = E;

  Vector2d e1 = V.row(E(0)) - V.row(E(2));
  Vector2d e2 = V.row(E(1)) - V.row(E(2));

  R.col(0) << e1;
  R.col(1) << e2;
  R = R.inverse().eval();

  area = 0.5 * (e1(0) * e2(1) - e2(0) * e1(1));

  Phi = Matrix2d::Zero();
  const int n = _phi.size();
  for (int i = 0; i < n; ++i)
  {
    Vector2d u(cos(i * pi / n), sin(i * pi / n));
    Phi += _phi(i) * u * u.transpose();
  }
  Phi /= n;
}

MuscleTissueElement::MuscleTissueElement(const Eigen::Ref<const fsim::Mat2<double>> V, const Eigen::Vector3i &E, double mean_theta, double concentration_eta, double phi)
{
  using namespace Eigen;
  using namespace std::numbers;

  idx = E;

  Vector2d e1 = V.row(E(0)) - V.row(E(2));
  Vector2d e2 = V.row(E(1)) - V.row(E(2));

  R.col(0) << e1;
  R.col(1) << e2;
  R = R.inverse().eval();

  area = 0.5 * (e1(0) * e2(1) - e2(0) * e1(1));

  Vector2d u0 = Vector2d(std::cos(mean_theta), std::sin(mean_theta));

  Phi = phi * (concentration_eta * Matrix2d::Identity() + (1 - 2 * concentration_eta) * u0 * u0.transpose());
}

Eigen::Matrix2d MuscleTissueElement::deformationGradient(const Eigen::Ref<const Eigen::VectorXd> X, double stretch) const
{
  using namespace Eigen;

  Matrix2d Ds;
  Ds.col(0) = X.segment<2>(2 * idx(0)) - X.segment<2>(2 * idx(2));
  Ds.col(1) = X.segment<2>(2 * idx(1)) - X.segment<2>(2 * idx(2));
  Matrix2d F = stretch * Ds * R;

  return F;
}

Eigen::Matrix2d MuscleTissueElement::stress(const Eigen::Matrix2d &F, double lambda, double mu, double stretch, double sigma) const
{
  using namespace Eigen;

  Matrix2d C = F.transpose() * F;
  double lnJ = log(C.determinant()) / 2;

  return mu * Matrix2d::Identity() + (lambda * lnJ - mu) * C.inverse() + sigma * Phi;
  // return mu * Matrix2d::Identity() + (lambda * lnJ - mu) * C.inverse() + sigma * 2 * ((C * Phi).trace() - 1) * Phi;
  // return mu * Matrix2d::Identity() + (lambda * lnJ - mu) * C.inverse() + sigma * (1 - 1 / std::sqrt((C * Phi).trace())) * Phi;
}


Eigen::Matrix3d MuscleTissueElement::elasticityTensor(const Eigen::Matrix2d &F, double lambda, double mu, double stretch, double sigma) const
{
  using namespace Eigen;

  Matrix2d C = F.transpose() * F;
  Matrix2d Cinv = C.inverse();

  double lnJ = log(C.determinant()) / 2;

  Matrix3d _C;
  _C << Cinv(0, 0) * Cinv(0, 0), Cinv(0, 1) * Cinv(0, 1), Cinv(0, 0) * Cinv(0, 1), 
        Cinv(1, 0) * Cinv(1, 0), Cinv(1, 1) * Cinv(1, 1), Cinv(0, 1) * Cinv(1, 1), 
        Cinv(0, 1) * Cinv(0, 0), Cinv(0, 1) * Cinv(1, 1), (Cinv(0, 0) * Cinv(1, 1) + Cinv(0, 1) * Cinv(0, 1)) / 2;
  _C *= 2 * (mu - lambda * lnJ);
  _C += lambda * Vector3d(Cinv(0, 0), Cinv(1, 1), Cinv(0, 1)) * RowVector3d(Cinv(0, 0), Cinv(1, 1), Cinv(0, 1));

  // Vector3d Phi_vec(Phi(0,0), Phi(1,1), Phi(0,1));
  // _C += 4 * sigma * Phi_vec * Phi_vec.transpose();
  // _C += sigma * std::pow((C * Phi).trace(), -3/2.) * Phi_vec * Phi_vec.transpose();

  return _C;
}

double MuscleTissueElement::energy(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch, double sigma) const
{
  using namespace Eigen;

  Matrix2d F = deformationGradient(X, stretch);
  Matrix2d C = F.transpose() * F;
  double lnJ = log(C.determinant()) / 2;

  return area * (mu / 2 * (C.trace() - 2 - 2 * lnJ) + lambda / 2 * pow(lnJ, 2) + sigma / 2 * ((C * Phi).trace() - 1));
  // return area * (mu / 2 * (C.trace() - 2 - 2 * lnJ) + lambda / 2 * pow(lnJ, 2) + sigma / 2 * std::pow((C * Phi).trace() - 1, 2));
  // return area * (mu / 2 * (C.trace() - 2 - 2 * lnJ) + lambda / 2 * pow(lnJ, 2) + sigma / 2 * std::pow(std::sqrt((C * Phi).trace()) - 1, 2));
}

fsim::Vec<double, 6>
MuscleTissueElement::gradient(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch, double sigma) const
{
  using namespace Eigen;

  Matrix2d F = deformationGradient(X, stretch);
  Matrix2d S = stress(F, lambda, mu, stretch, sigma);

  Matrix2d H = stretch * area * F * S * R.transpose();

  fsim::Vec<double, 6> grad;
  grad.segment<2>(0) = H.col(0);
  grad.segment<2>(2) = H.col(1);
  grad.segment<2>(4) = -H.col(0) - H.col(1);

  return grad;
}

// fsim::Vec<double, 6>
// MuscleTissueElement::gradient_derivative_sensitivity(const Eigen::Ref<const Eigen::VectorXd> X) const
// {
//   using namespace Eigen;

//   Matrix2d F = deformationGradient(X);
//   Matrix2d S = stress(1);

//   Matrix2d H = area * F * S * R.transpose();

//   fsim::Vec<double, 6> grad;
//   grad.segment<2>(0) = H.col(0);
//   grad.segment<2>(2) = H.col(1);
//   grad.segment<2>(4) = -H.col(0) - H.col(1);

//   return grad;
// }

fsim::Mat<double, 6, 6>
MuscleTissueElement::hessian(const Eigen::Ref<const Eigen::VectorXd> X, double lambda, double mu, double stretch, double sigma) const
{
  using namespace Eigen;

  Matrix2d F = deformationGradient(X, stretch);
  Matrix2d S = stress(F, lambda, mu, stretch, sigma);
  Matrix3d _C = elasticityTensor(F, lambda, mu, stretch, sigma);

  Matrix2d A = _C(0, 0) * F.col(0) * F.col(0).transpose() + S(0, 0) * Matrix2d::Identity() +
               _C(2, 2) * F.col(1) * F.col(1).transpose() + 2 * _C(0, 2) * fsim::sym(F.col(0) * F.col(1).transpose());
  Matrix2d B = _C(1, 1) * F.col(1) * F.col(1).transpose() + S(1, 1) * Matrix2d::Identity() +
               _C(2, 2) * F.col(0) * F.col(0).transpose() + 2 * _C(1, 2) * fsim::sym(F.col(0) * F.col(1).transpose());
  Matrix2d C = _C(0, 1) * F.col(0) * F.col(1).transpose() + S(0, 1) * Matrix2d::Identity() +
               _C(2, 2) * F.col(1) * F.col(0).transpose() + _C(0, 2) * F.col(0) * F.col(0).transpose() +
               _C(1, 2) * F.col(1) * F.col(1).transpose();

  Matrix<double, 6, 6> hess;

  for (int i = 0; i < 3; ++i)
    for (int j = i; j < 3; ++j)
      hess.block<2, 2>(2 * i, 2 * j) =
          R(i, 0) * R(j, 0) * A + R(i, 1) * R(j, 1) * B + R(i, 0) * R(j, 1) * C + R(i, 1) * R(j, 0) * C.transpose();

  hess.block<2, 2>(2, 0) = hess.block<2, 2>(0, 2).transpose();
  hess.block<4, 2>(0, 4) = -hess.block<4, 2>(0, 0) - hess.block<4, 2>(0, 2);
  hess.block<2, 4>(4, 0) = hess.block<4, 2>(0, 4).transpose();
  hess.block<2, 2>(4, 4) = -hess.block<2, 2>(0, 4) - hess.block<2, 2>(2, 4);

  // TinyAD::project_positive_definite<6, double>(hess, 1e-9);

  return area * stretch * stretch * hess;
}
