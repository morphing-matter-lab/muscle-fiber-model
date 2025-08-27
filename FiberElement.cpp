// FiberElement.cpp
//
// Author: David Jourdan (djourdan@berkelely.edu)
// Created: 3/24/2025

#include <cmath>
#include <numbers>
#include "fsim/util/geometry.h"
#include "FiberElement.h"

FiberElement::FiberElement(const Eigen::Ref<const fsim::Mat3<double>> V, const Eigen::Vector3i &E, const Eigen::VectorXd &_phi, double thickness, double sigma_max)
    : phi(_phi)
{
  using namespace Eigen;

  idx = E;

  Vector3d e1 = V.row(E(0)) - V.row(E(2));
  Vector3d e2 = V.row(E(1)) - V.row(E(2));

  R.col(0) << e1.squaredNorm(), 0;
  R.col(1) << e2.dot(e1), e2.cross(e1).norm();
  R /= e1.norm();
  R = R.inverse().eval();

  coeff = sigma_max * thickness / 2 * e1.cross(e2).norm();
}

Eigen::Matrix<double, 3, 2> FiberElement::deformationGradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> Ds;
  Ds.col(0) = X.segment<3>(3 * idx(0)) - X.segment<3>(3 * idx(2));
  Ds.col(1) = X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = Ds * R;

  return F;
}

double FiberElement::strain(const Eigen::Matrix<double, 3, 2> &F, const Eigen::Vector2d &u) const
{
  using namespace Eigen;

  Vector3d Fu = F * u;
  return 0.5 * (Fu.dot(Fu) - 1);
}

Eigen::Matrix2d FiberElement::stress(const Eigen::Matrix<double, 3, 2> &F, double e0, double e1) const
{
  using namespace Eigen;
  using namespace std::numbers;

  Matrix2d res = Matrix2d::Zero();
  const int n = phi.size();
  for (int i = 0; i < n; ++i)
  {
    Vector2d u(cos(i * pi / n), sin(i * pi / n));
    double e = strain(F, u);
    res += phi(i) * stress(e, e0, e1) * u * u.transpose();
  }

  return res / n;
}

double FiberElement::stress(double e, double e0, double e1) const
{
  // double res = std::exp(-std::pow(e / e0, 2));
  double res = 0;
  if (e >= 0)
    res += std::pow(e / e1, 2);
  return res;
}

Eigen::Matrix3d FiberElement::elasticityTensor(const Eigen::Matrix<double, 3, 2> &F, double e0, double e1) const
{
  using namespace Eigen;
  using namespace std::numbers;

  Matrix3d res = Matrix3d::Zero();
  const int n = phi.size();
  for (int i = 0; i < n; ++i)
  {
    Vector2d u(cos(i * pi / n), sin(i * pi / n));
    double e = strain(F, u);

    Vector3d uuT;
    uuT << u(0) * u(0), u(1) * u(1), u(1) * u(0);
    res += phi(i) * stressDeriv(e, e0, e1) * uuT * uuT.transpose();
  }

  return res / n;
}

double FiberElement::stressDeriv(double e, double e0, double e1) const
{
  // double stress_deriv = -2 * e / e0 / e0 * std::exp(-std::pow(e / e0, 2));
  double stress_deriv = 0;
  if (e >= 0)
    stress_deriv += 2 * e / std::pow(e1, 2);

  return stress_deriv;
}

double FiberElement::energy(double e, double e0, double e1) const
{
  double res = 0;
  if (e >= 0)
    res += std::pow(e / e1, 2) * e / 3;
  return res;
}

double FiberElement::energy(const Eigen::Ref<const Eigen::VectorXd> X, double e0, double e1) const
{
  using namespace Eigen;
  using namespace std::numbers;

  Matrix<double, 3, 2> F = deformationGradient(X);

  double res = 0;
    // double res = 0.5 / inv_sqrtpi * e0 * std::erf(e / e0);
  const int n = phi.size();
  for (int i = 0; i < n; ++i)
  {
    Vector2d u(cos(i * pi / n), sin(i * pi / n));
    double e = strain(F, u);
    res += phi(i) * energy(e, e0, e1);
  }

  return coeff * res / n;
}

fsim::Vec<double, 9>
FiberElement::gradient(const Eigen::Ref<const Eigen::VectorXd> X, double e0, double e1) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> F = deformationGradient(X);
  Matrix2d S = stress(F, e0, e1);

  Matrix<double, 3, 2> H = coeff * F * (S * R.transpose());

  fsim::Vec<double, 9> grad;
  grad.segment<3>(0) = H.col(0);
  grad.segment<3>(3) = H.col(1);
  grad.segment<3>(6) = -H.col(0) - H.col(1);

  return grad;
}

fsim::Mat<double, 9, 9>
FiberElement::hessian(const Eigen::Ref<const Eigen::VectorXd> X, double e0, double e1) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> F = deformationGradient(X);

  Matrix2d S = stress(F, e0, e1);
  Matrix3d _C = elasticityTensor(F, e0, e1);

  Matrix3d A = _C(0, 0) * F.col(0) * F.col(0).transpose() + S(0, 0) * Matrix3d::Identity() +
               _C(2, 2) * F.col(1) * F.col(1).transpose() + 2 * _C(0, 2) * fsim::sym(F.col(0) * F.col(1).transpose());
  Matrix3d B = _C(1, 1) * F.col(1) * F.col(1).transpose() + S(1, 1) * Matrix3d::Identity() +
               _C(2, 2) * F.col(0) * F.col(0).transpose() + 2 * _C(1, 2) * fsim::sym(F.col(0) * F.col(1).transpose());
  Matrix3d C = _C(0, 1) * F.col(0) * F.col(1).transpose() + S(0, 1) * Matrix3d::Identity() +
               _C(2, 2) * F.col(1) * F.col(0).transpose() + _C(0, 2) * F.col(0) * F.col(0).transpose() +
               _C(1, 2) * F.col(1) * F.col(1).transpose();

  Matrix<double, 9, 9> hess;

  for (int i = 0; i < 2; ++i)
    for (int j = i; j < 2; ++j)
      hess.block<3, 3>(3 * i, 3 * j) =
          R(i, 0) * R(j, 0) * A + R(i, 1) * R(j, 1) * B + R(i, 0) * R(j, 1) * C + R(i, 1) * R(j, 0) * C.transpose();

  hess.block<3, 3>(3, 0) = hess.block<3, 3>(0, 3).transpose();
  hess.block<6, 3>(0, 6) = -hess.block<6, 3>(0, 0) - hess.block<6, 3>(0, 3);
  hess.block<3, 6>(6, 0) = hess.block<6, 3>(0, 6).transpose();
  hess.block<3, 3>(6, 6) = -hess.block<3, 3>(0, 6) - hess.block<3, 3>(3, 6);
  return coeff * hess;
}
