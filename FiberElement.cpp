// FiberElement.cpp
//
// Author: David Jourdan (djourdan@berkelely.edu)
// Created: 3/24/2025

#include <cmath>
#include <numbers>
#include "fsim/util/geometry.h"
#include "FiberElement.h"

FiberElement::FiberElement(const Eigen::Ref<const fsim::Mat2<double>> V, const Eigen::Vector3i &E, const Eigen::VectorXd &_phi)
    : phi(_phi)
{
  using namespace Eigen;
  using namespace std::numbers;

  idx = E;

  Vector2d e1 = V.row(E(0)) - V.row(E(2));
  Vector2d e2 = V.row(E(1)) - V.row(E(2));

  R.col(0) << e1;
  R.col(1) << e2;
  R = R.inverse().eval();

  coeff = 0.5 * (e1(0) * e2(1) - e2(0) * e1(1));
}

Eigen::Matrix2d FiberElement::deformationGradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  Matrix2d Ds;
  Ds.col(0) = X.segment<2>(2 * idx(0)) - X.segment<2>(2 * idx(2));
  Ds.col(1) = X.segment<2>(2 * idx(1)) - X.segment<2>(2 * idx(2));
  Matrix2d F = Ds * R;

  return F;
}

Eigen::Matrix2d FiberElement::stress(double sigma) const
{
  using namespace Eigen;
  using namespace std::numbers;

  Matrix2d res = Matrix2d::Zero();
  const int n = phi.size();
  for (int i = 0; i < n; ++i)
  {
    Vector2d u(cos(i * pi / n), sin(i * pi / n));
    res += phi(i) * u * u.transpose();
  }

  return sigma * res / n;
}

double FiberElement::energy(const Eigen::Ref<const Eigen::VectorXd> X, double sigma) const
{
  using namespace Eigen;
  using namespace std::numbers;

  Matrix2d F = deformationGradient(X);

  double res = 0;
  const int n = phi.size();
  for (int i = 0; i < n; ++i)
  {
    Vector2d u(cos(i * pi / n), sin(i * pi / n));
    double e = 0.5 * ((F * u).dot(F * u) - 1);
    res += phi(i) * e;
  }

  return coeff * sigma * res / n;
}

fsim::Vec<double, 6>
FiberElement::gradient(const Eigen::Ref<const Eigen::VectorXd> X, double sigma) const
{
  using namespace Eigen;

  Matrix2d F = deformationGradient(X);
  Matrix2d S = stress(sigma);

  Matrix2d H = coeff * F * S * R.transpose();

  fsim::Vec<double, 6> grad;
  grad.segment<2>(0) = H.col(0);
  grad.segment<2>(2) = H.col(1);
  grad.segment<2>(4) = -H.col(0) - H.col(1);

  return grad;
}

fsim::Vec<double, 6>
FiberElement::gradient_derivative_sensitivity(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  Matrix2d F = deformationGradient(X);
  Matrix2d S = stress(1);

  Matrix2d H = coeff * F * S * R.transpose();

  fsim::Vec<double, 6> grad;
  grad.segment<2>(0) = H.col(0);
  grad.segment<2>(2) = H.col(1);
  grad.segment<2>(4) = -H.col(0) - H.col(1);

  return grad;
}

fsim::Mat<double, 6, 6>
FiberElement::hessian(const Eigen::Ref<const Eigen::VectorXd> X, double sigma) const
{
  using namespace Eigen;

  Matrix2d S = stress(sigma);

  Matrix<double, 6, 6> hess;

  for (int i = 0; i < 3; ++i)
    for (int j = i; j < 3; ++j)
      hess.block<2, 2>(2 * i, 2 * j) =
          (R(i, 0) * R(j, 0) * S(0, 0) + R(i, 1) * R(j, 1) * S(1, 1) + R(i, 0) * R(j, 1) * S(0, 1) + R(i, 1) * R(j, 0) * S(1, 0)) * Matrix2d::Identity();

  hess.block<2, 2>(2, 0) = hess.block<2, 2>(0, 2).transpose();
  hess.block<4, 2>(0, 4) = -hess.block<4, 2>(0, 0) - hess.block<4, 2>(0, 2);
  hess.block<2, 4>(4, 0) = hess.block<4, 2>(0, 4).transpose();
  hess.block<2, 2>(4, 4) = -hess.block<2, 2>(0, 4) - hess.block<2, 2>(2, 4);
  return coeff * hess;
}
