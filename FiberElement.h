// FiberElement.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 10/30/21

#pragma once

#include <cmath>
#include <numbers>
#include "fsim/util/geometry.h"
#include "fsim/ElementBase.h"
#include "fsim/util/typedefs.h"

template <int n>
class FiberElement : public fsim::ElementBase<3>
{
public:
  /**
   * Constructor for the FiberElement class
   * @param V  n by 2 list of vertex positions (each row is a vertex)
   * @param face  list of 3 indices, one per vertex of the triangle
   */
  FiberElement(const Eigen::Ref<const fsim::Mat3<double>> V, const Eigen::Vector3i &face, const fsim::Vec<double, n> &phi, double thickness, double sigma_max);

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  energy of the triangle element for a given material model
   */
  double energy(const Eigen::Ref<const Eigen::VectorXd> X, double e0, double e1) const;
  double energy(double e, double e0, double e1) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  gradient of the energy (9 by 1 vector), derivatives are stacked in the order of the triangle indices
   */
  LocalVector gradient(const Eigen::Ref<const Eigen::VectorXd> X, double e0, double e1) const;

  /**
   * @param X  a flat vector stacking all degrees of freedom
   * @return  hessian matrix of the energy w.r.t. all 9 degrees of freedom of the triangle
   */
  LocalMatrix hessian(const Eigen::Ref<const Eigen::VectorXd> X, double e0, double e1) const;

  /**
   * Computes the Cauchy-Green deformation tensor C = F^T F where F is the deformation gradient
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return Cauchy-Green deformation tensor
   */
  Eigen::Matrix<double, 3, 2> deformationGradient(const Eigen::Ref<const Eigen::VectorXd> X) const;

  /**
   * Computes the Second Piola-Kirchhoff stress tensor S = \frac{\partial f}{\partial E} where E is the Green strain
   * Uses Voigt's notation to express it as a vector
   * @param V  n by 3 list of vertex positions (each row is a vertex)
   * @return second Piola-Kirchhoff stress
   */
  Eigen::Matrix2d stress(const Eigen::Matrix<double, 3, 2> &F, double e0, double e1) const;
  double stress(double e, double e0, double e1) const;

  double strain(const Eigen::Matrix<double, 3, 2> &F, const Eigen::Vector2d &u) const;

  Eigen::Matrix3d elasticityTensor(const Eigen::Matrix<double, 3, 2> &F, double e0, double e1) const;
  double stressDeriv(double e, double e0, double e1) const;

  double coeff;
  Eigen::Matrix2d R;
  fsim::Vec<double, n> phi;
};

template <int n>
FiberElement<n>::FiberElement(const Eigen::Ref<const fsim::Mat3<double>> V, const Eigen::Vector3i &E, const fsim::Vec<double, n> &_phi, double thickness, double sigma_max)
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

template <int n>
Eigen::Matrix<double, 3, 2> FiberElement<n>::deformationGradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  Matrix<double, 3, 2> Ds;
  Ds.col(0) = X.segment<3>(3 * idx(0)) - X.segment<3>(3 * idx(2));
  Ds.col(1) = X.segment<3>(3 * idx(1)) - X.segment<3>(3 * idx(2));
  Matrix<double, 3, 2> F = Ds * R;

  return F;
}

template <int n>
double FiberElement<n>::strain(const Eigen::Matrix<double, 3, 2> &F, const Eigen::Vector2d &u) const
{
  using namespace Eigen;

  Vector3d Fu = F * u;
  return 0.5 * (Fu.dot(Fu) - 1);
}

template <int n>
Eigen::Matrix2d FiberElement<n>::stress(const Eigen::Matrix<double, 3, 2> &F, double e0, double e1) const
{
  using namespace Eigen;
  using namespace std::numbers;

  Matrix2d res = Matrix2d::Zero();
  for (int i = 0; i < n; ++i)
  {
    Vector2d u(cos(i * pi / n), sin(i * pi / n));
    double e = strain(F, u);
    res += phi(i) * stress(e, e0, e1) * u * u.transpose();
  }

  return res / n;
}

template <int n>
double FiberElement<n>::stress(double e, double e0, double e1) const
{
  double res = std::exp(-std::pow(e / e0, 2));
  if (e >= 0)
    res += std::pow(e / e1, 2);
  return res;
}

template <int n>
Eigen::Matrix3d FiberElement<n>::elasticityTensor(const Eigen::Matrix<double, 3, 2> &F, double e0, double e1) const
{
  using namespace Eigen;
  using namespace std::numbers;

  Matrix3d res = Matrix3d::Zero();
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

template <int n>
double FiberElement<n>::stressDeriv(double e, double e0, double e1) const
{
  double stress_deriv = -2 * e / e0 / e0 * std::exp(-std::pow(e / e0, 2));
  if (e >= 0)
    stress_deriv += 2 * e / std::pow(e1, 2);

  return stress_deriv;
}

template <int n>
double FiberElement<n>::energy(double e, double e0, double e1) const
{
  using namespace std::numbers;

  double res = 0.5 / inv_sqrtpi * e0 * std::erf(e / e0);
  if (e >= 0)
    res += std::pow(e / e1, 2) * e / 3;
  return res;
}

template <int n>
double FiberElement<n>::energy(const Eigen::Ref<const Eigen::VectorXd> X, double e0, double e1) const
{
  using namespace Eigen;
  using namespace std::numbers;

  Matrix<double, 3, 2> F = deformationGradient(X);

  double res = 0;
  for (int i = 0; i < n; ++i)
  {
    Vector2d u(cos(i * pi / n), sin(i * pi / n));
    double e = strain(F, u);
    res += phi(i) * energy(e, e0, e1);
  }

  return coeff * res / n;
}

template <int n>
fsim::Vec<double, 9>
FiberElement<n>::gradient(const Eigen::Ref<const Eigen::VectorXd> X, double e0, double e1) const
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

template <int n>
fsim::Mat<double, 9, 9>
FiberElement<n>::hessian(const Eigen::Ref<const Eigen::VectorXd> X, double e0, double e1) const
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
