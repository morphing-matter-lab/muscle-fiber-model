// Spring.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 04/10/18

#include <Eigen/Core>

struct AnchoredSpring
{
  AnchoredSpring(int _i, Eigen::Vector3d anchor_pos) : idx(_i), anchor(anchor_pos) {}

  double energy(const Eigen::Ref<const Eigen::VectorXd> pos) const
  {
    return (pos.segment<3>(3 * idx) - anchor).squaredNorm() / 2.;
  }

  Eigen::Vector3d gradient(const Eigen::Ref<const Eigen::VectorXd> pos) const
  {
    return pos.segment<3>(3 * idx) - anchor;
  }

  Eigen::Matrix3d hessian(const Eigen::Ref<const Eigen::VectorXd> pos) const
  {
    return Eigen::Matrix3d::Identity();
  }

  int idx;
  Eigen::Vector3d anchor;
};
