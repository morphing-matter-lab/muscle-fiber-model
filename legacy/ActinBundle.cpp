// ActinBundle.ipp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 12/06/21

#include "ActinBundle.h"

ActinBundle::ActinBundle(const Eigen::Ref<const fsim::Mat3<double>> V,
                         const Eigen::Ref<const fsim::Mat3<int>> F,
                         const Eigen::Ref<const Eigen::MatrixXd> Phi,
                         double thickness,
                         double sigma_max,
                         double _e0,
                         double _e1)
    : e0(_e0), e1(_e1)
{
  using namespace Eigen;

  nV = V.rows();
  int nF = F.rows();

  this->_elements.reserve(nF);
  for (int i = 0; i < nF; ++i)
    this->_elements.emplace_back(V, F.row(i), Phi.row(i), thickness, sigma_max);
}

double ActinBundle::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  return ModelBase<FiberElement>::energy(X, e0, e1);
}

void ActinBundle::gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
{
  ModelBase<FiberElement>::gradient(X, Y, e0, e1);
}

Eigen::VectorXd ActinBundle::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  VectorXd Y = VectorXd::Zero(X.size());
  gradient(X, Y);
  return Y;
}

Eigen::SparseMatrix<double> ActinBundle::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  return ModelBase<FiberElement>::hessian(X, e0, e1);
}

std::vector<Eigen::Triplet<double>>
ActinBundle::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  return ModelBase<FiberElement>::hessianTriplets(X, e0, e1);
}
