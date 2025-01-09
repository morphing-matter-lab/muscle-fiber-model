// PillarModel.cpp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 02/12/24

#include "PillarModel.h"

#include <algorithm>

PillarModel::PillarModel(const Eigen::MatrixXd &P,
                         const std::vector<int> &indices,
                         double modulus)
    : _modulus(modulus)
{
  for (int idx : indices)
    _springs.emplace_back(idx, P.row(idx));
}

double PillarModel::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  double w = 0.0;
  for (const auto &s : _springs)
    w += _modulus * s.energy(X);

  return w;
}

void PillarModel::gradient(const Eigen::Ref<const Eigen::VectorXd> X,
                           Eigen::Ref<Eigen::VectorXd> Y) const
{
  for (const auto &s : _springs)
    Y.segment<3>(3 * s.idx) += _modulus * s.gradient(X);
}

Eigen::VectorXd PillarModel::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  VectorXd Y = VectorXd::Zero(X.size());
  gradient(X, Y);
  return Y;
}

std::vector<Eigen::Triplet<double>>
PillarModel::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  std::vector<Triplet<double>> triplets(_springs.size() * 9);

  for (int k = 0; k < _springs.size(); ++k)
  {
    auto s = _springs[k];
    Matrix3d h = _modulus * s.hessian(X);

    int id = 0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        triplets[9 * k + id++] = Triplet<double>(3 * s.idx + a, 3 * s.idx + b, h(a, b));
  }
  return triplets;
}

Eigen::SparseMatrix<double> PillarModel::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  Eigen::SparseMatrix<double> hess(X.size(), X.size());
  std::vector<Eigen::Triplet<double>> triplets = hessianTriplets(X);
  hess.setFromTriplets(triplets.begin(), triplets.end());
  hess.makeCompressed();
  return hess;
}
