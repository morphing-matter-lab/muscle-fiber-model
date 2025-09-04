#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "AnchoredSpring.h"

class PillarModel
{
public:
  PillarModel(const Eigen::MatrixXd &P,
              const std::vector<int> &indices,
              double modulus);

  double energy(const Eigen::Ref<const Eigen::VectorXd> X) const;
  void gradient(const Eigen::Ref<const Eigen::VectorXd> X,
                Eigen::Ref<Eigen::VectorXd> Y) const;
  Eigen::VectorXd gradient(const Eigen::Ref<const Eigen::VectorXd> X) const;
  Eigen::SparseMatrix<double> hessian(const Eigen::Ref<const Eigen::VectorXd> X) const;
  std::vector<Eigen::Triplet<double>> hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const;

  double _modulus;
  std::vector<AnchoredSpring> _springs;
  int nV;
};