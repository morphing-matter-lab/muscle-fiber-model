// ActinBundle.ipp
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 12/06/21

#include "ActinBundle.h"

ActinBundle::ActinBundle(const Eigen::Ref<const fsim::Mat2<double>> V,
                         const Eigen::Ref<const fsim::Mat3<int>> F,
                         const Eigen::Ref<const Eigen::MatrixXd> Phi,
                         double sigma_max)
    : sigma(sigma_max)
{
  using namespace Eigen;

  nV = V.rows();
  int nF = F.rows();

  this->_elements.reserve(nF);
  for (int i = 0; i < nF; ++i)
    this->_elements.emplace_back(V, F.row(i), Phi.row(i));
}

double ActinBundle::energy(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  double result = 0;
  for(auto &element: _elements)
  {
    result += element.energy(X, sigma);
  }
  return result;
}

Eigen::VectorXd ActinBundle::gradient(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  VectorXd Y = VectorXd::Zero(X.size());
  gradient(X, Y);
  return Y;
}

void ActinBundle::gradient(const Eigen::Ref<const Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> Y) const
{
  for(auto &element: _elements)
  {
    fsim::Vec<double, 6> grad = element.gradient(X, sigma);

    for(int j = 0; j < 3; ++j)
      Y.segment<2>(2 * element.idx(j)) += grad.segment<2>(2 * j);
  }
}

Eigen::VectorXd ActinBundle::gradient_derivative_sensitivity(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;
  VectorXd Y = VectorXd::Zero(X.size());
  for(auto &element: _elements)
  {
    fsim::Vec<double, 6> grad = element.gradient_derivative_sensitivity(X);

    for(int j = 0; j < 3; ++j)
      Y.segment<2>(2 * element.idx(j)) += grad.segment<2>(2 * j);
  }
  return Y;
}

Eigen::SparseMatrix<double> ActinBundle::hessian(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  std::vector<Triplet<double>> triplets = hessianTriplets(X);

  SparseMatrix<double> hess(X.size(), X.size());
  hess.setFromTriplets(triplets.begin(), triplets.end());
  return hess;
}

std::vector<Eigen::Triplet<double>>
ActinBundle::hessianTriplets(const Eigen::Ref<const Eigen::VectorXd> X) const
{
  using namespace Eigen;

  constexpr int n = 6 * (6 + 3) / 2;
  std::vector<Triplet<double>> triplets(n * _elements.size());

#pragma omp parallel for if(_elements.size() > 1000)
  for(int i = 0; i < _elements.size(); ++i)
  {
    auto &e = _elements[i];
    auto hess = e.hessian(X, sigma);

    int id = 0;
    for(int j = 0; j < 3; ++j)
      for(int k = 0; k < 3; ++k)
        if(e.idx(j) <= e.idx(k))
          for(int l = 0; l < 2; ++l)
            for(int m = 0; m < 2; ++m)
              triplets[n * i + id++] = Triplet<double>(2 * e.idx(j) + l, 2 * e.idx(k) + m, hess(2 * j + l, 2 * k + m));
  }

  return triplets;
}
