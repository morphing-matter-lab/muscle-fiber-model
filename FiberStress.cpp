#include "FiberStress.h"

Eigen::VectorXd directional_fiber_stress(const Eigen::MatrixXd& V, const Eigen::MatrixXd& P, const Eigen::MatrixXd& F, double theta)
{
  using namespace Eigen;

  const double e0 = 1.2e-1;
  const double e1 = 1.7e-1;

  VectorXd stress(F.rows());
  for(int i = 0; i < F.rows(); ++i)
  {
    Matrix2d Ds;
    Ds.col(0) = V.row(F(i, 1)) - V.row(F(i, 0));
    Ds.col(1) = V.row(F(i, 2)) - V.row(F(i, 0));
    Matrix2d R;
    R.col(0) = P.row(F(i, 1)) - P.row(F(i, 0));
    R.col(1) = P.row(F(i, 2)) - P.row(F(i, 0));
    
    Vector2d u(cos(theta), sin(theta));
    Vector2d Fu = Ds * R.inverse() * u;
    double strain = 0.5 * (Fu.dot(Fu) - 1);

    stress(i) = std::exp(-std::pow(strain / e0, 2));
    
    if(strain > 0)
        stress(i) += std::pow(strain / e1, 2);
  }
  return stress;
}