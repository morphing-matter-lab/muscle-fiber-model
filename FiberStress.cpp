#include "FiberStress.h"

namespace nb = nanobind;

Eigen::VectorXd directional_fiber_stress(const Eigen::MatrixXd &V, const Eigen::MatrixXd &P, const Eigen::MatrixXd &F, double theta)
{
  using namespace Eigen;

  const double e0 = 1.2e-1;
  const double e1 = 1.7e-1;

  VectorXd stress(F.rows());
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads() - 1)
  for (int i = 0; i < F.rows(); ++i)
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

    if (strain > 0)
      stress(i) += std::pow(strain / e1, 2);
  }
  return stress;
}

Eigen::MatrixXd directional_strain(const Eigen::MatrixXd &V, const Eigen::MatrixXd &P, const Eigen::MatrixXd &F, int n)
{
  using namespace Eigen;

  const double e0 = 1.2e-1;
  const double e1 = 1.7e-1;

  MatrixXd strain(F.rows(), n);
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads() - 1)
  for (int i = 0; i < F.rows(); ++i)
  {
    Matrix2d Ds;
    Ds.col(0) = V.row(F(i, 1)) - V.row(F(i, 0));
    Ds.col(1) = V.row(F(i, 2)) - V.row(F(i, 0));
    Matrix2d R;
    R.col(0) = P.row(F(i, 1)) - P.row(F(i, 0));
    R.col(1) = P.row(F(i, 2)) - P.row(F(i, 0));

    Matrix2d def_gradient = Ds * R.inverse();

    for (int j = 0; j < n; ++j)
    {
      double theta = j * 3.14159 / n;
      Vector2d Fu = def_gradient * Vector2d(cos(theta), sin(theta));

      strain(i, j) = 0.5 * (Fu.dot(Fu) - 1);
    }
  }
  return strain;
}

Eigen::MatrixXd fiber_stress(const Eigen::MatrixXd &V, const Eigen::MatrixXd &P, const Eigen::MatrixXd &F, int n, double e0, double e1)
{
  using namespace Eigen;

  MatrixXd stress(F.rows(), n);
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads() - 1)
  for (int i = 0; i < F.rows(); ++i)
  {
    Matrix2d Ds;
    Ds.col(0) = V.row(F(i, 1)) - V.row(F(i, 0));
    Ds.col(1) = V.row(F(i, 2)) - V.row(F(i, 0));
    Matrix2d R;
    R.col(0) = P.row(F(i, 1)) - P.row(F(i, 0));
    R.col(1) = P.row(F(i, 2)) - P.row(F(i, 0));

    Matrix2d def_gradient = Ds * R.inverse();

    for (int j = 0; j < n; ++j)
    {
      double theta = j * 3.14159 / n;
      Vector2d Fu = def_gradient * Vector2d(cos(theta), sin(theta));
      double strain = 0.5 * (Fu.dot(Fu) - 1);

      stress(i, j) = std::exp(-std::pow(strain / e0, 2));

      if (strain > 0)
        stress(i, j) += std::pow(strain / e1, 2);
    }
  }
  return stress;
}

void polymer_fraction_one_step(nb::DRef<Eigen::MatrixXd> polymer_fraction, const Eigen::MatrixXd &stress, double k0, double k1, double kd, double frac_f, double frac_s, double dt)
{
  using namespace Eigen;

  int n = polymer_fraction.cols();
  assert(stress.cols() == n);

#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads() - 1)
  for (int i = 0; i < polymer_fraction.rows(); ++i)
  {
    VectorXd kf = k0 * RowVectorXd::Ones(n) + k1 * stress.row(i);
    VectorXd b = polymer_fraction.row(i) + dt / frac_f * (1 - frac_s - frac_f) * kf.transpose();
    MatrixXd M = (1 + dt * kd) * MatrixXd::Identity(n, n) + dt / frac_f / n * kf * RowVectorXd::Ones(n);
    polymer_fraction.row(i) = M.colPivHouseholderQr().solve(b);
  }
}

Eigen::MatrixXd polymer_fraction_steady_state(const Eigen::MatrixXd &stress, double k0, double k1, double kd, double frac_f, double frac_s)
{
  using namespace Eigen;

  int n = stress.cols();
  assert(stress.cols() == n);

  Eigen::MatrixXd polymer_fraction(stress.rows(), n);

#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads() - 1)
  for (int i = 0; i < polymer_fraction.rows(); ++i)
  {
    VectorXd kf = k0 * RowVectorXd::Ones(n) + k1 * stress.row(i);
    VectorXd b = 1 / frac_f * (1 - frac_s - frac_f) * kf.transpose();
    MatrixXd M = kd * MatrixXd::Identity(n, n) + 1 / frac_f / n * kf * RowVectorXd::Ones(n);
    polymer_fraction.row(i) = M.colPivHouseholderQr().solve(b);
  }

  return polymer_fraction;
}


Eigen::MatrixXd polymer_fraction_reduced(const Eigen::MatrixXd &stress, double k1, double kd, double frac_f, double frac_s)
{
  return polymer_fraction_steady_state(stress, 1, k1, kd, frac_f, frac_s);
}