#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/tuple.h>

#include <optim/NewtonSolver.h>
#include <fsim/CompositeModel.h>
#include <fsim/ElasticMembrane.h>

#include <iostream>

#include "PillarModel.h"
#include "FiberStress.h"
#include "ActinBundle.h"

namespace nb = nanobind;
using namespace nb::literals;

std::tuple<std::vector<double>, Eigen::MatrixXd> compute_stretch_angles(const Eigen::MatrixXd &V,
                                                                        const Eigen::MatrixXd &P,
                                                                        const Eigen::MatrixXi &F)
{
  std::vector<double> angles(F.rows());
  Eigen::MatrixXd eigenvalues(F.rows(), 2);

#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads() - 1)
  for (int i = 0; i < F.rows(); ++i)
  {
    // Get 3D vertex positions
    Eigen::Vector3d ar = V.row(F(i, 0));
    Eigen::Vector3d br = V.row(F(i, 1));
    Eigen::Vector3d cr = V.row(F(i, 2));
    Eigen::Matrix<double, 3, 2> Mr;
    Mr.col(0) = br - ar;
    Mr.col(1) = cr - ar;

    // Get 2D vertex positions
    Eigen::Vector2d a = P.row(F(i, 0));
    Eigen::Vector2d b = P.row(F(i, 1));
    Eigen::Vector2d c = P.row(F(i, 2));
    Eigen::Matrix2d M;
    M.col(0) = b - a;
    M.col(1) = c - a;

    // Compute deformation gradient
    Eigen::Matrix<double, 3, 2> F = Mr * M.inverse();

    // Compute first fundamental form
    Eigen::Matrix2d I = F.transpose() * F;

    // when considering the metric tensor I (instead of the differential J), the angles should be divided by two
    angles[i] = atan2(I(0, 1) + I(1, 0), I(0, 0) - I(1, 1));
    if (angles[i] < 0)
      angles[i] += 3.14159;
    else
      angles[i] -= 3.14159;
    angles[i] /= 2;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(I);
    if (eigensolver.info() != Eigen::Success)
      std::cout << "Failed\n";
    eigenvalues.row(i) = eigensolver.eigenvalues();
  }

  return std::make_tuple(angles, eigenvalues);
}

Eigen::VectorXd
compute_membrane_energies(const Eigen::MatrixXd &V,
                          const Eigen::MatrixXi &F,
                          const Eigen::MatrixXd &NV,
                          double stretch_factor,
                          double poisson_ratio)
{
  using namespace Eigen;

  // declare NeohookeanMembrane object
  double thickness = 1;
  double young_modulus = 1;
  double mass = 0;

  fsim::NeoHookeanMembrane model(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass);
  double lambda = young_modulus * poisson_ratio / (1 - std::pow(poisson_ratio, 2));
  double mu = 0.5 * young_modulus / (1 + poisson_ratio);

  auto elements = model.getElements();
  VectorXd vec(elements.size());
  for (int i = 0; i < elements.size(); ++i)
  {
    vec(i) = elements[i].energy(NV.reshaped<RowMajor>(), lambda, mu, mass);
  }

  return vec;
}

Eigen::MatrixXd
compute_membrane_forces(const Eigen::MatrixXd &V,
                        const Eigen::MatrixXi &F,
                        const Eigen::MatrixXd &NV,
                        double stretch_factor,
                        double poisson_ratio)
{
  using namespace Eigen;

  // declare NeohookeanMembrane object
  double thickness = 1;
  double young_modulus = 1;
  double mass = 0;

  fsim::NeoHookeanMembrane model(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass);

  VectorXd force = -model.gradient(NV.reshaped<RowMajor>());
  return Map<fsim::Mat3<double>>(force.data(), V.rows(), 3);
}

void simulate_membrane(nb::DRef<Eigen::MatrixXd> V,
                       const nb::DRef<Eigen::MatrixXd> &P,
                       const nb::DRef<Eigen::MatrixXi> &F,
                       const nb::DRef<Eigen::MatrixXd> &Phi,
                       const std::vector<int> &fixed_idx,
                       double stretch_factor,
                       double poisson_ratio,
                       double sigma_max,
                       double e0,
                       double e1)
{
  using namespace Eigen;

  // declare NeohookeanMembrane object
  double thickness = 1;
  double young_modulus = 1;
  double pillar_modulus = 100;
  double mass = 0;

  fsim::CompositeModel model(
      fsim::NeoHookeanMembrane(P / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass),
      ActinBundle(P / stretch_factor, F, Phi, thickness, sigma_max, e0, e1));

  // declare NewtonSolver object
  optim::NewtonSolver<double> solver;
  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be
  solver.options.fixed_dofs = fixed_idx;
  solver.options.display = optim::SolverDisplay::quiet;

  solver.solve(model, V.reshaped<RowMajor>());

  V = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3);
}

NB_MODULE(fabsim_py, m)
{
  m.def("simulate_membrane", &simulate_membrane);
  m.def("compute_membrane_energies", &compute_membrane_energies);
  m.def("compute_membrane_forces", &compute_membrane_forces);
  m.def("compute_stretch_angles", &compute_stretch_angles);
  m.def("directional_fiber_stress", &directional_fiber_stress);
  m.def("directional_strain", &directional_strain);
  m.def("fiber_stress", &fiber_stress, "V"_a, "P"_a, "F"_a, "n"_a, "e0"_a = 1.2e-1, "e1"_a = 1.7e-1);
  m.def("polymer_fraction_one_step", &polymer_fraction_one_step);
  m.def("polymer_fraction_steady_state", &polymer_fraction_steady_state);
  m.def("polymer_fraction_reduced", &polymer_fraction_reduced);
}