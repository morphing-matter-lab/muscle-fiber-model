#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/tuple.h>

#include <optim/NewtonSolver.h>
#include <fsim/ElasticMembrane.h>
#include <fsim/ElasticShell.h>
#include "geometrycentral/surface/boundary_first_flattening.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <iostream>

namespace nb = nanobind;
using namespace nb::literals;

std::vector<double> compute_stretch_angles(const Eigen::MatrixXd &V,
                                           const Eigen::MatrixXd &P,
                                           const Eigen::MatrixXi &F)
{
  std::vector<double> angles(F.rows());

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

    // when considering the metric tensor I (instead of the differential J), the angles should be multiplied by two
    angles[i] = atan2(I(0, 1) + I(1, 0), I(0, 0) - I(1, 1));
    if (angles[i] < 0)
      angles[i] += 3.14159;
    else
      angles[i] -= 3.14159;
    angles[i] /= 2;
  }

  return angles;
}

std::vector<double> compute_area_distortion(const Eigen::MatrixXd &V,
                                            const Eigen::MatrixXd &P,
                                            const Eigen::MatrixXi &F)
{
  std::vector<double> areas(F.rows());

#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads() - 1)
  for (int i = 0; i < F.rows(); ++i)
  {
    // Get 3D vertex positions
    Eigen::Vector3d ar = V.row(F(i, 0));
    Eigen::Vector3d br = V.row(F(i, 1));
    Eigen::Vector3d cr = V.row(F(i, 2));

    // Get 2D vertex positions
    Eigen::Vector3d a = P.row(F(i, 0));
    Eigen::Vector3d b = P.row(F(i, 1));
    Eigen::Vector3d c = P.row(F(i, 2));

    areas[i] = (b - a).cross(c - a).norm() / (br - ar).cross(cr - ar).norm();
  }

  return areas;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
boundary_first_flattening(const Eigen::MatrixXd &V,
                          const Eigen::MatrixXi &F)
{
  using namespace geometrycentral;
  using namespace geometrycentral::surface;

  // create geometry-central objects
  ManifoldSurfaceMesh mesh(F);
  VertexPositionGeometry geometry(mesh, V);

  VertexData<Vector2> parameterization = parameterizeBFF(mesh, geometry);

  Eigen::MatrixXd NV(V.rows(), V.cols());

  for (int i = 0; i < NV.rows(); ++i)
  {
    Vector2 v_coord = parameterization[i];
    NV.row(i) << v_coord.x, v_coord.y, 0;
  }

  return std::make_tuple(NV, F);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
simulate_membrane(const Eigen::MatrixXd &V,
                  const Eigen::MatrixXi &F,
                  const std::vector<int> &fixed_idx,
                  double stretch_factor,
                  double poisson_ratio)
{
  using namespace Eigen;

  // declare NeohookeanMembrane object
  double thickness = 1;
  double young_modulus = 1;
  double mass = 0;

  fsim::NeoHookeanMembrane model(V / stretch_factor, F, thickness, young_modulus, poisson_ratio, mass);

  // declare NewtonSolver object
  optim::NewtonSolver<double> solver;
  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
  solver.options.fixed_dofs = fixed_idx;
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be

  solver.solve(model, V.reshaped<RowMajor>());

  MatrixXd NV = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3);
  return std::make_tuple(NV, F);
}

Eigen::MatrixXd
simulate_shell(const Eigen::MatrixXd &V,
               const Eigen::MatrixXd &P,
               const Eigen::MatrixXi &F,
               const std::vector<int> &fixed_idx,
               double thickness,
               double poisson_ratio)
{
  using namespace Eigen;

  // declare ElasticShell object
  double young_modulus = 1;
  double mass = 0;

  fsim::CompositeModel<fsim::DiscreteShell<>, fsim::NeoHookeanMembrane> model(
      fsim::DiscreteShell<>(P, F, thickness, young_modulus, poisson_ratio),
      fsim::NeoHookeanMembrane(V, F, thickness, young_modulus, poisson_ratio, mass));

  // declare NewtonSolver object
  optim::NewtonSolver<double> solver;
  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
  solver.options.fixed_dofs = fixed_idx;
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be

  solver.solve(model, P.reshaped<RowMajor>());

  MatrixXd NV = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3);
  return NV;
}

Eigen::MatrixXd
simulate_shell_timestep(const Eigen::MatrixXd &V,
                        const Eigen::MatrixXd &P,
                        const Eigen::MatrixXd &X,
                        const Eigen::MatrixXi &F,
                        double young_modulus,
                        double thickness,
                        double poisson_ratio,
                        double timestep)
{
  using namespace Eigen;

  // build lumped mass matrix
  geometrycentral::surface::ManifoldSurfaceMesh mesh(F);
  geometrycentral::surface::VertexPositionGeometry geometry(mesh, P);
  geometry.requireVertexLumpedMassMatrix();
  SparseMatrix<double> per_vertex_mass = geometry.vertexLumpedMassMatrix;

  // Convert the per-vertex mass matrix to a per-DOF mass matrix
  std::vector<Triplet<double>> triplets;
  for (int k = 0; k < per_vertex_mass.outerSize(); ++k)
  {
    for (SparseMatrix<double>::InnerIterator it(per_vertex_mass, k); it; ++it)
    {
      triplets.emplace_back(3 * it.row() + 0, 3 * it.col() + 0, it.value());
      triplets.emplace_back(3 * it.row() + 1, 3 * it.col() + 1, it.value());
      triplets.emplace_back(3 * it.row() + 2, 3 * it.col() + 2, it.value());
    }
  }
  SparseMatrix<double> M(X.size(), X.size());
  M.setFromTriplets(triplets.begin(), triplets.end());

  // declare ElasticShell object
  double mass = 0;

  fsim::CompositeModel<fsim::DiscreteShell<>, fsim::NeoHookeanMembrane> model(
      fsim::DiscreteShell<>(P, F, thickness, young_modulus, poisson_ratio),
      fsim::NeoHookeanMembrane(V, F, thickness, young_modulus, poisson_ratio, mass));

  // declare NewtonSolver object
  optim::NewtonSolver<double> solver;
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be

  VectorXd guess = VectorXd::Zero(X.size());
  VectorXd x = X.reshaped<RowMajor>();

  // // damping matrix
  // SparseMatrix<double> D = lambda * model.hessian(x) + mu * M;
  // auto energy = [&](const auto &dx) -> double
  // { return model.energy(x + timestep * dx) + 0.5 * dx.dot((M + timestep * D) * dx); };
  // auto grad = [&](const auto &dx) -> VectorXd
  // { return timestep * model.gradient(x + timestep * dx) + (M + timestep * D) * dx; };
  // auto hess = [&](const auto &dx) -> SparseMatrix<double>
  // { return timestep * timestep * model.hessian(x + timestep * dx) + M + timestep * D; };

  auto energy = [&](const auto &dx) -> double
  { return model.energy(x + timestep * dx) + 0.5 * dx.dot((M) * dx); };
  auto grad = [&](const auto &dx) -> VectorXd
  { return timestep * model.gradient(x + timestep * dx) + M * dx; };
  auto hess = [&](const auto &dx) -> SparseMatrix<double>
  { return timestep * timestep * model.hessian(x + timestep * dx) + M; };

  solver.solve(energy, grad, hess, guess);

  MatrixXd dX = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3);
  return X + timestep * dX;
}

NB_MODULE(fabsim_py, m)
{
  m.def("simulate_membrane", &simulate_membrane);
  m.def("simulate_shell", &simulate_shell);
  m.def("simulate_shell_timestep", &simulate_shell_timestep);
  m.def("compute_stretch_angles", &compute_stretch_angles);
  m.def("boundary_first_flattening", &boundary_first_flattening);
  m.def("compute_area_distortion", &compute_area_distortion);
}