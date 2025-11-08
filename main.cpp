#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>

#include <optim/NewtonSolver.h>
#include <optim/filter_var.h>
#include <fsim/CompositeModel.h>
#include <fsim/ElasticMembrane.h>
#include <fsim/util/finite_differences.h>

#include "TinyAD/Utils/Helpers.hh"
#include "TinyAD/ScalarFunction.hh"

#include <igl/barycentric_coordinates.h>
#include <igl/boundary_loop.h>
#include <igl/triangle/triangulate.h>

#include <iostream>
#include <numbers>

#include "FiberStress.h"
#include "newton.h"
#include "distance.h"
#include "MuscleTissueModel.h"
#include "data_to_mesh.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std::numbers;

// Eigen::VectorXd sensitivity_gradient(nb::DRef<Eigen::MatrixXd> V,
//                                   const nb::DRef<Eigen::MatrixXd> &P,
//                                   const nb::DRef<Eigen::MatrixXi> &F,
//                                   const nb::DRef<Eigen::MatrixXd> &Phi,
//                                   const nb::DRef<Eigen::VectorXd> &distance_grad,
//                                   const std::vector<int> &fixed_idx,
//                                   double stretch,
//                                   double poisson_ratio,
//                                   double sigma_max)
// {
//   using namespace Eigen;

//   const double young_modulus = 1;
//   MuscleTissueModel model(P, F, Phi, young_modulus, poisson_ratio, stretch, sigma_max);
//   LLTSolver solver;
//   SparseMatrix<double> H = model.hessian(V.reshaped<RowMajor>());
//   filter_var(H, fixed_idx);

//   solver.compute(H);
//   if (solver.info() != Success)
//     std::cout << "Factorization failed.\n";

//   VectorXd distance_grad_cp(distance_grad);
//   filter_var(distance_grad_cp, fixed_idx);
//   VectorXd res = -solver.solve(distance_grad_cp);

//   VectorXd gradient_stretch = model.getModel<0>().gradient_derivative_sensitivity(V.reshaped<RowMajor>(), stretch);
//   VectorXd gradient_sigma = model.getModel<1>().gradient_derivative_sensitivity(V.reshaped<RowMajor>());
//   filter_var(gradient_stretch, fixed_idx);
//   filter_var(gradient_sigma, fixed_idx);

//   return Vector2d(res.dot(gradient_stretch), res.dot(gradient_sigma));
// }

Eigen::MatrixXd model_hessian_finite_differences(nb::DRef<Eigen::MatrixXd> V,
                                                 const nb::DRef<Eigen::MatrixXd> &P,
                                                 const nb::DRef<Eigen::MatrixXi> &F,
                                                 const nb::DRef<Eigen::MatrixXd> &Phi,
                                                 const std::vector<int> &fixed_idx,
                                                 double stretch,
                                                 double poisson_ratio,
                                                 double sigma_max)
{
  using namespace Eigen;

  const double young_modulus = 1;
  MuscleTissueModel model(P, F, Phi, young_modulus, poisson_ratio, stretch, sigma_max);

  SparseMatrix<double> H = fsim::finite_differences_sparse([&](const VectorXd &X)
                                                           { return model.gradient(X); }, V.reshaped<RowMajor>());
  filter_var(H, fixed_idx);

  return MatrixXd(H);
}

Eigen::MatrixXd model_hessian(nb::DRef<Eigen::MatrixXd> V,
                              const nb::DRef<Eigen::MatrixXd> &P,
                              const nb::DRef<Eigen::MatrixXi> &F,
                              const nb::DRef<Eigen::MatrixXd> &Phi,
                              const std::vector<int> &fixed_idx,
                              double stretch,
                              double poisson_ratio,
                              double sigma_max)
{
  using namespace Eigen;

  const double young_modulus = 1;
  MuscleTissueModel model(P, F, Phi, young_modulus, poisson_ratio, stretch, sigma_max);

  SparseMatrix<double> H = model.hessian(V.reshaped<RowMajor>());
  // filter_var(H, fixed_idx);

  return MatrixXd(H).selfadjointView<Upper>();
}

// Eigen::MatrixXd sensitivity_matrix(nb::DRef<Eigen::MatrixXd> V,
//                                   const nb::DRef<Eigen::MatrixXd> &P,
//                                   const nb::DRef<Eigen::MatrixXi> &F,
//                                   const nb::DRef<Eigen::MatrixXd> &Phi,
//                                   const std::vector<int> &fixed_idx,
//                                   double stretch,
//                                   double poisson_ratio,
//                                   double sigma_max)
// {
//   using namespace Eigen;

//   const double young_modulus = 1;
//   MuscleTissueModel model(P, F, Phi, young_modulus, poisson_ratio, stretch, sigma_max);
//   LLTSolver solver;
//   SparseMatrix<double> H = model.hessian(V.reshaped<RowMajor>());

//   // Build matrix proj
//   Eigen::SparseMatrix<double> proj = projectionMatrix(fixed_idx, V.size());
//   H = (proj * H * proj.transpose()).eval();

//   solver.compute(H);
//   if (solver.info() != Success)
//     std::cout << "Factorization failed.\n";

//   VectorXd gradient_stretch = proj * model.getModel<0>().gradient_derivative_sensitivity(V.reshaped<RowMajor>(), stretch);
//   VectorXd gradient_sigma = proj * model.getModel<1>().gradient_derivative_sensitivity(V.reshaped<RowMajor>());

//   MatrixXd res(V.size(), 2);
//   res.col(0) = -proj.transpose() * solver.solve(gradient_stretch);
//   res.col(1) = -proj.transpose() * solver.solve(gradient_sigma);

//   return res;
// }

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
      angles[i] += pi;
    else
      angles[i] -= pi;
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
                       const nb::DRef<Eigen::VectorXd> &theta0,
                       const nb::DRef<Eigen::VectorXd> &eta,
                       const nb::DRef<Eigen::MatrixXd> &phi,
                       const std::vector<int> &fixed_idx,
                       double stretch_factor,
                       double poisson_ratio,
                       double sigma_max)
{
  using namespace Eigen;

  const double young_modulus = 1;
  if (theta0.size() != 0 && eta.size() != 0 && phi.cols() == 1)
  {
    // declare NeohookeanMembrane object
    MuscleTissueModel model(P, F, theta0, eta, phi, young_modulus, poisson_ratio, stretch_factor, sigma_max);

    // declare NewtonSolver object
    optim::NewtonSolver<double> solver;
    // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
    solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be
    solver.options.fixed_dofs = fixed_idx;
    // solver.options.display = optim::SolverDisplay::quiet;

    solver.solve(model, V.reshaped<RowMajor>());

    V = Map<fsim::Mat2<double>>(solver.var().data(), V.rows(), 2);
  }
  else if (phi.size() != 0 && phi.cols() == 2)
  {
    // declare NeohookeanMembrane object
    MuscleTissueModel model(P, F, phi, young_modulus, poisson_ratio, stretch_factor, sigma_max);

    // declare NewtonSolver object
    optim::NewtonSolver<double> solver;
    // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
    solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be
    solver.options.fixed_dofs = fixed_idx;
    // solver.options.display = optim::SolverDisplay::quiet;

    solver.solve(model, V.reshaped<RowMajor>());

    V = Map<fsim::Mat2<double>>(solver.var().data(), V.rows(), 2);
  }
  else
  {
    // declare NeohookeanMembrane object
    MuscleTissueModel model(P, F, VectorXd::Zero(F.rows()), VectorXd::Zero(F.rows()), VectorXd::Ones(F.rows()), young_modulus, poisson_ratio, stretch_factor, 0.);

    // declare NewtonSolver object
    optim::NewtonSolver<double> solver;
    // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
    solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be
    solver.options.fixed_dofs = fixed_idx;
    // solver.options.display = optim::SolverDisplay::quiet;

    solver.solve(model, V.reshaped<RowMajor>());

    V = Map<fsim::Mat2<double>>(solver.var().data(), V.rows(), 2);
  }
}

Eigen::VectorXd I5(const nb::DRef<Eigen::MatrixXd> &V,
                   const nb::DRef<Eigen::MatrixXd> &P,
                   const nb::DRef<Eigen::MatrixXi> &F,
                   const nb::DRef<Eigen::MatrixXd> &orientations,
                   const nb::DRef<Eigen::VectorXd> &eta,
                   double stretch_factor)
{
  using namespace Eigen;

  // declare NeohookeanMembrane object
  double young_modulus = 1;
  double poisson_ratio = 0.49;
  double sigma_max = 1;
  MatrixXd Phi = MatrixXd::Zero(F.rows(), 2);
  MuscleTissueModel model(P, F, Phi, young_modulus, poisson_ratio, stretch_factor, sigma_max);

  VectorXd res(model._elements.size());
  for (int i = 0; i < model._elements.size(); ++i)
  {
    Matrix2d F = model._elements[i].deformationGradient(V.reshaped<RowMajor>(), stretch_factor);
    Matrix2d C = F.transpose() * F;
    Vector2d u = orientations.row(i);
    res(i) = eta(i) * C.trace() + (1 - 2 * eta(i)) * (F * u).squaredNorm();
  }

  return res;
}

Eigen::MatrixXd theta0(const nb::DRef<Eigen::MatrixXd> &V,
                       const nb::DRef<Eigen::MatrixXd> &P,
                       const nb::DRef<Eigen::MatrixXi> &F,
                       double stretch_factor)
{
  using namespace Eigen;

  // declare NeohookeanMembrane object
  double young_modulus = 1;
  double poisson_ratio = 0.49;
  double sigma_max = 1;
  MatrixXd Phi = MatrixXd::Zero(F.rows(), 2);
  MuscleTissueModel model(P, F, Phi, young_modulus, poisson_ratio, stretch_factor, sigma_max);

  return model.theta0(V.reshaped<RowMajor>());
}

void phi_ode(nb::DRef<Eigen::MatrixXd> Phi,
             nb::DRef<Eigen::MatrixXd> V,
             const nb::DRef<Eigen::MatrixXd> &P,
             const nb::DRef<Eigen::MatrixXi> &F,
             const std::vector<int> &fixed_idx,
             double stretch_factor,
             double poisson_ratio,
             double sigma_max,
             double k0,
             double k1,
             double kd,
             double dt,
             int n)
{
  using namespace Eigen;
  double young_modulus = 1;
    // declare NeohookeanMembrane object
  MuscleTissueModel model(P, F, Phi, young_modulus, poisson_ratio, stretch_factor, sigma_max);
  model.phi_ODE(V.reshaped<Eigen::RowMajor>(), k0, k1, kd, dt, n);

  // declare NewtonSolver object
  optim::NewtonSolver<double> solver;
  // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
  solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be
  solver.options.fixed_dofs = fixed_idx;
  // solver.options.display = optim::SolverDisplay::quiet;

  solver.solve([&](const VectorXd& x) { 
                model.phi_ODE(V.reshaped<Eigen::RowMajor>(), k0, k1, kd, dt, n);
                return model.energy(x); },
                [&model](const VectorXd& x) { return model.gradient(x); },
                [&model](const VectorXd& x) { return model.hessian(x); }, V.reshaped<RowMajor>());

  V = Map<fsim::Mat2<double>>(solver.var().data(), V.rows(), 2);
}

void update_phi(nb::DRef<Eigen::MatrixXd> V,
                const nb::DRef<Eigen::MatrixXd> &P,
                const nb::DRef<Eigen::MatrixXi> &F,
                nb::DRef<Eigen::VectorXd> theta0,
                const nb::DRef<Eigen::VectorXd> &eta,
                const nb::DRef<Eigen::VectorXd> &phi,
                const std::vector<int> &fixed_idx,
                double stretch_factor,
                double poisson_ratio,
                double sigma_max)
{
  using namespace Eigen;

  // declare NeohookeanMembrane object
  double young_modulus = 1;

  MuscleTissueModel model(P, F, theta0, eta, phi, young_modulus, poisson_ratio, stretch_factor, sigma_max);

  theta0 += model.updatePhi(V.reshaped<RowMajor>());
}

// Eigen::VectorXd model_gradient(nb::DRef<Eigen::MatrixXd> V,
//                        const nb::DRef<Eigen::MatrixXd> &P,
//                        const nb::DRef<Eigen::MatrixXi> &F,
//                        const nb::DRef<Eigen::MatrixXd> &Phi,
//                        const std::vector<int> &fixed_idx,
//                        double stretch_factor,
//                        double poisson_ratio,
//                        double sigma_max)
// {
//   using namespace Eigen;

//   // declare NeohookeanMembrane object
//   double young_modulus = 1;

//   MuscleTissueModel model(P, F, Phi, young_modulus, poisson_ratio, stretch_factor, sigma_max);

//   VectorXd grad = model.getModel<0>().gradient_derivative_sensitivity(V.reshaped<RowMajor>(), stretch_factor);
//   // filter_var(grad, fixed_idx);

//   return grad;
// }

Eigen::VectorXd model_gradient(nb::DRef<Eigen::MatrixXd> V,
                               const nb::DRef<Eigen::MatrixXd> &P,
                               const nb::DRef<Eigen::MatrixXi> &F,
                               const nb::DRef<Eigen::MatrixXd> &Phi,
                               const std::vector<int> &fixed_idx,
                               double stretch_factor,
                               double poisson_ratio,
                               double sigma_max)
{
  using namespace Eigen;
  double young_modulus = 1;

  MuscleTissueModel model(P, F, Phi, young_modulus, poisson_ratio, stretch_factor, sigma_max);

  return model.gradient(V.reshaped<RowMajor>());
}

Eigen::VectorXd model_gradient_finite_differences(nb::DRef<Eigen::MatrixXd> V,
                                                  const nb::DRef<Eigen::MatrixXd> &P,
                                                  const nb::DRef<Eigen::MatrixXi> &F,
                                                  const nb::DRef<Eigen::MatrixXd> &Phi,
                                                  const std::vector<int> &fixed_idx,
                                                  double stretch_factor,
                                                  double poisson_ratio,
                                                  double sigma_max)
{
  using namespace Eigen;
  double young_modulus = 1;

  MuscleTissueModel model(P, F, Phi, young_modulus, poisson_ratio, stretch_factor, sigma_max);

  return fsim::finite_differences([&](const VectorXd &X)
                                  { return model.energy(X); }, V.reshaped<RowMajor>());
}

Eigen::VectorXd distance_finite_differences(const Eigen::MatrixXd &V, const std::vector<int> &indices, const Eigen::MatrixXd &distanceMap)
{
  using namespace Eigen;

  return fsim::finite_differences([&](const VectorXd &X)
                                  { return distance(X.reshaped<RowMajor>(V.rows(), 2), indices, distanceMap); }, V.reshaped<RowMajor>());
}

void simulate3D(nb::DRef<Eigen::MatrixXd> NV,
                const nb::DRef<Eigen::MatrixXd> &V,
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

  TinyAD::ScalarFunction<3, double, Eigen::Index> func = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));

  const double young_modulus = 1;
  const double lambda = young_modulus * poisson_ratio / (1 - std::pow(poisson_ratio, 2));
  const double mu = 0.5 * young_modulus / (1 + poisson_ratio);

  std::vector<Eigen::Matrix3d> DmInv(F.rows());

  for (int i = 0; i < F.rows(); ++i)
  {
    Eigen::Matrix3d Dm;
    Dm.col(0) = (V.row(F(i, 0)) - V.row(F(i, 3))) / stretch_factor;
    Dm.col(1) = (V.row(F(i, 1)) - V.row(F(i, 3))) / stretch_factor;
    Dm.col(2) = (V.row(F(i, 2)) - V.row(F(i, 3))) / stretch_factor;

    DmInv[i] = Dm.inverse();
  }

  func.add_elements<4>(
      TinyAD::range(F.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element)
      {
    using T = TINYAD_SCALAR_TYPE(element);
    Eigen::Index f_idx = element.handle;

    Eigen::Matrix<T, 3, 3> Ds;
    Ds.col(0) = element.variables(F(f_idx, 0)) - element.variables(F(f_idx, 3));
    Ds.col(1) = element.variables(F(f_idx, 1)) - element.variables(F(f_idx, 3));
    Ds.col(2) = element.variables(F(f_idx, 2)) - element.variables(F(f_idx, 3));
    Eigen::Matrix<T, 3, 3> defo_gradient = Ds * DmInv[f_idx];
    T J = defo_gradient.determinant();
    TINYAD_ASSERT_G(J, 0);

    double coeff = 1 / 6. * std::abs(DmInv[f_idx].determinant());
    T total_energy = 0;
    const int n = Phi.cols();
    for (int i = 0; i < n; ++i)
    {
      Vector3d u(cos(i * pi / n), sin(i * pi / n), 0);
      T e = 0.5 * ((defo_gradient * u).dot(defo_gradient * u) - 1);
            // total_energy += Phi(f_idx, i) * 0.5 / inv_sqrtpi * e0 * erf(e / e0) / n;
      if (e >= 0)
        total_energy += Phi(f_idx, i) * pow(e / e1, 2) * e / 3;
    }
    total_energy *= sigma_max;
    total_energy += mu / 2 * ((defo_gradient.transpose() * defo_gradient).trace() - 3) - mu * log(J) + lambda / 2 * pow(log(J), 2);
    total_energy *= coeff;

    return total_energy; });

  Eigen::VectorXd x = NV.reshaped<Eigen::RowMajor>();

  LLTSolver solver;
  newton(x, func, solver, 10, 1e-6, true, fixed_idx);

  NV = x.reshaped<Eigen::RowMajor>(V.rows(), 3);
}

Eigen::MatrixXd barycentric_coordinates(const Eigen::MatrixXd &P, const Eigen::MatrixXd &NV, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
{
  using namespace Eigen;

  MatrixXd VA(NV.rows(), NV.cols()), VB(NV.rows(), NV.cols()), VC(NV.rows(), NV.cols());
  MatrixXd PA(NV.rows(), NV.cols()), PB(NV.rows(), NV.cols()), PC(NV.rows(), NV.cols());

  for (int i = 0; i < NV.rows(); ++i)
  {
    for (int j = 0; j < F.rows(); ++j)
    {
      if (is_point_in_triangle(NV.row(i), F.row(j), V))
      {
        VA.row(i) = V.row(F(j, 0));
        VB.row(i) = V.row(F(j, 1));
        VC.row(i) = V.row(F(j, 2));

        PA.row(i) = P.row(F(j, 0));
        PB.row(i) = P.row(F(j, 1));
        PC.row(i) = P.row(F(j, 2));
        break;
      }
      if (j == F.rows() - 1)
        std::cout << "Point " << i << " pos: " << NV.row(i) << " outside mesh\n";
    }
  }

  MatrixXd L;
  igl::barycentric_coordinates(NV, VA, VB, VC, L);

  return L.col(0).asDiagonal() * PA + L.col(1).asDiagonal() * PB + L.col(2).asDiagonal() * PC;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXi> remesh(const nb::DRef<Eigen::MatrixXd> &V,
                                                                     const nb::DRef<Eigen::MatrixXd> &P,
                                                                     const nb::DRef<Eigen::MatrixXi> &F,
                                                                     const std::string &flags)
{
  using namespace Eigen;

  MatrixXd BV; // boundary vertices
  MatrixXi BE; // boundary edges

  std::vector<std::vector<Index>> L;
  igl::boundary_loop(F, L);

  for (auto &list : L)
  {
    int n = BV.rows();
    BV.conservativeResize(n + list.size(), 2);
    BE.conservativeResize(n + list.size(), 2);

    for (int i = 0; i < list.size(); ++i)
    {
      BV.row(n + i) << V(list[i], 0), V(list[i], 1);
      BE.row(n + i) << n + i, n + ((i + 1) % list.size());
    }
  }

  MatrixXd H(2, 2); // hole positions
  H.row(0) << -2.275, 0;
  H.row(1) << 2.275, 0;
  MatrixXd V2; // new vertices
  MatrixXi F2; // new faces
  igl::triangle::triangulate(BV, BE, H, flags, V2, F2);

  V2.conservativeResize(V2.rows(), 3);
  V2.col(2).setZero();

  MatrixXd P2 = barycentric_coordinates(P, V2, V, F);

  return std::make_tuple(V2, P2, F2);
}

NB_MODULE(fabsim_py, m)
{
  m.def("simulate_membrane", &simulate_membrane);
  m.def("remesh", &remesh);
  m.def("boundary_loops", [](const nb::DRef<Eigen::MatrixXi> &F)
        { 
    std::vector<std::vector<Eigen::Index>> L;
    igl::boundary_loop(F, L);
    return L; });
  m.def("triangulate", [](const nb::DRef<Eigen::MatrixXd> &P, const nb::DRef<Eigen::MatrixXi> &E, const nb::DRef<Eigen::MatrixXi> &H, const std::string &flags)
        {
    Eigen::MatrixXd V2; // new vertices
    Eigen::MatrixXi F2; // new faces
    igl::triangle::triangulate(P, E, H, flags, V2, F2);
    return std::make_tuple(V2, F2); });
  m.def("simulate3D", &simulate3D);
  m.def("compute_membrane_energies", &compute_membrane_energies);
  m.def("compute_membrane_forces", &compute_membrane_forces);
  m.def("compute_stretch_angles", &compute_stretch_angles);
  m.def("directional_fiber_stress", &directional_fiber_stress);
  m.def("directional_strain", &directional_strain);
  m.def("fiber_stress", &fiber_stress, "V"_a, "P"_a, "F"_a, "n"_a, "e0"_a = 1.2e-1, "e1"_a = 1.7e-1);
  m.def("polymer_fraction_one_step", &polymer_fraction_one_step);
  m.def("polymer_fraction_steady_state", &polymer_fraction_steady_state);
  m.def("transfer_data_to_3D_mesh", &transfer_data_to_3D_mesh);
  m.def("image_data_to_mesh", &image_data_to_mesh);
  m.def("orientation_data_to_mesh", &orientation_data_to_mesh);
  m.def("distance", &distance);
  m.def("distance_gradient", &distanceGrad);
  m.def("histogram_data_to_mesh", &histogram_data_to_mesh);
  // m.def("sensitivity_matrix", &sensitivity_matrix);
  // m.def("sensitivity_gradient", &sensitivity_gradient);
  m.def("model_hessian_finite_differences", &model_hessian_finite_differences);
  m.def("model_hessian", &model_hessian);
  m.def("model_gradient", &model_gradient);
  // m.def("model_gradient", &model_gradient);
  m.def("model_gradient_finite_differences", &model_gradient_finite_differences);
  m.def("update_Phi", &update_phi);
  m.def("I5", &I5);
  m.def("theta0", &theta0);
  m.def("phi_ode", &phi_ode);
}