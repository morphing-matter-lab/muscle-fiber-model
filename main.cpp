#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>

#include <optim/NewtonSolver.h>
#include <fsim/CompositeModel.h>
#include <fsim/ElasticMembrane.h>

#include "TinyAD/Utils/Helpers.hh"
#include "TinyAD/ScalarFunction.hh"

#include <igl/barycentric_coordinates.h>
#include <igl/boundary_loop.h>
#include <igl/triangle/triangulate.h>

#include <iostream>
#include <numbers>

#include "PillarModel.h"
#include "FiberStress.h"
#include "ActinBundle.h"
#include "newton.h"
#include "Model.h"
#include "distance.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std::numbers;

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

bool is_point_in_triangle(const Eigen::VectorXd &p, const Eigen::Vector3i &face, const Eigen::MatrixXd &V)
{
  // compute barycentric coordinates
  double detT = (V(face(0), 0) - V(face(2), 0)) * (V(face(1), 1) - V(face(2), 1)) - (V(face(0), 1) - V(face(2), 1)) * (V(face(1), 0) - V(face(2), 0));
  double u = ((p(0) - V(face(2), 0)) * (V(face(1), 1) - V(face(2), 1)) - (p(1) - V(face(2), 1)) * (V(face(1), 0) - V(face(2), 0))) / detT;
  double v = ((V(face(0), 0) - V(face(2), 0)) * (p(1) - V(face(2), 1)) - (V(face(0), 1) - V(face(2), 1)) * (p(0) - V(face(2), 0))) / detT;
  double w = ((V(face(0), 0) - p(0)) * (V(face(1), 1) - p(1)) - (V(face(0), 1) - p(1)) * (V(face(1), 0) - p(0))) / detT;

  return (u >= -1e-6 && v >= -1e-6 && 1 - u - v >= -1e-6);
}

Eigen::MatrixXd transfer_data_to_3D_mesh(const nb::DRef<Eigen::MatrixXd> &V,
                                         const nb::DRef<Eigen::MatrixXi> &F,
                                         const nb::DRef<Eigen::MatrixXd> &Phi,
                                         const nb::DRef<Eigen::MatrixXd> &V_3D,
                                         const nb::DRef<Eigen::MatrixXi> &F_3D)
{
  using namespace Eigen;

  MatrixXd PhiV3D = MatrixXd::Zero(V_3D.rows(), Phi.cols());
  for (int i = 0; i < V_3D.rows(); ++i)
  {
    for (int j = 0; j < F.rows(); ++j)
    {
      if (is_point_in_triangle(V_3D.row(i), F.row(j), V))
      {
        PhiV3D.row(i) = Phi.row(j);
        break;
      }
      if (j == F.rows() - 1)
        std::cout << "Point " << i << " pos: " << V_3D.row(i) << " outside mesh\n";
    }
  }

  MatrixXd Phi_3D(F_3D.rows(), Phi.cols());
  for (int i = 0; i < F_3D.rows(); ++i)
  {
    Phi_3D.row(i) = (PhiV3D.row(F_3D(i, 0)) + PhiV3D.row(F_3D(i, 1)) + PhiV3D.row(F_3D(i, 2)) + PhiV3D.row(F_3D(i, 3))) / 4;
  }

  return Phi_3D;
}

Eigen::MatrixXd histogram_data_to_mesh(const nb::DRef<Eigen::MatrixXd> &V,
                                       const nb::DRef<Eigen::MatrixXi> &F,
                                       const nb::DRef<Eigen::MatrixXd> &image,
                                       double world_coords_to_px,
                                       double radius,
                                       int n)
{
  using namespace Eigen;

  MatrixXd Phi = MatrixXd::Zero(V.rows(), n);
  int nX = image.cols();
  int nY = image.rows();

  const double sigma = 0.3;

  //  lower normal part of gaussian
  const double normal = 1 / std::sqrt(2 * pi) / sigma;

  for (int i = 0; i < V.rows(); ++i)
  {
    double minX = V(i, 0) - radius;
    double maxX = V(i, 0) + radius;
    double minY = V(i, 1) - radius;
    double maxY = V(i, 1) + radius;

    int minIdxX = std::clamp(int(std::floor(world_coords_to_px * minX + nX / 2)), 0, nX);
    int minIdxY = std::clamp(int(std::floor(-world_coords_to_px * maxY + nY / 2)), 0, nY);
    int maxIdxX = std::clamp(int(std::ceil(world_coords_to_px * maxX + nX / 2)), 0, nX);
    int maxIdxY = std::clamp(int(std::ceil(-world_coords_to_px * minY + nY / 2)), 0, nY);

    for (int x = minIdxX; x < maxIdxX; ++x)
    {
      for (int y = minIdxY; y < maxIdxY; ++y)
      {
        Vector2d coords;
        coords << x - nX / 2, -y + nY / 2;
        coords /= world_coords_to_px;

        if (image(y, x) != 0)
        {
          int k = int(std::round(image(y, x) * n / pi + n / 2)) % n; // rotate the data with + n / 2
          double dist = (V.block<1, 2>(i, 0) - coords.transpose()).norm();
          Phi(i, k) += std::exp(-std::pow(dist / radius, 2) / 2 / std::pow(sigma, 2)) * normal;
        }
      }
    }
  }

  return Phi;
}

Eigen::MatrixXd image_data_to_mesh(const nb::DRef<Eigen::MatrixXd> &V,
                                   const nb::DRef<Eigen::MatrixXi> &F,
                                   const nb::DRef<Eigen::MatrixXd> &image,
                                   double world_coords_to_px)
{
  using namespace Eigen;

  VectorXd res = VectorXd::Zero(V.rows());
  VectorXd count = VectorXd::Zero(V.rows());

  int nX = image.cols();
  int nY = image.rows();

  for (int i = 0; i < F.rows(); ++i)
  {
    double minX = V(F(i, 0), 0);
    double maxX = V(F(i, 0), 0);
    double minY = V(F(i, 0), 1);
    double maxY = V(F(i, 0), 1);

    for (int j = 1; j < 3; ++j)
    {
      if (V(F(i, j), 0) < minX)
        minX = V(F(i, j), 0);
      if (V(F(i, j), 0) > maxX)
        maxX = V(F(i, j), 0);
      if (V(F(i, j), 1) < minY)
        minY = V(F(i, j), 1);
      if (V(F(i, j), 1) > maxY)
        maxY = V(F(i, j), 1);
    }

    int minIdxX = std::clamp(int(std::floor(world_coords_to_px * minX + nX / 2)), 0, nX);
    int minIdxY = std::clamp(int(std::floor(-world_coords_to_px * maxY + nY / 2)), 0, nY);
    int maxIdxX = std::clamp(int(std::ceil(world_coords_to_px * maxX + nX / 2)), 0, nX);
    int maxIdxY = std::clamp(int(std::ceil(-world_coords_to_px * minY + nY / 2)), 0, nY);

    int k = 0;

    for (int x = minIdxX; x < maxIdxX; ++x)
    {
      for (int y = minIdxY; y < maxIdxY; ++y)
      {
        Vector2d coords;
        coords << x - nX / 2, -y + nY / 2;
        coords /= world_coords_to_px;

        if (is_point_in_triangle(coords, F.row(i), V))
        {
          if (image(y, x) != 0)
          {
            res(F(i, 0)) += image(y, x);
            res(F(i, 1)) += image(y, x);
            res(F(i, 2)) += image(y, x);
            k += 1;
          }
        }
      }
    }
    count(F(i, 0)) += k;
    count(F(i, 1)) += k;
    count(F(i, 2)) += k;
  }

  for(int i = 0; i < V.rows(); ++i)
    if(count(i) > 0)
      res(i) = res(i) / count(i);

  return res;
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
  m.def("distance", &distance);
  m.def("distance_gradient", &distanceGrad);
  m.def("histogram_data_to_mesh", &histogram_data_to_mesh);
  nb::class_<Model>(m, "Model")
      .def(nb::init<const nb::DRef<Eigen::MatrixXd> &,
                    const nb::DRef<Eigen::MatrixXi> &,
                    const nb::DRef<Eigen::MatrixXd> &,
                    const std::vector<int> &,
                    double,
                    double,
                    double,
                    double,
                    double,
                    double,
                    double,
                    double>())
      .def("force", &Model::force)
      .def("residual", &Model::residual)
      .def("acceleration", &Model::acceleration)
      .def("solve_timestep_newmark", &Model::solve_timestep_newmark);
}