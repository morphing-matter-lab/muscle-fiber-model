#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/tuple.h>

#include <optim/NewtonSolver.h>
#include <fsim/ElasticMembrane.h>

#include <iostream>

namespace nb = nanobind;
using namespace nb::literals;

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>
simulate_membrane(const Eigen::MatrixXd &V,
          const Eigen::MatrixXi& F,
          const std::vector<int>& fixed_idx,
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

  solver.solve(model, V.reshaped());

  MatrixXd NV = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3);
  return std::make_tuple(V, F);
}

NB_MODULE(fabsim_py, m)
{
  m.def("simulate_membrane", &simulate_membrane);
}