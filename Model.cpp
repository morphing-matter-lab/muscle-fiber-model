#include "Model.h"
#include "newton.h"

#include <igl/massmatrix.h>
#include <numbers>

Model::Model(const nb::DRef<Eigen::MatrixXd>& V,
             const nb::DRef<Eigen::MatrixXi>& F,
             const nb::DRef<Eigen::MatrixXd>& Phi,
             const std::vector<int> &fixed_idx,
             double stretch_factor,
             double young_modulus,
             double poisson_ratio,
             double sigma_max,
             double e0,
             double e1,
             double damping,
             double dt)
    : _F{F},
      _Phi{Phi},
      _lambda{young_modulus * poisson_ratio / (1 - std::pow(poisson_ratio, 2))},
      _mu{0.5 * young_modulus / (1 + poisson_ratio)},
      _damping{damping},
      _dt{dt},
      _e0{e0},
      _e1{e1},
      _sigma_max{sigma_max}
{
  using namespace Eigen;
  using namespace std::numbers;

  for(int i = 0; i < F.rows(); ++i)
  {
    Eigen::Matrix3d Dm;
    Dm.col(0) = (V.row(F(i, 0)) - V.row(F(i, 3))) / stretch_factor;
    Dm.col(1) = (V.row(F(i, 1)) - V.row(F(i, 3))) / stretch_factor;
    Dm.col(2) = (V.row(F(i, 2)) - V.row(F(i, 3))) / stretch_factor;

    _DmInv.push_back(Dm.inverse());
  }

  _func = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));
  _func.add_elements<4>(
      TinyAD::range(_F.rows()), [&](auto& element) -> TINYAD_SCALAR_TYPE(element) {
    using T = TINYAD_SCALAR_TYPE(element);
    Eigen::Index f_idx = element.handle;

    Eigen::Matrix<T, 3, 3> Ds;
    Ds.col(0) = element.variables(_F(f_idx, 0)) - element.variables(_F(f_idx, 3));
    Ds.col(1) = element.variables(_F(f_idx, 1)) - element.variables(_F(f_idx, 3));
    Ds.col(2) = element.variables(_F(f_idx, 2)) - element.variables(_F(f_idx, 3));
    Eigen::Matrix<T, 3, 3> defo_gradient = Ds * _DmInv[f_idx];
    T J = defo_gradient.determinant();
    TINYAD_ASSERT_G(J, 0);

    double coeff = 1 / 6. * std::abs(_DmInv[f_idx].determinant());
    T total_energy = 0;
    const int n = _Phi.cols();
    for (int i = 0; i < n; ++i)
    {
      Vector3d u(cos(i * pi / n), sin(i * pi / n), 0);
      T e = 0.5 * ((defo_gradient * u).dot(defo_gradient * u) - 1);
            // total_energy += _Phi(f_idx, i) * 0.5 / inv_sqrtpi * _e0 * erf(e / _e0) / n;
      if (e >= 0)
        total_energy += _Phi(f_idx, i) * pow(e / _e1, 2) * e / 3;
    }
    total_energy *= _sigma_max;
    total_energy += _mu / 2 * ((defo_gradient.transpose() * defo_gradient).trace() - 3) - _mu * log(J) + _lambda / 2 * pow(log(J), 2);
    total_energy *= coeff;

    return total_energy;
  }); 

  // mass matrix
  SparseMatrix<double> per_vertex_mass;
  igl::massmatrix(V, _F, igl::MASSMATRIX_TYPE_DEFAULT, per_vertex_mass);

  // Convert the per-vertex mass matrix to a per-DOF mass matrix
  std::vector<Triplet<double>> triplets;
  for(int k = 0; k < per_vertex_mass.outerSize(); ++k)
  {
    for(SparseMatrix<double>::InnerIterator it(per_vertex_mass, k); it; ++it)
    {
      triplets.emplace_back(3 * it.row() + 0, 3 * it.col() + 0, it.value());
      triplets.emplace_back(3 * it.row() + 1, 3 * it.col() + 1, it.value());
      triplets.emplace_back(3 * it.row() + 2, 3 * it.col() + 2, it.value());
    }
  }
  _M = SparseMatrix<double>(V.size(), V.size());
  _M.setFromTriplets(triplets.begin(), triplets.end());

  // projection matrix (fix DOFs)
  std::vector<int> orderedIdx = fixed_idx;
  std::sort(orderedIdx.begin(), orderedIdx.end());
  _P = projectionMatrix(orderedIdx, V.size());

  // restrict _M to free variables
  _M = (_P * _M * _P.transpose()).eval();
}

Eigen::VectorXd Model::force(const Eigen::VectorXd& x, const Eigen::VectorXd& v)
{  
  auto [f, g, K] = _func.eval_with_hessian_proj(x);
  K = (_P * K * _P.transpose()).eval();

  return -_damping * (_M + K) * _P * v -_P * g;
}

Eigen::VectorXd Model::residual(const Eigen::VectorXd& x, const Eigen::VectorXd& v, const Eigen::VectorXd& a)
{
  // std::cout << (_P * a).norm() << " " << (_P * v).norm() << " " << force(x, v).norm() << "\n";

  return _M * _P * a - force(x, v);
}

Eigen::VectorXd Model::acceleration(const Eigen::VectorXd& x, const Eigen::VectorXd& v)
{
  _solver.compute(_M);
  return _solver.solve(force(x, v));
}

Eigen::VectorXd
Model::solve_residual(const Eigen::VectorXd& x, const Eigen::VectorXd& v, const Eigen::VectorXd& a)
{
  using namespace Eigen;

  auto [f, g, K] = _func.eval_with_hessian_proj(x);
  K = (_P * K * _P.transpose()).eval();
  VectorXd r = _M * _P * a + _damping * (_M + K) * _P * v + _P * g;

  _solver.compute(_M + _dt * _damping * (_M + K) + _dt * _dt / 4 * K);
  if(_solver.info() != Eigen::Success)
    std::cout << "Error\n";
  return _solver.solve(-r);
}

Eigen::SparseMatrix<double> Model::stiffness_matrix(const Eigen::VectorXd& x, const Eigen::VectorXd& v)
{
  auto [f, g, K] = _func.eval_with_hessian_proj(x);

  // restrict K to free variables
  return _P * K * _P.transpose();
}

void Model::solve_timestep_newmark(nb::DRef<Eigen::VectorXd> x, nb::DRef<Eigen::VectorXd> v, nb::DRef<Eigen::VectorXd> a)
{
  x = x + _dt * v + _dt * _dt * a;
  v = v + _dt * a;

  double f_norm = force(x, v).norm();
  while(residual(x, v, a).norm() / f_norm > 1e-6)
  {
    Eigen::VectorXd da = _P.transpose() * solve_residual(x, v, a);

    a = a + da;
    v = v + _dt * da / 2;
    x = x + _dt * _dt * da / 4;
    std::cout << residual(x, v, a).norm() << "\n";
  }
}
