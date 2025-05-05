#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#include <fsim/CompositeModel.h>
#include <fsim/ElasticMembrane.h>
#include <fsim/ElasticShell.h>

#include <TinyAD/ScalarFunction.hh>
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;

class Model
{
public:
  Model(const nb::DRef<Eigen::MatrixXd>& P,
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
    double dt);

  Eigen::VectorXd force(const Eigen::VectorXd &x, const Eigen::VectorXd &v);
  Eigen::VectorXd residual(const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &a);
  Eigen::VectorXd solve_residual(const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &a);
  Eigen::VectorXd acceleration(const Eigen::VectorXd &x, const Eigen::VectorXd &v);
  Eigen::SparseMatrix<double> stiffness_matrix(const Eigen::VectorXd &x, const Eigen::VectorXd &v);
  void solve_timestep_newmark(nb::DRef<Eigen::VectorXd> x, nb::DRef<Eigen::VectorXd> v, nb::DRef<Eigen::VectorXd> a);
  // Eigen::VectorXd newmark_predict(const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &a);


private:
  Eigen::SparseMatrix<double> _M;
  Eigen::SparseMatrix<double> _P;
  Eigen::MatrixXi _F;
  Eigen::MatrixXd _Phi;
  double _lambda;
  double _mu;
  double _damping;
  double _dt;
  double _e0, _e1, _sigma_max;
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> _solver;
  TinyAD::ScalarFunction<3, double, Eigen::Index> _func;
  std::vector<Eigen::Matrix3d> _DmInv;
};