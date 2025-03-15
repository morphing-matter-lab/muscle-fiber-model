#pragma once

#include <Eigen/Dense>
#include <nanobind/eigen/dense.h>

Eigen::VectorXd directional_fiber_stress(const Eigen::MatrixXd& V, const Eigen::MatrixXd& P, const Eigen::MatrixXd& F, double angle);
Eigen::MatrixXd fiber_stress(const Eigen::MatrixXd& V, const Eigen::MatrixXd& P, const Eigen::MatrixXd& F, int nb_directions);
void polymer_fraction_one_step(nanobind::DRef<Eigen::MatrixXd> polymer_fraction, const Eigen::MatrixXd &stress, double k0, double k1, double kd, double frac_f, double frac_s, double dt);
Eigen::MatrixXd directional_strain(const Eigen::MatrixXd &V, const Eigen::MatrixXd &P, const Eigen::MatrixXd &F, int n);
