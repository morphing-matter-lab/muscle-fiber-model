#pragma once

#include <Eigen/Dense>

Eigen::VectorXd directional_fiber_stress(const Eigen::MatrixXd& V, const Eigen::MatrixXd& P, const Eigen::MatrixXd& F, double angle);