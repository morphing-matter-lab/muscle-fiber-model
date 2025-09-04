#pragma once
#include <Eigen/Dense>

double distance(const Eigen::MatrixXd &V, const std::vector<int> &indices, const Eigen::MatrixXd &distanceMap);

Eigen::VectorXd distanceGrad(const Eigen::MatrixXd &V, const std::vector<int> &indices, const Eigen::MatrixXd &gradMapX, const Eigen::MatrixXd &gradMapY);