#include "distance.h"
#include <iostream>

double distance(const Eigen::MatrixXd &V, const std::vector<int> &indices, const Eigen::MatrixXd &distanceMap)
{
  const double mm_to_px = 1287.2;
  const double y_max = (distanceMap.rows() - 1) / mm_to_px;

  double res = 0;

  for(int idx: indices)
  {
    // transform coordinates to upper-right corner and flip y axis
    double v_x = std::abs(V(idx, 0));  
    double v_y = y_max - std::abs(V(idx, 1));

    int px_j = static_cast<int>(std::floor(v_x * mm_to_px));
    int px_i = static_cast<int>(std::floor(v_y * mm_to_px));

    // compute barycentric coordinates
    double alpha = px_i + px_j + 1 - (v_x + v_y) * mm_to_px;
    if(alpha > 0)
    { // upper triangle
      double beta = v_y * mm_to_px - px_i;
      double gamma = v_x * mm_to_px - px_j;
      res += distanceMap(px_i, px_j) * alpha + distanceMap(px_i + 1, px_j) * beta + distanceMap(px_i, px_j + 1) * gamma;
    }
    else
    { // lower triangle
      alpha *= -1;
      double beta = px_j + 1 - v_x * mm_to_px;
      double gamma = px_i + 1 - v_y * mm_to_px;
      res += distanceMap(px_i + 1, px_j + 1) * alpha + distanceMap(px_i + 1, px_j) * beta + distanceMap(px_i, px_j + 1) * gamma;
    }
  }

  return res;
}

Eigen::VectorXd distanceGrad(const Eigen::MatrixXd &V, const std::vector<int> &indices, const Eigen::MatrixXd &gradMapX, const Eigen::MatrixXd &gradMapY)
{
  const double mm_to_px = 1287.2;
  const double y_max = (gradMapX.rows() - 1) / mm_to_px;

  Eigen::VectorXd res = Eigen::VectorXd::Zero(V.rows() * 2);

  for(int idx: indices)
  {
    // transform coordinates to upper-right corner and flip y axis
    double v_x = std::abs(V(idx, 0));  
    double v_y = y_max - std::abs(V(idx, 1));

    int px_j = static_cast<int>(std::floor(v_x * mm_to_px));
    int px_i = static_cast<int>(std::floor(v_y * mm_to_px));

    // compute barycentric coordinates
    double alpha = px_i + px_j + 1 - (v_x + v_y) * mm_to_px;
    if(alpha > 0)
    { // upper triangle
      double beta = v_y * mm_to_px - px_i;
      double gamma = v_x * mm_to_px - px_j;
      res(2 * idx) = gradMapX(px_i, px_j) * alpha + gradMapX(px_i + 1, px_j) * beta + gradMapX(px_i, px_j + 1) * gamma;
      res(2 * idx + 1) = gradMapY(px_i, px_j) * alpha + gradMapY(px_i + 1, px_j) * beta + gradMapY(px_i, px_j + 1) * gamma;
    }
    else
    { // lower triangle
      alpha *= -1;
      double beta = px_j + 1 - v_x * mm_to_px;
      double gamma = px_i + 1 - v_y * mm_to_px;
      res(2 * idx) = gradMapX(px_i + 1, px_j + 1) * alpha + gradMapX(px_i + 1, px_j) * beta + gradMapX(px_i, px_j + 1) * gamma;
      res(2 * idx + 1) = gradMapY(px_i + 1, px_j + 1) * alpha + gradMapY(px_i + 1, px_j) * beta + gradMapY(px_i, px_j + 1) * gamma;
    }
    if(V(idx, 0) < 0) 
      res(2 * idx) *= -1;
    if(V(idx, 1) > 0) 
      res(2 * idx + 1) *= -1;
  }

  return res;
}