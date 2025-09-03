#include "distance.h"
#include <iostream>

double distance(const Eigen::MatrixXd &V, const std::vector<int> &indices, const Eigen::MatrixXd &distanceMap)
{
  const double x_max = 4.225;
  const double y_max = 1.95;
  const double mm_to_px = 1287.2;

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

Eigen::VectorXd distanceGrad(const Eigen::MatrixXd &V, const std::vector<int> &indices, const Eigen::MatrixXd &distanceMap)
{
  const double x_max = 4.225;
  const double y_max = 1.95;
  const double mm_to_px = 1287.2;

  Eigen::VectorXd res = Eigen::VectorXd::Zero(V.rows() * 2);

  for(int idx: indices)
  {
    // transform coordinates to upper-right corner and flip y axis
    double v_x = std::abs(V(idx, 0));  
    double v_y = y_max - std::abs(V(idx, 1));

    int px_j = static_cast<int>(std::floor(v_x * mm_to_px));
    int px_i = static_cast<int>(std::floor(v_y * mm_to_px));

    int iMinusOne = px_i > 0 ? px_i - 1 : 0;
    int jPlusTwo = px_j < distanceMap.cols() - 2 ? px_j + 2 : px_j + 1;

    // compute barycentric coordinates
    double alpha = px_i + px_j + 1 - (v_x + v_y) * mm_to_px;
    double beta, gamma;
    if(alpha > 0)
    { // upper triangle
      beta = v_y * mm_to_px - px_i;
      gamma = v_x * mm_to_px - px_j;
      res(2 * idx + 1) += (distanceMap(px_i, px_j) - distanceMap(iMinusOne, px_j)) * alpha;
      res(2 * idx) += (distanceMap(px_i, px_j + 1) - distanceMap(px_i, px_j)) * alpha;
    }
    else
    { // lower triangle
      alpha *= -1;
      beta = px_j + 1 - v_x * mm_to_px;
      gamma = px_i + 1 - v_y * mm_to_px;
      res(2 * idx + 1) += (distanceMap(px_i + 1, px_j + 1) - distanceMap(px_i, px_j + 1)) * alpha;
      res(2 * idx) += (distanceMap(px_i + 1, jPlusTwo) - distanceMap(px_i + 1, px_j + 1)) * alpha;
    }
    res(2 * idx + 1) += (distanceMap(px_i + 1, px_j) - distanceMap(px_i, px_j)) * beta;
    res(2 * idx + 1) += (distanceMap(px_i, px_j + 1) - distanceMap(iMinusOne, px_j + 1)) * gamma;
      
    res(2 * idx) += (distanceMap(px_i + 1, px_j + 1) - distanceMap(px_i + 1, px_j)) * beta;
    res(2 * idx) += (distanceMap(px_i, jPlusTwo) - distanceMap(px_i, px_j + 1)) * gamma;

    // if(px_i == distanceMap.rows() - 1)
    // {
    //   std::cout << px_i << " " << px_j << "\n";
    //   std::cout << iMinusOne << " " << jPlusTwo << "\n";
    //   std::cout << alpha << " " << beta << " " << gamma << "\n";
    // }

    if(V(idx, 0) < 0) 
      res(2 * idx) *= -1;
    if(V(idx, 1) > 0) 
      res(2 * idx + 1) *= -1;
  }

  return res;
}