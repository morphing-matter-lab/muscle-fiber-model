
#include "data_to_mesh.h"
#include <complex>
#include <iostream>
#include <Eigen/SparseCore>
#include <numbers>

namespace nb = nanobind;
using namespace std::numbers;

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
                                       const nb::DRef<Eigen::MatrixXd> &alpha,
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

#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads() - 1)
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
          Phi(i, k) += std::exp(-std::pow(dist / radius, 2) / 2 / std::pow(sigma, 2)) * normal * alpha(y, x);
        }
      }
    }
  }

  return Phi;
}

Eigen::VectorXd image_data_to_mesh(const nb::DRef<Eigen::MatrixXd> &V,
                                   const nb::DRef<Eigen::MatrixXi> &F,
                                   const nb::DRef<Eigen::MatrixXd> &image,
                                   double world_coords_to_px)
{
  using namespace Eigen;

  VectorXd res = VectorXd::Zero(F.rows());

  int nX = image.cols();
  int nY = image.rows();

#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads() - 1)
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
            res(i) += image(y, x);
            k += 1;
          }
        }
      }
    }
    if (k > 0)
      res(i) /= k;
  }

  return res;
}

Eigen::MatrixXd orientation_data_to_mesh(const nb::DRef<Eigen::MatrixXd> &V,
                                         const nb::DRef<Eigen::MatrixXi> &F,
                                         const nb::DRef<Eigen::MatrixXd> &angles,
                                         double world_coords_to_px)
{
  using namespace Eigen;

  VectorXcd dirVec = VectorXcd::Zero(F.rows());

  int nX = angles.cols();
  int nY = angles.rows();

#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads() - 1)
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

    // int k = 0;
    for (int x = minIdxX; x < maxIdxX; ++x)
    {
      for (int y = minIdxY; y < maxIdxY; ++y)
      {
        Vector2d coords;
        coords << x - nX / 2, -y + nY / 2;
        coords /= world_coords_to_px;

        if (is_point_in_triangle(coords, F.row(i), V))
        {
          if (angles(y, x) != 0)
          {
            dirVec(i) += std::polar(1., 2 * angles(y, x));
            // k += 1;
          }
        }
      }
    }
    dirVec(i) /= (V(F(i, 1), 0) - V(F(i, 0), 0)) * (V(F(i, 2), 1) - V(F(i, 0), 1)) -
                 (V(F(i, 1), 1) - V(F(i, 0), 1)) * (V(F(i, 2), 0) - V(F(i, 0), 0));
    // if (k > 0)
    //   dirVec(i) /= k;
  }

  MatrixXd res(F.rows(), 2);
  for (int i = 0; i < F.rows(); ++i)
  {
    res.row(i) << std::cos(std::arg(dirVec(i)) / 2), std::sin(std::arg(dirVec(i)) / 2);
    res.row(i) *= std::norm(dirVec(i));
  }

  return res;
}
