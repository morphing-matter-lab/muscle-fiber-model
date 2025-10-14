#pragma once

#include <Eigen/Dense>
#include <nanobind/eigen/dense.h>

bool is_point_in_triangle(const Eigen::VectorXd &p, const Eigen::Vector3i &face, const Eigen::MatrixXd &V);

Eigen::MatrixXd transfer_data_to_3D_mesh(const nanobind::DRef<Eigen::MatrixXd> &V,
                                         const nanobind::DRef<Eigen::MatrixXi> &F,
                                         const nanobind::DRef<Eigen::MatrixXd> &Phi,
                                         const nanobind::DRef<Eigen::MatrixXd> &V_3D,
                                         const nanobind::DRef<Eigen::MatrixXi> &F_3D);

Eigen::MatrixXd histogram_data_to_mesh(const nanobind::DRef<Eigen::MatrixXd> &V,
                                       const nanobind::DRef<Eigen::MatrixXi> &F,
                                       const nanobind::DRef<Eigen::MatrixXd> &image,
                                       const nanobind::DRef<Eigen::MatrixXd> &alpha,
                                       double world_coords_to_px,
                                       double radius,
                                       int n);

Eigen::VectorXd image_data_to_mesh(const nanobind::DRef<Eigen::MatrixXd> &V,
                                   const nanobind::DRef<Eigen::MatrixXi> &F,
                                   const nanobind::DRef<Eigen::MatrixXd> &image,
                                   double world_coords_to_px);

Eigen::MatrixXd orientation_data_to_mesh(const nanobind::DRef<Eigen::MatrixXd> &V,
                                         const nanobind::DRef<Eigen::MatrixXi> &F,
                                         const nanobind::DRef<Eigen::MatrixXd> &angles,
                                         double world_coords_to_px);