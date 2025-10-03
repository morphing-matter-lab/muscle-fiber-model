#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <TinyAD/ScalarFunction.hh>

using LLTSolver = Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>;

void newton(
    Eigen::VectorXd& x,
    TinyAD::ScalarFunction<3, double, Eigen::Index>& func,
    LLTSolver& solver,
    int max_iters = 1000,
    double lim = 1e-6,
    bool verbose = true,
    const std::vector<int>& fixedIdx = {},
    const std::function<void(const Eigen::VectorXd&)>& callBack = [](const auto&) {});

Eigen::SparseMatrix<double> projectionMatrix(const std::vector<int>& fixedIdx, int size);