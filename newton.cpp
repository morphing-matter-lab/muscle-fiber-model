#include "newton.h"

#include <TinyAD/Utils/NewtonDecrement.hh>
#include <igl/colon.h>
#include <igl/slice.h>
#include <igl/slice_into.h>

Eigen::SparseMatrix<double> buildHGN(const Eigen::VectorXd& masses,
                                     const Eigen::SparseMatrix<double>& P,
                                     const Eigen::SparseMatrix<double>& M_theta,
                                     const Eigen::SparseMatrix<double>& H)
{
  using namespace Eigen;

  int n = P.cols();
  int m = P.rows();
  int nTheta = M_theta.rows();

  // Mass matrix x (P * M * P')
  SparseMatrix<double> D(n, n);
  D.reserve(n);
  for(int i = 0; i < n; ++i)
    D.insert(i, i) = masses(i);
  D = (P * D * P.transpose()).eval();

  std::vector<Triplet<double>> v;
  for(int i = 0; i < D.outerSize(); i++)
    for(typename SparseMatrix<double>::InnerIterator it(D, i); it; ++it)
      v.emplace_back(it.row(), it.col(), it.value());

  // Mass matrix theta
  for(int i = 0; i < M_theta.outerSize(); i++)
    for(typename SparseMatrix<double>::InnerIterator it(M_theta, i); it; ++it)
      v.emplace_back(it.row() + m, it.col() + m, it.value());

  // P * (df / dx)' * P'
  SparseMatrix<double> A = (P * H.block(0, 0, n, n) * P.transpose()).eval();
  for(int i = 0; i < A.outerSize(); i++)
    for(typename SparseMatrix<double>::InnerIterator it(A, i); it; ++it)
    {
      v.emplace_back(it.row(), it.col() + m + nTheta, it.value());
      v.emplace_back(it.col() + m + nTheta, it.row(), it.value());
    }

  // (df / dθ)' * P'
  A = (H.block(n, 0, nTheta, n) * P.transpose()).eval();
  for(int i = 0; i < A.outerSize(); i++)
    for(typename SparseMatrix<double>::InnerIterator it(A, i); it; ++it)
    {
      v.emplace_back(it.row() + m, it.col() + m + nTheta, it.value());
      v.emplace_back(it.col() + m + nTheta, it.row() + m, it.value());
    }

  SparseMatrix<double> HGN(2 * m + nTheta, 2 * m + nTheta);
  HGN.setFromTriplets(v.begin(), v.end());

  return HGN;
}

void updateHGN(Eigen::SparseMatrix<double>& HGN,
               const Eigen::SparseMatrix<double>& P,
               const Eigen::SparseMatrix<double>& H)
{
  using namespace Eigen;

  int n = P.cols();
  int m = P.rows();
  int nTheta = H.rows() - n;

  // P * (df / dx)' * P'
  Eigen::SparseMatrix<double> A = (P * H.block(0, 0, n, n) * P.transpose()).eval();
  igl::slice_into(A, igl::colon<int>(0, m - 1), igl::colon<int>(m + nTheta, 2 * m + nTheta - 1), HGN);

  // (df / dθ)' * P'
  A = (H.block(n, 0, nTheta, n) * P.transpose()).eval();
  igl::slice_into(A, igl::colon<int>(m, m + nTheta - 1), igl::colon<int>(m + nTheta, 2 * m + nTheta - 1), HGN);

  HGN = SparseMatrix<double>(HGN.selfadjointView<Upper>());
}

std::vector<int> findCenterFaceIndices(const Eigen::MatrixXd& P, const Eigen::MatrixXi& F)
{
  int centerIdx = 0;
  double dist = (P.row(F(0, 0)) + P.row(F(0, 1)) + P.row(F(0, 2))).norm();
  for(int i = 0; i < F.rows(); ++i)
  {
    if((P.row(F(i, 0)) + P.row(F(i, 1)) + P.row(F(i, 2))).norm() < dist)
    {
      dist = (P.row(F(i, 0)) + P.row(F(i, 1)) + P.row(F(i, 2))).norm();
      centerIdx = i;
    }
  }

  // Fixed Indices
  std::vector<int> fixedIdx = {3 * F(centerIdx, 0), 3 * F(centerIdx, 0) + 1, 3 * F(centerIdx, 0) + 2,
                               3 * F(centerIdx, 1), 3 * F(centerIdx, 1) + 1, 3 * F(centerIdx, 1) + 2,
                               3 * F(centerIdx, 2), 3 * F(centerIdx, 2) + 1, 3 * F(centerIdx, 2) + 2};
  std::sort(fixedIdx.begin(), fixedIdx.end());

  return fixedIdx;
}

double lineSearch(const Eigen::VectorXd& x0,
                  const Eigen::VectorXd& d,
                  const double f,
                  const Eigen::VectorXd& g,
                  TinyAD::ScalarFunction<3, double, Eigen::Index>& eval,
                  const double shrink = 0.8,
                  const int max_iters = 64)
{
  Eigen::VectorXd x_new = x0;
  double s = 1.0;
  for(int i = 0; i < max_iters; ++i)
  {
    x_new = x0 + s * d;
    const double f_new = eval(x_new);
    if(f_new <= f + 1e-4 * s * d.dot(g)) // Armijo condition
      return s;

    s *= shrink;
  }
  // line search couldn't find improvement
  return -1;
}

Eigen::SparseMatrix<double> projectionMatrix(const std::vector<int>& fixedIdx, int size)
{
  using namespace Eigen;

  VectorXi indices(size - fixedIdx.size());
  int k = 0;
  for(int i = 0; i < size; ++i)
  {
    if(!std::binary_search(fixedIdx.begin(), fixedIdx.end(), i))
    {
      indices(k) = i;
      ++k;
    }
  }

  SparseMatrix<double> P, Id(size, size);
  Id.setIdentity();

  igl::slice(Id, indices, 1, P);

  return P;
}

void newton(Eigen::VectorXd& x,
            TinyAD::ScalarFunction<3, double, Eigen::Index>& func,
            LLTSolver& solver,
            int max_iters,
            double lim,
            bool verbose,
            const std::vector<int>& fixedIdx,
            const std::function<void(const Eigen::VectorXd&)>& callback)
{
  if(verbose)
    std::cout << "Initial energy: " << func.eval(x) << std::endl;

  std::vector<int> orderedIdx = fixedIdx;
  std::sort(orderedIdx.begin(), orderedIdx.end());
  Eigen::SparseMatrix<double> P = projectionMatrix(orderedIdx, x.size());

  for(int i = 0; i < max_iters; ++i)
  {
    auto [f, g, H] = func.eval_with_derivatives(x);

    for(int j = 0; j < H.cols(); ++j)
      H.coeffRef(j, j) += 1e-10;

    // restrict H and g to free variables
    H = (P * H * P.transpose()).eval();
    g = P * g;

    // Newton direction
    if(i == 0)
      solver.compute(H);
    else
      solver.factorize(H);

    bool exact = true;
    if(solver.info() != Eigen::Success)
    {
      exact = false;
      auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
      H_proj = (P * H_proj * P.transpose()).eval();

      H = 0.9 * H + 0.1 * H_proj;
      solver.factorize(H);
      if(solver.info() != Eigen::Success)
        solver.factorize(H_proj);
    }

    Eigen::VectorXd d = -solver.solve(g);

    if(verbose)
    {
      std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(d, g);
      if(exact)
        std::cout << "\tFactorization = Exact\n";
      else
        std::cout << "\tFactorization = Project\n";
    }

    d = P.transpose() * d;
    g = P.transpose() * g;

    double s = lineSearch(x, d, f, g, func);
    if(s < 0)
      break;
    x += s * d;
  
    if(TinyAD::newton_decrement(d, g) < lim && exact)
      break;

    callback(x);
  }
  if(verbose)
    std::cout << "Final energy: " << func.eval(x) << "\n";
}


// Eigen::MatrixXd
// sparse_gauss_newton(const Eigen::MatrixXd &V, 
//                     const Eigen::MatrixXd &P, 
//                     const std::vector<int> &boundaryIndices, 
//                     const Eigen::MatrixXd &distanceMap,
//                     const Eigen::MatrixXd &distanceResidualX,
//                     const Eigen::MatrixXd &distanceResidualY,
//                     const TinyAD::ScalarFunction<1, double, Eigen::Index>& adjointFunc,
//                     const std::vector<int>& fixedIdx,
//                     int max_iters,
//                     double lim,
//                     const std::function<void(const Eigen::VectorXd&)>& callback)
// {

//   LLTSolver adjointSolver;

//   auto distance = [&](double stretch_factor) {
//     fsim::NeoHookeanMembrane model(P / stretch_factor, F, 1, 1, 0.49, 0);

//     // declare NewtonSolver object
//     optim::NewtonSolver<double> solver;
//     // specify fixed degrees of freedom (here the 4 corners of the mesh are fixed)
//     solver.options.threshold = 1e-6; // specify how small the gradient's norm has to be
//     solver.options.fixed_dofs = fixed_idx;
//     solver.options.display = optim::SolverDisplay::quiet;

//     solver.solve(model, V.reshaped<RowMajor>());

//     V = Map<fsim::Mat3<double>>(solver.var().data(), V.rows(), 3);

//     return distance(V, boundaryIndices, distanceMap);
//   };

//   // Build matrix P
//   Eigen::SparseMatrix<double> P = projectionMatrix(fixedIdx, x.size());

//   // Hessian matrix H
//   Eigen::VectorXd X(targetV.size() + theta.size());
//   X.head(targetV.size()) = x;
//   X.tail(theta.size()) = theta;
//   Eigen::SparseMatrix<double> H = adjointFunc.eval_hessian(X);

//   // Build HGN matrix
//   Eigen::SparseMatrix<double> HGN = buildHGN(12 * Eigen::VectorXd::Ones(nBoundary), P, Eigen::SparseMatrix<double>{}, H);

//   auto distanceGrad = [&](const Eigen::VectorXd& th) -> Eigen::VectorXd {
//     Eigen::VectorXd X(targetV.size() + th.size());
//     X.head(targetV.size()) = x;
//     X.tail(th.size()) = th;
//     H = adjointFunc.eval_hessian(X);

//     for(int j = 0; j < targetV.size(); ++j)
//       H.coeffRef(j, j) += 1e-10;

//     Eigen::SparseMatrix<double> A = (P * H.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();

//     adjointSolver.factorize(A);
//     if(adjointSolver.info() != Eigen::Success)
//     {
//       auto [f, g, A_proj] = adjointFunc.eval_with_hessian_proj(X);
//       A_proj = (P * A_proj.block(0, 0, targetV.size(), targetV.size()) * P.transpose()).eval();

//       A = 0.9 * A + 0.1 * A_proj;
//       adjointSolver.factorize(A);
//       if(adjointSolver.info() != Eigen::Success)
//         adjointSolver.factorize(A_proj);
//     }

//     Eigen::VectorXd b = P * masses.cwiseProduct(x - xTarget);
//     Eigen::VectorXd dir = adjointSolver.solve(b);
//     if(adjointSolver.info() != Eigen::Success)
//       std::cout << "Solver error\n";

//     dir = P.transpose() * dir;

//     return -2 * H.block(targetV.size(), 0, th.size(), targetV.size()) * dir + 2 * wM * M_theta * th + 2 * wL * L * th;
//   };

//   double energy = distance(theta);
//   std::cout << "Initial energy: " << energy << std::endl;

//   LUSolver solver;

//   for(int i = 0; i < max_iters; ++i)
//   {
//     double f = distance(theta);
//     Eigen::VectorXd g = distanceGrad(theta);

//     Eigen::VectorXd b(2 * x.size() - 2 * fixedIdx.size() + theta.size());
//     b.setZero();
//     b.segment(x.size() - fixedIdx.size(), theta.size()) = -g;

//     // Update HGN
//     updateHGN(HGN, P, H);

//     if(i == 0)
//       solver.compute(HGN);
//     else
//       solver.factorize(HGN);

//     if(solver.info() != Eigen::Success)
//     {
//       std::cout << "Solver error\n";
//       return targetV;
//     }

//     Eigen::VectorXd d = solver.solve(b);
//     Eigen::VectorXd deltaTheta = d.segment(x.size() - fixedIdx.size(), theta.size());
//     Eigen::VectorXd deltaX = d.segment(0, x.size() - fixedIdx.size());
//     deltaX = P.transpose() * deltaX;

//     // LINE SEARCH
//     Eigen::VectorXd x_old = x;
//     double s = lineSearch(theta, deltaTheta, f, g, distance, [&](double s) { x = x_old + s * deltaX; });
//     if(s < 0)
//     {
//       std::cout << "Line search failed\n";
//       break;
//     }
//     theta += s * deltaTheta;

//     std::cout << "Decrement in iteration " << i << ": " << TinyAD::newton_decrement(deltaTheta, g)
//               << "\tDistance: " << (x - xTarget).dot(masses.cwiseProduct(x - xTarget)) << "\tStep size: " << s
//               << std::endl;
//     if(TinyAD::newton_decrement(deltaTheta, g) < lim || solver.info() != Eigen::Success)
//       break;

//     callback(x);
//   }

//   std::cout << "Final energy: " << distance(theta) << "\n";

//   Eigen::MatrixXd V(targetV.rows(), 3);
//   for(int i = 0; i < targetV.rows(); ++i)
//     for(int j = 0; j < 3; ++j)
//       V(i, j) = x(3 * i + j);

//   theta2.fromVector(theta);
//   return V;
// }