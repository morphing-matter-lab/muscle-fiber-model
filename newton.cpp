#include "newton.h"

#include <TinyAD/Utils/NewtonDecrement.hh>
#include <igl/slice.h>

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
