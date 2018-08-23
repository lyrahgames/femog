#include "conjugate_gradient.h"
#include <iostream>

namespace Femog::Fem::Cg {

void conjugate_gradient(const Eigen::SparseMatrix<float> A, float* data,
                        const Eigen::VectorXf& b) {
  Eigen::Map<Eigen::VectorXf> x(data, b.size());
  // x = Eigen::VectorXf::Zero(b.size());
  // Eigen::VectorXf r = b - A * x;
  Eigen::VectorXf r = A * x - b;
  Eigen::VectorXf p = -r;
  Eigen::VectorXf y;
  float res = r.squaredNorm();
  // for (auto i = 0; i < b.size(); ++i) {
  while (res > 1e-6f) {
    y = A * p;
    const float alpha = res / p.dot(y);
    x += alpha * p;
    // r -= alpha * y;
    r += alpha * y;
    const float new_res = r.squaredNorm();
    // if (new_res <= 1e-2f) break;
    const float beta = new_res / res;
    res = new_res;
    // p = r + beta * p;
    p = beta * p - r;
  }
  // std::cout << "residuum = " << res << std::endl;
}

}  // namespace Femog::Fem::Cg