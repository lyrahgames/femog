#include "conjugate_gradient.h"
#include <iostream>

namespace Femog::Fem::Cg {

void multiply(int n, const float* matrix_values, const int* matrix_row_cdf,
              const int* matrix_col_index, const float* input, float* output) {
  for (int i = 0; i < n; ++i) {
    output[i] = 0;
    for (int j = matrix_row_cdf[i]; j < matrix_row_cdf[i + 1]; ++j) {
      output[i] += matrix_values[j] * input[matrix_col_index[j]];
    }
  }
}

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

void conjugate_gradient_custom(
    const Eigen::SparseMatrix<float, Eigen::RowMajor> A, float* data,
    const Eigen::VectorXf& b) {
  Eigen::Map<Eigen::VectorXf> x(data, b.size());
  // x = Eigen::VectorXf::Zero(b.size());
  // Eigen::VectorXf r = b - A * x;
  // Eigen::VectorXf r = A * x - b;

  Eigen::VectorXf r(b.size());
  multiply(b.size(), A.valuePtr(), A.outerIndexPtr(), A.innerIndexPtr(), data,
           r.data());
  r -= b;

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