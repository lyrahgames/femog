#ifndef FEMOG_FEM_CONJUGATE_GRADIENT_H_
#define FEMOG_FEM_CONJUGATE_GRADIENT_H_

#include <Eigen/Eigen>
#include <vector>

namespace Femog::Fem::Cg {

void multiply(int n, const float* matrix_values, const int* matrix_row_cdf,
              const int* matrix_col_index, const float* input, float* output);

void conjugate_gradient(const Eigen::SparseMatrix<float> A, float* data,
                        const Eigen::VectorXf& b);

void conjugate_gradient_custom(
    const Eigen::SparseMatrix<float, Eigen::RowMajor> A, float* data,
    const Eigen::VectorXf& b);

}  // namespace Femog::Fem::Cg

#endif  // FEMOG_FEM_CONJUGATE_GRADIENT_H_