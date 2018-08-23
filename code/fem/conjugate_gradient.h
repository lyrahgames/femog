#ifndef FEMOG_FEM_CONJUGATE_GRADIENT_H_
#define FEMOG_FEM_CONJUGATE_GRADIENT_H_

#include <Eigen/Eigen>
#include <vector>

namespace Femog::Fem::Cg {

void conjugate_gradient(const Eigen::SparseMatrix<float> A, float* data,
                        const Eigen::VectorXf& b);
}

#endif  // FEMOG_FEM_CONJUGATE_GRADIENT_H_