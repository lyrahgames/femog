#ifndef FEMOG_FEM_GPU_SOLVER_H_
#define FEMOG_FEM_GPU_SOLVER_H_

#include <Eigen/Eigen>

namespace Femog {
namespace Fem {

void conjugate_gradient(Eigen::SparseMatrix<float, Eigen::RowMajor>& matrix,
                        Eigen::Map<Eigen::VectorXf>& wave,
                        Eigen::VectorXf& rhs);

void solve_wave_problem(
    Eigen::SparseMatrix<float, Eigen::RowMajor>& mass_matrix,
    Eigen::SparseMatrix<float, Eigen::RowMajor>& stiffness_matrix,
    Eigen::Map<Eigen::VectorXf>& wave, Eigen::Map<Eigen::VectorXf>& evolution,
    float dt, float c);

}  // namespace Fem
}  // namespace Femog

#endif  // FEMOG_FEM_GPU_SOLVER_H_