#ifndef FEMOG_FEM_GPU_WAVE_SOLVER_H_
#define FEMOG_FEM_GPU_WAVE_SOLVER_H_

namespace Femog {
namespace Fem {

class Wave_solver {
 public:
  Wave_solver(int n, float* mass, float* stiffness, int* row, int* col,
              float* wave, float* evolution);
  ~Wave_solver();

  void operator()(float c, float dt);

  void copy_wave(float* output);

  float* mass_values;
  float* stiffness_values;
  int* row_cdf;
  int* col_index;
  float* wave;
  float* evolution;
  int dimension;
  int nnz;
  int threads_per_block;
  int blocks;

  float* tmp_p;
  float* tmp_r;
  float* tmp_y;
};

}  // namespace Fem
}  // namespace Femog

#endif  // FEMOG_FEM_GPU_WAVE_SOLVER_H_