#include "gpu_solver.h"

#include <cusp/copy.h>
#include <cusp/csr_matrix.h>
#include <cusp/elementwise.h>
#include <cusp/krylov/cg.h>

namespace Femog {
namespace Fem {

void conjugate_gradient(Eigen::SparseMatrix<float, Eigen::RowMajor>& matrix,
                        Eigen::Map<Eigen::VectorXf>& wave,
                        Eigen::VectorXf& rhs) {
  typedef cusp::array1d<int, cusp::host_memory> IndexArray;
  typedef cusp::array1d<float, cusp::host_memory> ValueArray;
  typedef typename IndexArray::view IndexArrayView;
  typedef typename ValueArray::view ValueArrayView;

  IndexArrayView row_offsets(matrix.outerIndexPtr(),
                             matrix.outerIndexPtr() + matrix.outerSize() + 1);
  IndexArrayView col(matrix.innerIndexPtr(),
                     matrix.innerIndexPtr() + matrix.nonZeros());
  ValueArrayView values(matrix.valuePtr(),
                        matrix.valuePtr() + matrix.nonZeros());

  cusp::csr_matrix_view<IndexArrayView, IndexArrayView, ValueArrayView> A(
      matrix.rows(), matrix.cols(), static_cast<int>(matrix.nonZeros()),
      row_offsets, col, values);

  cusp::csr_matrix<int, float, cusp::device_memory> B(A);
  cusp::array1d<float, cusp::device_memory> x(B.num_rows, 0);
  cusp::array1d<float, cusp::device_memory> b(rhs.data(),
                                              rhs.data() + rhs.size());
  cusp::krylov::cg(B, x, b);

  cusp::array1d<float, cusp::host_memory>::view wave_view(
      wave.data(), wave.data() + wave.size());
  // wave_view = x;
  cusp::copy(x, wave_view);
}

struct saxpy_functor : public thrust::binary_function<float, float, float> {
  const float a;

  saxpy_functor(float _a) : a(_a) {}

  __host__ __device__ float operator()(const float& x, const float& y) const {
    return a * x + y;
  }
};

void solve_wave_problem(
    Eigen::SparseMatrix<float, Eigen::RowMajor>& mass_matrix,
    Eigen::SparseMatrix<float, Eigen::RowMajor>& stiffness_matrix,
    Eigen::Map<Eigen::VectorXf>& wave, Eigen::Map<Eigen::VectorXf>& evolution,
    float dt, float c) {
  typedef cusp::array1d<int, cusp::host_memory> IndexArray;
  typedef cusp::array1d<float, cusp::host_memory> ValueArray;
  typedef typename IndexArray::view IndexArrayView;
  typedef typename ValueArray::view ValueArrayView;

  IndexArrayView mass_rows(
      mass_matrix.outerIndexPtr(),
      mass_matrix.outerIndexPtr() + mass_matrix.outerSize() + 1);
  IndexArrayView mass_cols(
      mass_matrix.innerIndexPtr(),
      mass_matrix.innerIndexPtr() + mass_matrix.nonZeros());
  ValueArrayView mass_values(mass_matrix.valuePtr(),
                             mass_matrix.valuePtr() + mass_matrix.nonZeros());
  cusp::csr_matrix_view<IndexArrayView, IndexArrayView, ValueArrayView>
  mass_view(mass_matrix.rows(), mass_matrix.cols(),
            static_cast<int>(mass_matrix.nonZeros()), mass_rows, mass_cols,
            mass_values);
  cusp::csr_matrix<int, float, cusp::device_memory> mass_device(mass_view);

  IndexArrayView stiffness_rows(
      stiffness_matrix.outerIndexPtr(),
      stiffness_matrix.outerIndexPtr() + stiffness_matrix.outerSize() + 1);
  IndexArrayView stiffness_cols(
      stiffness_matrix.innerIndexPtr(),
      stiffness_matrix.innerIndexPtr() + stiffness_matrix.nonZeros());
  ValueArrayView stiffness_values(
      stiffness_matrix.valuePtr(),
      stiffness_matrix.valuePtr() + stiffness_matrix.nonZeros());
  cusp::csr_matrix_view<IndexArrayView, IndexArrayView, ValueArrayView>
  stiffness_view(stiffness_matrix.rows(), stiffness_matrix.cols(),
                 static_cast<int>(stiffness_matrix.nonZeros()), stiffness_rows,
                 stiffness_cols, stiffness_values);
  cusp::csr_matrix<int, float, cusp::device_memory> stiffness_device(
      stiffness_view);

  cusp::array1d<float, cusp::device_memory> x(wave.data(),
                                              wave.data() + wave.size());
  cusp::array1d<float, cusp::device_memory> y(
      evolution.data(), evolution.data() + evolution.size());

  cusp::array1d<float, cusp::device_memory> tmp(wave.size());
  cusp::array1d<float, cusp::device_memory> tmp2(wave.size());

  cusp::multiply(stiffness_device, x, tmp);
  thrust::transform(tmp.begin(), tmp.end(),
                    thrust::make_constant_iterator(dt * c * c), tmp.begin(),
                    thrust::multiplies<float>());
  cusp::multiply(mass_device, y, x);
  thrust::transform(x.begin(), x.end(),
                    thrust::make_constant_iterator(1.0 - dt), x.begin(),
                    thrust::multiplies<float>());
  thrust::transform(x.begin(), x.end(), tmp.begin(), x.begin(),
                    thrust::minus<float>());

  cusp::krylov::cg(mass_device, y, x);

  thrust::transform(x.begin(), x.end(), y.begin(), x.begin(),
                    saxpy_functor(dt));

  cusp::array1d<float, cusp::host_memory>::view wave_view(
      wave.data(), wave.data() + wave.size());
  cusp::copy(x, wave_view);
  cusp::array1d<float, cusp::host_memory>::view evolution_view(
      evolution.data(), evolution.data() + evolution.size());
  cusp::copy(y, evolution_view);
}

}  // namespace Fem
}  // namespace Femog