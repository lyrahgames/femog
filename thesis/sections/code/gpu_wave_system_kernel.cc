namespace Fem {
namespace {

template <typename T>
struct axpy_functor : public thrust::binary_function<T, T, T> {
  using value_type = T;

  const value_type a;

  axpy_functor(value_type _a) : a(_a) {}

  __host__ __device__ value_type operator()(const value_type& x,
                                            const value_type& y) const {
    return a * x + y;
  }
};

template <typename T>
struct axpby_functor : public thrust::binary_function<T, T, T> {
  using value_type = T;

  const value_type a;
  const value_type b;

  axpby_functor(value_type _a, value_type _b) : a(_a), b(_b) {}

  __host__ __device__ value_type operator()(const value_type& x,
                                            const value_type& y) const {
    return a * x + b * y;
  }
};

template <typename value_type>
__global__ void rhs_kernel(
    int n, const value_type* mass_values, const value_type* stiffness_values,
    const int* row_cdf, const int* col_index,
    const value_type* boundary_mass_values,
    const value_type* boundary_stiffness_values, const int* boundary_row_cdf,
    const int* boundary_col_index, const value_type* wave_values,
    const value_type* evolution_values, value_type dt, value_type* output) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < n) {
    value_type mass_dot = 0;
    value_type stiffness_dot = 0;

    const int start = row_cdf[row];
    const int end = row_cdf[row + 1];
    for (int j = start; j < end; ++j) {
      mass_dot += mass_values[j] * evolution_values[col_index[j]];
      stiffness_dot += stiffness_values[j] * wave_values[col_index[j]];
    }

    const int boundary_start = boundary_row_cdf[row];
    const int boundary_end = boundary_row_cdf[row + 1];
    for (int j = boundary_start; j < boundary_end; ++j) {
      mass_dot +=
          boundary_mass_values[j] * evolution_values[n + boundary_col_index[j]];
      stiffness_dot +=
          boundary_stiffness_values[j] * wave_values[n + boundary_col_index[j]];
    }

    output[row] = mass_dot - dt * stiffness_dot;
  }
}

template <typename value_type>
__global__ void spmv_csr_kernel(int n, value_type alpha,
                                const value_type* values, const int* row_cdf,
                                const int* col_index, const value_type* input,
                                value_type beta, value_type* output) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < n) {
    value_type dot = 0;

    const int start = row_cdf[row];
    const int end = row_cdf[row + 1];

    for (int j = start; j < end; ++j) dot += values[j] * input[col_index[j]];

    output[row] = beta * output[row] + alpha * dot;
  }
}

}  // namespace
}  // namespace Fem