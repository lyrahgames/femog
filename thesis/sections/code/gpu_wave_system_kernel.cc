namespace Fem {
namespace {

struct saxpy_functor : public thrust::binary_function<float, float, float> {
  const float a;

  saxpy_functor(float _a) : a(_a) {}

  __host__ __device__ float operator()(const float& x, const float& y) const {
    return a * x + y;
  }
};

struct saxpby_functor : public thrust::binary_function<float, float, float> {
  const float a;
  const float b;

  saxpby_functor(float _a, float _b) : a(_a), b(_b) {}

  __host__ __device__ float operator()(const float& x, const float& y) const {
    return a * x + b * y;
  }
};

__global__ void spmv_csr_kernel(int n, float alpha, const float* values,
                                const int* row_cdf, const int* col_index,
                                const float* input, float beta, float* output) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < n) {
    float dot = 0;

    const int start = row_cdf[row];
    const int end = row_cdf[row + 1];

    for (int j = start; j < end; ++j) dot += values[j] * input[col_index[j]];

    output[row] = beta * output[row] + alpha * dot;
  }
}

}  // namespace
}  // namespace Fem