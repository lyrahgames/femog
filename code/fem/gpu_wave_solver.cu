#include "gpu_wave_solver.h"

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

namespace Femog {
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

__global__ void rhs_kernel(int n, const float* mass_values,
                           const float* stiffness_values, const int* row_cdf,
                           const int* col_index, const float* wave_values,
                           const float* evolution_values, float dt,
                           float* output) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < n) {
    float mass_dot = 0;
    float stiffness_dot = 0;

    const int start = row_cdf[row];
    const int end = row_cdf[row + 1];

    for (int j = start; j < end; ++j) {
      mass_dot += mass_values[j] * evolution_values[col_index[j]];
      stiffness_dot += stiffness_values[j] * wave_values[col_index[j]];
    }

    output[row] = mass_dot - dt * stiffness_dot;
  }
}

__global__ void spmv_csr_vector_kernel(int n, float alpha, const float* values,
                                       const int* row_cdf, const int* col_index,
                                       const float* input, float beta,
                                       float* output) {
  __shared__ float vals[1024];
  const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  const int warp_id = thread_id / 32;
  const int lane = thread_id & (32 - 1);
  const int row = warp_id;

  if (row < n) {
    const int start = row_cdf[row];
    const int end = row_cdf[row + 1];

    vals[threadIdx.x] = 0;
    for (int j = start + lane; j < end; j += 32)
      vals[threadIdx.x] += values[j] * input[col_index[j]];

    __syncthreads();
    if (lane < 16) vals[threadIdx.x] += vals[threadIdx.x + 16];
    __syncthreads();
    if (lane < 8) vals[threadIdx.x] += vals[threadIdx.x + 8];
    __syncthreads();
    if (lane < 4) vals[threadIdx.x] += vals[threadIdx.x + 4];
    __syncthreads();
    if (lane < 2) vals[threadIdx.x] += vals[threadIdx.x + 2];
    __syncthreads();
    if (lane < 1) vals[threadIdx.x] += vals[threadIdx.x + 1];

    __syncthreads();
    if (lane == 0) output[row] = beta * output[row] + alpha * vals[threadIdx.x];
  }
}

}  // namespace

Wave_solver::Wave_solver(int n, float* mass, float* stiffness, int* row,
                         int* col, float* wave_data, float* evolution_data)
    : dimension(n), nnz(row[n]) {
  // std::cout << "dimension = " << dimension << "\nnnz = " << nnz << std::endl;
  // thrust::copy(mass, mass + nnz, std::ostream_iterator<float>(std::cout,
  // ","));

  cudaMalloc((void**)&mass_values, nnz * sizeof(float));
  cudaMalloc((void**)&stiffness_values, nnz * sizeof(float));
  cudaMalloc((void**)&wave, dimension * sizeof(float));
  cudaMalloc((void**)&evolution, dimension * sizeof(float));
  cudaMalloc((void**)&row_cdf, (dimension + 1) * sizeof(int));
  cudaMalloc((void**)&col_index, nnz * sizeof(int));

  cudaMemcpy(mass_values, mass, nnz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(stiffness_values, stiffness, nnz * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(wave, wave_data, dimension * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(evolution, evolution_data, dimension * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(row_cdf, row, (dimension + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(col_index, col, nnz * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&tmp_p, dimension * sizeof(float));
  cudaMalloc((void**)&tmp_r, dimension * sizeof(float));
  cudaMalloc((void**)&tmp_y, dimension * sizeof(float));

  cudaDeviceProp property;
  // int count;
  // cudaGetDeviceCount(&count);
  // for (int i = 0; i < count; ++i){
  cudaGetDeviceProperties(&property, 0);
  // }

  threads_per_block = property.maxThreadsPerBlock;
  blocks = (dimension + threads_per_block - 1) / threads_per_block;

  std::cout << "CUDA Device Name = " << property.name << std::endl
            << "total global memory = " << property.totalGlobalMem << std::endl
            << "shared memory per block = " << property.sharedMemPerBlock
            << std::endl
            << "total const memory = " << property.totalConstMem << std::endl
            << "warp size = " << property.warpSize << std::endl
            << "maximum threads per block = " << property.maxThreadsPerBlock
            << std::endl;

  std::cout << "maximum threads dimension = (";
  std::copy(property.maxThreadsDim, property.maxThreadsDim + 3,
            std::ostream_iterator<int>(std::cout, ","));
  std::cout << ")" << std::endl;

  std::cout << "maximum block dimension = (";
  std::copy(property.maxGridSize, property.maxGridSize + 3,
            std::ostream_iterator<int>(std::cout, ","));
  std::cout << ")" << std::endl << std::endl;

  std::cout << "used threads_per_block = " << threads_per_block << std::endl
            << "used blocks = " << blocks << std::endl;
}

Wave_solver::~Wave_solver() {
  cudaFree(mass_values);
  cudaFree(stiffness_values);
  cudaFree(wave);
  cudaFree(evolution);
  cudaFree(row_cdf);
  cudaFree(col_index);

  cudaFree(tmp_p);
  cudaFree(tmp_r);
  cudaFree(tmp_y);
}

void Wave_solver::operator()(float c, float dt) {
  thrust::device_ptr<float> dev_evolution =
      thrust::device_pointer_cast(evolution);
  thrust::device_ptr<float> dev_wave(wave);
  thrust::device_ptr<float> dev_tmp_y(tmp_y);
  thrust::device_ptr<float> dev_tmp_p(tmp_p);
  thrust::device_ptr<float> dev_tmp_r(tmp_r);

  // Eigen::VectorXf rhs =
  //     (1.0f - gamma * dt) * mass_matrix * y - dt * c * c * stiffness_matrix *
  //     x;
  // spmv_csr_kernel<<<blocks, threads_per_block>>>(
  //     dimension, 1.0f, mass_values, row_cdf, col_index, evolution, 0.0f,
  //     tmp_y);
  // spmv_csr_kernel<<<blocks, threads_per_block>>>(dimension, -dt * c * c,
  //                                                stiffness_values, row_cdf,
  //                                                col_index, wave, 1.0f,
  //                                                tmp_y);
  rhs_kernel<<<blocks, threads_per_block>>>(
      dimension, mass_values, stiffness_values, row_cdf, col_index, wave,
      evolution, c * c * dt, tmp_r);

  // Eigen::VectorXf r = A * x - b;
  spmv_csr_kernel<<<blocks, threads_per_block>>>(dimension, 1.0f, mass_values,
                                                 row_cdf, col_index, evolution,
                                                 -1.0f, tmp_r);
  // thrust::copy(dev_tmp_y, dev_tmp_y + dimension, dev_tmp_r);

  // Eigen::VectorXf p = -r;
  thrust::transform(dev_tmp_r, dev_tmp_r + dimension, dev_tmp_p,
                    thrust::negate<float>());

  // float res = r.squaredNorm();
  float res =
      thrust::inner_product(dev_tmp_r, dev_tmp_r + dimension, dev_tmp_r, 0.0f);
  // std::cout << "res = " << res << std::endl;
  int it = 0;

  // for (auto i = 0; i < dimension; ++i) {
  while ((it < 1 || res > 1e-6f) && it < dimension) {
    // y = A * p;
    spmv_csr_kernel<<<blocks, threads_per_block>>>(
        dimension, 1.0f, mass_values, row_cdf, col_index, tmp_p, 0.0f, tmp_y);

    // thrust::copy(dev_tmp_y, dev_tmp_y + dimension,
    //              std::ostream_iterator<float>(std::cout, " "));

    // const float alpha = res / p.dot(y);
    const float tmp = thrust::inner_product(dev_tmp_p, dev_tmp_p + dimension,
                                            dev_tmp_y, 0.0f);
    // std::cout << "tmp = " << tmp << std::endl;
    const float alpha = res / tmp;
    // std::cout << "alpha = " << alpha << std::endl;

    // x += alpha * p;
    thrust::transform(dev_tmp_p, dev_tmp_p + dimension, dev_evolution,
                      dev_evolution, saxpy_functor(alpha));
    // r += alpha * y;
    thrust::transform(dev_tmp_y, dev_tmp_y + dimension, dev_tmp_r, dev_tmp_r,
                      saxpy_functor(alpha));

    // const float new_res = r.squaredNorm();
    const float new_res = thrust::inner_product(
        dev_tmp_r, dev_tmp_r + dimension, dev_tmp_r, 0.0f);

    // std::cout << "res = " << res << "\tnew_res = " << new_res << std::endl;

    const float beta = new_res / res;
    res = new_res;

    // p = beta * p - r;
    thrust::transform(dev_tmp_p, dev_tmp_p + dimension, dev_tmp_r, dev_tmp_p,
                      saxpby_functor(beta, -1.0f));
    ++it;
  }
  // std::cout << "res = " << res << "\tit = " << it << std::endl;

  // x = x + dt * y;
  thrust::transform(dev_evolution, dev_evolution + dimension, dev_wave,
                    dev_wave, saxpy_functor(dt));
}

void Wave_solver::copy_wave(float* output) {
  thrust::device_ptr<float> dev_wave(wave);
  thrust::copy(dev_wave, dev_wave + dimension, output);
}

}  // namespace Fem
}  // namespace Femog