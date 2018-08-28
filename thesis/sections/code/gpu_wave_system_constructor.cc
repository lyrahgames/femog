namespace Fem {

template <typename Domain>
Gpu_wave_system<Domain>::Gpu_wave_system(const Domain& domain) {
  // construct mass and stiffness matrix
  // ...

  // allocate memory on DRAM of GPU
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
  cudaGetDeviceProperties(&property, 0);

  threads_per_block = property.maxThreadsPerBlock;
  blocks = (dimension + threads_per_block - 1) / threads_per_block;
}

}  // namespace Fem