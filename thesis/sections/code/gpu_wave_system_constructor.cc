namespace Fem {

template <typename Domain>
Gpu_wave_system<Domain>::Gpu_wave_system(const Domain& domain) {
  // construct system matrice and permutation
  // as done in Cpu_wave_system
  // ...

  // allocate memory on DRAM of GPU
  cudaMalloc((void**)&mass_values_, nnz_ * sizeof(value_type));
  cudaMalloc((void**)&stiffness_values_, nnz_ * sizeof(value_type));
  cudaMalloc((void**)&row_cdf_, (inner_dimension_ + 1) * sizeof(int));
  cudaMalloc((void**)&col_index_, nnz_ * sizeof(int));

  cudaMalloc((void**)&boundary_mass_values_,
             boundary_nnz_ * sizeof(value_type));
  cudaMalloc((void**)&boundary_stiffness_values_,
             boundary_nnz_ * sizeof(value_type));
  cudaMalloc((void**)&boundary_row_cdf_, (inner_dimension_ + 1) * sizeof(int));
  cudaMalloc((void**)&boundary_col_index_, boundary_nnz_ * sizeof(int));

  cudaMalloc((void**)&wave_,
             (inner_dimension_ + boundary_dimension_) * sizeof(value_type));
  cudaMalloc((void**)&evolution_,
             (inner_dimension_ + boundary_dimension_) * sizeof(value_type));

  cudaMalloc((void**)&tmp_p_, inner_dimension_ * sizeof(value_type));
  cudaMalloc((void**)&tmp_r_, inner_dimension_ * sizeof(value_type));
  cudaMalloc((void**)&tmp_y_, inner_dimension_ * sizeof(value_type));

  // transfer data from RAM to DRAM
  cudaMemcpy(mass_values_, mass_matrix.valuePtr(), nnz_ * sizeof(value_type),
             cudaMemcpyHostToDevice);
  cudaMemcpy(stiffness_values_, stiffness_matrix.valuePtr(),
             nnz_ * sizeof(value_type), cudaMemcpyHostToDevice);
  cudaMemcpy(row_cdf_, mass_matrix.outerIndexPtr(),
             (inner_dimension_ + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(col_index_, mass_matrix.innerIndexPtr(), nnz_ * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMemcpy(boundary_mass_values_, boundary_mass_matrix.valuePtr(),
             boundary_nnz_ * sizeof(value_type), cudaMemcpyHostToDevice);
  cudaMemcpy(boundary_stiffness_values_, boundary_stiffness_matrix.valuePtr(),
             boundary_nnz_ * sizeof(value_type), cudaMemcpyHostToDevice);
  cudaMemcpy(boundary_row_cdf_, boundary_mass_matrix.outerIndexPtr(),
             (inner_dimension_ + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(boundary_col_index_, boundary_mass_matrix.innerIndexPtr(),
             boundary_nnz_ * sizeof(int), cudaMemcpyHostToDevice);

  // get properties of GPU to compute blocks and threads_per_block
  cudaDeviceProp property;
  cudaGetDeviceProperties(&property, 0);
  threads_per_block_ = property.maxThreadsPerBlock;
  blocks_ = (inner_dimension + threads_per_block - 1) / threads_per_block;
}

}  // namespace Fem