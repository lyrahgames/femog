namespace Fem {

template <typename Domain>
Gpu_wave_system<Domain>::~Gpu_wave_system() {
  cudaFree(mass_values_);
  cudaFree(stiffness_values_);
  cudaFree(row_cdf_);
  cudaFree(col_index_);

  cudaFree(boundary_mass_values_);
  cudaFree(boundary_stiffness_values_);
  cudaFree(boundary_row_cdf_);
  cudaFree(boundary_col_index_);

  cudaFree(wave_);
  cudaFree(evolution_);

  cudaFree(tmp_p_);
  cudaFree(tmp_r_);
  cudaFree(tmp_y_);
}

}  // namespace Fem