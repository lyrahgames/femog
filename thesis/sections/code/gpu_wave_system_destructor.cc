namespace Fem {

template <typename Domain>
Gpu_wave_system<Domain>::~Gpu_wave_system() {
  cudaFree(mass_values);
  cudaFree(stiffness_values);
  cudaFree(row_cdf);
  cudaFree(col_index);

  cudaFree(wave);
  cudaFree(evolution);

  cudaFree(tmp_p);
  cudaFree(tmp_r);
  cudaFree(tmp_y);
}

}  // namespace Fem