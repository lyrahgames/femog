namespace Fem {

template <typename Domain>
template <typename Iterator>
Gpu_wave_system<Domain>& Gpu_wave_system<Domain>::copy_wave(
    Iterator wave_start) const {
  thrust::device_ptr<value_type> dev_wave(wave);
  thrust::copy(dev_wave, dev_wave + dimension, wave_start);
}

template <typename Domain>
template <typename Iterator>
Gpu_wave_system<Domain>& Gpu_wave_system<Domain>::initial_state(
    Iterator wave_begin, Iterator evolution_begin) {
  cudaMemcpy(wave, wave_begin, dimension * sizeof(value_type),
             cudaMemcpyHostToDevice);
  cudaMemcpy(evolution, evolution_begin, dimension * sizeof(value_type),
             cudaMemcpyHostToDevice);
}

}  // namespace Fem