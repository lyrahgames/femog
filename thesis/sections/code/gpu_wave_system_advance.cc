namespace Fem {

template <typename Domain>
Gpu_wave_system<Domain>& Gpu_wave_system<Domain>::advance(value_type dt) {
  thrust::device_ptr<value_type> dev_evolution(evolution);
  thrust::device_ptr<value_type> dev_wave(wave);
  thrust::device_ptr<value_type> dev_tmp_y(tmp_y);
  thrust::device_ptr<value_type> dev_tmp_p(tmp_p);
  thrust::device_ptr<value_type> dev_tmp_r(tmp_r);

  spmv_csr_kernel<value_type><<<blocks, threads_per_block>>>(
      dimension, 1.0f, mass_values, row_cdf, col_index, evolution, 0.0f, tmp_y);
  spmv_csr_kernel<value_type>
      <<<blocks, threads_per_block>>>(dimension, -dt * c * c, stiffness_values,
                                      row_cdf, col_index, wave, 1.0f, tmp_y);

  spmv_csr_kernel<value_type>
      <<<blocks, threads_per_block>>>(dimension, 1.0f, mass_values, row_cdf,
                                      col_index, evolution, -1.0f, tmp_y);
  thrust::copy(dev_tmp_y, dev_tmp_y + dimension, dev_tmp_r);

  thrust::transform(dev_tmp_r, dev_tmp_r + dimension, dev_tmp_p,
                    thrust::negate<value_type>());

  value_type res =
      thrust::inner_product(dev_tmp_r, dev_tmp_r + dimension, dev_tmp_r, 0.0f);
  int it = 0;

  while ((it < 1 || res > 1e-6f) && it < dimension) {
    spmv_csr_kernel<value_type><<<blocks, threads_per_block>>>(
        dimension, 1.0f, mass_values, row_cdf, col_index, tmp_p, 0.0f, tmp_y);

    const value_type tmp = thrust::inner_product(
        dev_tmp_p, dev_tmp_p + dimension, dev_tmp_y, 0.0f);
    const value_type alpha = res / tmp;

    thrust::transform(dev_tmp_p, dev_tmp_p + dimension, dev_evolution,
                      dev_evolution, saxpy_functor(alpha));
    thrust::transform(dev_tmp_y, dev_tmp_y + dimension, dev_tmp_r, dev_tmp_r,
                      saxpy_functor(alpha));

    const value_type new_res = thrust::inner_product(
        dev_tmp_r, dev_tmp_r + dimension, dev_tmp_r, 0.0f);
    const value_type beta = new_res / res;
    res = new_res;

    thrust::transform(dev_tmp_p, dev_tmp_p + dimension, dev_tmp_r, dev_tmp_p,
                      saxpby_functor(beta, -1.0f));
    ++it;
  }

  thrust::transform(dev_evolution, dev_evolution + dimension, dev_wave,
                    dev_wave, saxpy_functor(dt));

  return *this;
}

}  // namespace Fem