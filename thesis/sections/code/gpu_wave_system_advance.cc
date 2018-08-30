namespace Fem {

template <typename Domain>
Gpu_wave_system<Domain>& Gpu_wave_system<Domain>::advance(value_type dt) {
  thrust::device_ptr<value_type> dev_evolution(evolution_);
  thrust::device_ptr<value_type> dev_wave(wave_);
  thrust::device_ptr<value_type> dev_tmp_y(tmp_y_);
  thrust::device_ptr<value_type> dev_tmp_p(tmp_p_);
  thrust::device_ptr<value_type> dev_tmp_r(tmp_r_);

  rhs_kernel<value_type><<<blocks, threads_per_block>>>(
      inner_dimension_, mass_values_, stiffness_values_, row_cdf_, col_index_,
      boundary_mass_values_, boundary_stiffness_values_, boundary_row_cdf_,
      boundary_col_index_, wave_, evolution_, dt, tmp_r_);

  spmv_csr_kernel<value_type><<<blocks_, threads_per_block_>>>(
      inner_dimension_, 1.0f, mass_values_, row_cdf_, col_index_, evolution_,
      -1.0f, tmp_r_);

  thrust::transform(dev_tmp_r, dev_tmp_r + inner_dimension_, dev_tmp_p,
                    thrust::negate<value_type>());

  value_type res = thrust::inner_product(
      dev_tmp_r, dev_tmp_r + inner_dimension_, dev_tmp_r, 0.0f);
  int it = 0;

  while ((it < 1 || res > 1e-6f) && it < inner_dimension_) {
    spmv_csr_kernel<value_type><<<blocks_, threads_per_block_>>>(
        inner_dimension_, 1.0f, mass_values_, row_cdf_, col_index_, tmp_p_,
        0.0f, tmp_y_);

    const value_type tmp = thrust::inner_product(
        dev_tmp_p, dev_tmp_p + inner_dimension_, dev_tmp_y, 0.0f);
    const value_type alpha = res / tmp;

    thrust::transform(dev_tmp_p, dev_tmp_p + inner_dimension_, dev_evolution,
                      dev_evolution, axpy_functor<value_type>(alpha));
    thrust::transform(dev_tmp_y, dev_tmp_y + inner_dimension_, dev_tmp_r,
                      dev_tmp_r, axpy_functor<value_type>(alpha));

    const value_type new_res = thrust::inner_product(
        dev_tmp_r, dev_tmp_r + inner_dimension_, dev_tmp_r, 0.0f);
    const value_type beta = new_res / res;
    res = new_res;

    thrust::transform(dev_tmp_p, dev_tmp_p + inner_dimension_, dev_tmp_r,
                      dev_tmp_p, axpby_functor<value_type>(beta, -1.0f));
    ++it;
  }

  thrust::transform(dev_evolution, dev_evolution + inner_dimension_, dev_wave,
                    dev_wave, axpy_functor<value_type>(dt));

  return *this;
}

}  // namespace Fem