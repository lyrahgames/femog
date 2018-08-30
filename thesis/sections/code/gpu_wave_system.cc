namespace Fem {

template <typename Domain>
class Gpu_wave_system {
 public:
  using value_type = typename Domain::Vertex::value_type;
  using Permutation = std::vector<int>;

  Gpu_wave_system(const Domain& domain);
  ~Gpu_wave_system();

  const Permutation& permutation() const { return permutation_; }
  template <typename Iterator>
  Gpu_wave_system& copy_wave(Iterator wave_begin) const;
  template <typename Iterator>
  Gpu_wave_system& initial_state(Iterator wave_begin, Iterator evolution_begin);
  Gpu_wave_system& advance(value_type dt);

 private:
  Permutation permutation_;

  // CSR format of inner matrices
  value_type* mass_values_;
  value_type* stiffness_values_;
  int* row_cdf_;
  int* col_index_;
  int nnz_;

  // CSR format of boundary matrices
  value_type* boundary_mass_values_;
  value_type* boundary_stiffness_values_;
  int* boundary_row_cdf_;
  int* boundary_col_index_;
  int boundary_nnz_;

  // handlers for wave data
  value_type* wave_;
  value_type* evolution_;

  // inner matrix dimension: inner_dimension x inner_dimension
  // boundary matrix dimension: inner_dimension x boundary_dimension
  int inner_dimension_;
  int boundary_dimension_;

  // values to efficiently launch GPU kernel
  int threads_per_block_;
  int blocks_;

  // preallocated data for faster computation
  // of conjugate gradient method
  value_type* tmp_p_;
  value_type* tmp_r_;
  value_type* tmp_y_;
};

}  // namespace Fem