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
  Gpu_wave_system& copy_wave(Iterator wave_start) const;

  template <typename Iterator>
  Gpu_wave_system& initial_state(Iterator wave_begin, Iterator evolution_begin);

  Gpu_wave_system& advance(value_type dt);

 private:
  Permutation permutation_;
  float* mass_values;
  float* stiffness_values;
  int* row_cdf;
  int* col_index;
  float* wave;
  float* evolution;
  int dimension;
  int nnz;
  int threads_per_block;
  int blocks;

  float* tmp_p;
  float* tmp_r;
  float* tmp_y;
};

}  // namespace Fem