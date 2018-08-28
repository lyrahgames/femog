namespace Fem {

template <typename Domain>
class Cpu_wave_system {
 public:
  using value_type = typename Domain::Vertex::value_type;
  using Field = std::vector<value_type>;
  using Matrix = Eigen::SparseMatrix<value_type, Eigen::RowMajor>;
  using Permutation = std::vector<int>;

  Cpu_wave_system(const Domain& domain);
  ~Cpu_wave_system();

  const Field& wave() const { return wave_; }
  const Field& evolution() const { return evolution_; }
  const Permutation& permutation() const { return permutation_; }

  template <typename Iterator>
  Cpu_wave_system& initial_state(Iterator wave_begin, Iterator evolution_begin);

  Cpu_wave_system& advance(value_type dt);

 private:
  Field wave_;
  Field evolution_;
  Matrix mass_matrix_;
  Matrix stiffness_matrix_;
  Matrix boundary_mass_matrix_;
  Matrix boundary_stiffness_matrix_;
  Permutation permutation_;
};

}  // namespace Fem