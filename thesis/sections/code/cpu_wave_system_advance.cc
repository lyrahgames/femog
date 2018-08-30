namespace Fem {

template <typename Domain>
Cpu_wave_system<Domain>& Cpu_wave_system<Domain>::advance(value_type dt) {
  using Vector = Eigen::Matrix<value_type, Eigen::Dynamic, 1>;
  using Vector_map = Eigen::Map<Vector>;

  Vector_map x(wave_.data(), mass_matrix_.cols());
  Vector_map y(evolution_.data(), mass_matrix_.cols());
  Vector_map boundary_x(wave_.data(), boundary_mass_matrix_.cols());
  Vector_map boundary_y(evolution_.data(), boundary_mass_matrix_.cols());

  Vector rhs =
      mass_matrix_ * y + boundary_mass_matrix_ * boundary_y -
      dt * (stiffness_matrix_ * x + boundary_stiffness_matrix_ * boundary_x);
  Eigen::ConjugateGradient<Matrix> solver;
  solver.compute(mass_matrix_);
  y = solver.solve(rhs);
  x = x + dt * y;

  return *this;
}

}  // namespace Fem