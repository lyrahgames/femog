namespace Fem {

template <typename Domain>
Cpu_wave_system<Domain>& Cpu_wave_system<Domain>::advance(value_type dt) {
  Eigen::VectorXf x(mass_matrix.rows());
  Eigen::VectorXf y(x.size());
  for (int i = 0; i < mass_matrix.rows(); ++i) {
    x[i] = wave()[permutation[i]];
    y[i] = evolution()[permutation[i]];
  }

  const float c = 2.0f;
  Eigen::VectorXf rhs = mass_matrix * y - dt() * c * c * stiffness_matrix * x;

  // Cg::conjugate_gradient_custom(mass_matrix, evolution().data(), rhs);
  Eigen::ConjugateGradient<Eigen::SparseMatrix<float, Eigen::RowMajor>> solver;
  solver.compute(mass_matrix);
  y = solver.solve(rhs);

  x = x + dt() * y;

  for (int i = 0; i < x.size(); ++i) {
    wave()[permutation[i]] = x[i];
    evolution()[permutation[i]] = y[i];
  }

  return *this;
}

}  // namespace Fem