#include "system.h"
#include <Eigen/Sparse>
#include <vector>

namespace Femog::Fem {

System& System::solve() {
  std::vector<Eigen::Triplet<float>> stiffness_triplets;
  std::vector<Eigen::Triplet<float>> mass_triplets;

  for (const auto& primitive : domain().primitive_data()) {
    Eigen::Vector2f edge[3];

    for (auto i = 0; i < 3; ++i) {
      edge[i] = domain().vertex_data()[primitive[(i + 1) % 3]] -
                domain().vertex_data()[primitive[i]];
    }

    const float area =
        0.5 * std::abs(-edge[0].x() * edge[2].y() + edge[0].y() * edge[2].x());
    const float inverse_area_4 = 0.25 / area;

    for (unsigned int i = 0; i < 3; ++i) {
      for (unsigned int j = 0; j < 3; ++j) {
        const float stiffness_value =
            inverse_area_4 * edge[(i + 1) % 3].dot(edge[(j + 1) % 3]);
        stiffness_triplets.push_back(
            {primitive[i], primitive[j], stiffness_value});

        const float mass_value = ((i == j) ? (2.0) : (1.0)) * area / 12.0;
        mass_triplets.push_back({primitive[i], primitive[j], mass_value});
      }
    }
  }

  Eigen::SparseMatrix<float> stiffness_matrix(domain().vertex_data().size(),
                                              domain().vertex_data().size());
  stiffness_matrix.setFromTriplets(stiffness_triplets.begin(),
                                   stiffness_triplets.end());

  Eigen::SparseMatrix<float> mass_matrix(domain().vertex_data().size(),
                                         domain().vertex_data().size());
  mass_matrix.setFromTriplets(mass_triplets.begin(), mass_triplets.end());

  Eigen::Map<Eigen::VectorXf> y(evolution().data(),
                                evolution().values().size());
  Eigen::Map<Eigen::VectorXf> x(wave().data(), wave().values().size());

  const float c = 2.0f;
  const float gamma = 0.0f;

  Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
  solver.compute(mass_matrix);
  Eigen::VectorXf rhs = (1.0f - dt() * gamma) * mass_matrix * y -
                        dt() * c * c * stiffness_matrix * x;

  y = solver.solve(rhs);
  x = x + dt() * y;

  return *this;
}

}  // namespace Femog::Fem