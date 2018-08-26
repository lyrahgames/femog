#include "system.h"
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

#include "conjugate_gradient.h"
#include "gpu_solver.h"

namespace Femog::Fem {

System& System::gpu_solve() {
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

  Eigen::SparseMatrix<float, Eigen::RowMajor> stiffness_matrix(
      domain().vertex_data().size(), domain().vertex_data().size());
  stiffness_matrix.setFromTriplets(stiffness_triplets.begin(),
                                   stiffness_triplets.end());

  Eigen::SparseMatrix<float, Eigen::RowMajor> mass_matrix(
      domain().vertex_data().size(), domain().vertex_data().size());
  mass_matrix.setFromTriplets(mass_triplets.begin(), mass_triplets.end());

  Eigen::Map<Eigen::VectorXf> y(evolution().data(),
                                evolution().values().size());
  Eigen::Map<Eigen::VectorXf> x(wave().data(), wave().values().size());

  const float c = 2.0f;
  const float gamma = 0.0f;

  solve_wave_problem(mass_matrix, stiffness_matrix, x, y, dt(), c);

  // Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
  // solver.compute(mass_matrix);
  // Eigen::VectorXf rhs = (1.0f - dt() * gamma) * mass_matrix * y -
  //                       dt() * c * c * stiffness_matrix * x;

  // conjugate_gradient(mass_matrix, y, rhs);

  // y = solver.solve(rhs);

  // x = x + dt() * y;

  return *this;
}

System& System::solve() {
  // std::vector<Eigen::Triplet<float>> stiffness_triplets;
  // std::vector<Eigen::Triplet<float>> mass_triplets;

  // for (const auto& primitive : domain().primitive_data()) {
  //   Eigen::Vector2f edge[3];

  //   for (auto i = 0; i < 3; ++i) {
  //     edge[i] = domain().vertex_data()[primitive[(i + 1) % 3]] -
  //               domain().vertex_data()[primitive[i]];
  //   }

  //   const float area =
  //       0.5 * std::abs(-edge[0].x() * edge[2].y() + edge[0].y() *
  //       edge[2].x());
  //   const float inverse_area_4 = 0.25 / area;

  //   for (unsigned int i = 0; i < 3; ++i) {
  //     for (unsigned int j = 0; j < 3; ++j) {
  //       const float stiffness_value =
  //           inverse_area_4 * edge[(i + 1) % 3].dot(edge[(j + 1) % 3]);
  //       stiffness_triplets.push_back(
  //           {primitive[i], primitive[j], stiffness_value});

  //       const float mass_value = ((i == j) ? (2.0) : (1.0)) * area / 12.0;
  //       mass_triplets.push_back({primitive[i], primitive[j], mass_value});
  //     }
  //   }
  // }

  // Eigen::SparseMatrix<float> stiffness_matrix(domain().vertex_data().size(),
  //                                             domain().vertex_data().size());
  // stiffness_matrix.setFromTriplets(stiffness_triplets.begin(),
  //                                  stiffness_triplets.end());

  // Eigen::SparseMatrix<float> mass_matrix(domain().vertex_data().size(),
  //                                        domain().vertex_data().size());
  // mass_matrix.setFromTriplets(mass_triplets.begin(), mass_triplets.end());

  Eigen::Map<Eigen::VectorXf> y(evolution().data(),
                                evolution().values().size());
  Eigen::Map<Eigen::VectorXf> x(wave().data(), wave().values().size());

  const float c = 2.0f;
  const float gamma = 0.0f;
  Eigen::VectorXf rhs = (1.0f - dt() * gamma) * mass_matrix * y -
                        dt() * c * c * stiffness_matrix * x;

  // Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
  // solver.compute(mass_matrix);
  // y = solver.solve(rhs);
  // std::cout << "iterations = " << solver.iterations()
  //           << "\terror = " << solver.error() << std::endl;

  Cg::conjugate_gradient(mass_matrix, evolution().data(), rhs);

  x = x + dt() * y;

  return *this;
}

System& System::solve_custom() {
  // Eigen::Map<Eigen::VectorXf> y(evolution().data(),
  //                               evolution().values().size());
  // Eigen::Map<Eigen::VectorXf> x(wave().data(), wave().values().size());

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

void System::generate() {
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

  stiffness_matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>(
      domain().vertex_data().size(), domain().vertex_data().size());
  stiffness_matrix.setFromTriplets(stiffness_triplets.begin(),
                                   stiffness_triplets.end());

  mass_matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>(
      domain().vertex_data().size(), domain().vertex_data().size());
  mass_matrix.setFromTriplets(mass_triplets.begin(), mass_triplets.end());

  wave_solver = new Wave_solver(
      domain().vertex_data().size(), mass_matrix.valuePtr(),
      stiffness_matrix.valuePtr(), mass_matrix.outerIndexPtr(),
      mass_matrix.innerIndexPtr(), wave().data(), evolution().data());
}

void System::generate_with_boundary() {
  // mark boundary vertices
  std::vector<int> is_boundary(domain().vertex_data().size(), 0);
  for (const auto& pair : domain().edge_map()) {
    if (pair.second.insertions != 1 || pair.second.is_neumann_boundary)
      continue;
    is_boundary[pair.first[0]] = 1;
    is_boundary[pair.first[1]] = 1;
  }
  // construct the permutation to separate boundary vertices
  // counting sort: count
  int vertex_count[2] = {0, 0};
  for (auto i = 0; i < is_boundary.size(); ++i) ++vertex_count[is_boundary[i]];
  // counting sort: move
  int inner_vertex_count = vertex_count[0];
  int boundary_vertex_count = vertex_count[1];
  vertex_count[0] = 0;
  vertex_count[1] = inner_vertex_count;
  permutation.resize(is_boundary.size());
  for (auto i = 0; i < is_boundary.size(); ++i) {
    permutation[vertex_count[is_boundary[i]]] = i;
    ++vertex_count[is_boundary[i]];
  }
  // construct inverse permutation
  std::vector<int> inverse_permutation(permutation.size());
  for (auto i = 0; i < permutation.size(); ++i) {
    inverse_permutation[permutation[i]] = i;
  }

  std::cout << "inner_vertex_count = " << inner_vertex_count << std::endl
            << "boundary_vertex_count = " << boundary_vertex_count << std::endl;

  // output
  // for (auto i = 0; i < permutation.size(); ++i) {
  //   std::cout << i << ":\t" << is_boundary[i] << "\t" << permutation[i] <<
  //   "\t"
  //             << inverse_permutation[i] << std::endl;
  // }

  // construct triplets for inner and boundary matrices
  std::vector<Eigen::Triplet<float>> stiffness_triplets;
  std::vector<Eigen::Triplet<float>> boundary_stiffness_triplets;
  std::vector<Eigen::Triplet<float>> mass_triplets;
  std::vector<Eigen::Triplet<float>> boundary_mass_triplets;

  for (const auto& primitive : domain().primitive_data()) {
    Eigen::Vector2f edge[3];

    for (auto i = 0; i < 3; ++i) {
      edge[i] = domain().vertex_data()[primitive[(i + 1) % 3]] -
                domain().vertex_data()[primitive[i]];
    }

    const float area =
        0.5 * std::abs(-edge[0].x() * edge[2].y() + edge[0].y() * edge[2].x());
    const float inverse_area_4 = 0.25 / area;

    // diagonal entries
    for (auto i = 0; i < 3; ++i) {
      if (is_boundary[primitive[i]]) continue;
      const float stiffness_value =
          inverse_area_4 * edge[(i + 1) % 3].squaredNorm();
      const float mass_value = area / 6.0;
      const int index = inverse_permutation[primitive[i]];
      stiffness_triplets.push_back({index, index, stiffness_value});
      mass_triplets.push_back({index, index, mass_value});
    }
    // lower triangle
    for (unsigned int i = 0; i < 3; ++i) {
      if (is_boundary[primitive[i]]) continue;
      const int index_i = inverse_permutation[primitive[i]];

      for (unsigned int j = 0; j < i; ++j) {
        const float stiffness_value =
            inverse_area_4 * edge[(i + 1) % 3].dot(edge[(j + 1) % 3]);
        const float mass_value = area / 12.0;

        if (is_boundary[primitive[j]]) {
          // boundary matrices are not symmetric
          const int index_j =
              inverse_permutation[primitive[j]] - inner_vertex_count;
          boundary_stiffness_triplets.push_back(
              {index_i, index_j, stiffness_value});
          boundary_mass_triplets.push_back({index_i, index_j, mass_value});
        } else {
          // inner matrices have to be symmetric
          const int index_j = inverse_permutation[primitive[j]];
          stiffness_triplets.push_back({index_i, index_j, stiffness_value});
          stiffness_triplets.push_back({index_j, index_i, stiffness_value});
          mass_triplets.push_back({index_i, index_j, mass_value});
          mass_triplets.push_back({index_j, index_i, mass_value});
        }
      }
    }
  }

  // construct matrices from triplets
  stiffness_matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>(
      inner_vertex_count, inner_vertex_count);
  stiffness_matrix.setFromTriplets(stiffness_triplets.begin(),
                                   stiffness_triplets.end());

  boundary_stiffness_matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>(
      inner_vertex_count, boundary_vertex_count);
  boundary_stiffness_matrix.setFromTriplets(boundary_stiffness_triplets.begin(),
                                            boundary_stiffness_triplets.end());

  mass_matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>(inner_vertex_count,
                                                            inner_vertex_count);
  mass_matrix.setFromTriplets(mass_triplets.begin(), mass_triplets.end());

  boundary_mass_matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>(
      inner_vertex_count, boundary_vertex_count);
  boundary_mass_matrix.setFromTriplets(boundary_mass_triplets.begin(),
                                       boundary_mass_triplets.end());

  // output
  // std::cout << "stiffness_matrix:\n" << stiffness_matrix << std::endl;
  // std::cout << "boundary stiffness_matrix:\n"
  //           << boundary_stiffness_matrix << std::endl;
  // std::cout << "mass matrix:\n" << mass_matrix << std::endl;
  // std::cout << "boundary mass_matrix:\n" << boundary_mass_matrix <<
  // std::endl;

  Eigen::VectorXf x(mass_matrix.rows());
  Eigen::VectorXf y(x.size());
  for (int i = 0; i < mass_matrix.rows(); ++i) {
    x[i] = wave()[permutation[i]];
    y[i] = evolution()[permutation[i]];
  }

  wave_solver =
      new Wave_solver(inner_vertex_count, mass_matrix.valuePtr(),
                      stiffness_matrix.valuePtr(), mass_matrix.outerIndexPtr(),
                      mass_matrix.innerIndexPtr(), x.data(), y.data());
}

System& System::gpu_wave_solve() {
  const float c = 2.0f;
  const float gamma = 0.0f;

  (*wave_solver)(c, dt());
  static int count = 0;
  ++count;
  if (count == 5) {
    Eigen::VectorXf x(mass_matrix.rows());
    wave_solver->copy_wave(x.data());
    for (int i = 0; i < x.size(); ++i) wave()[permutation[i]] = x[i];
    count = 0;
  }

  return *this;
}

System3& System3::solve() {
  std::vector<Eigen::Triplet<float>> stiffness_triplets;
  std::vector<Eigen::Triplet<float>> mass_triplets;

  for (const auto& primitive : domain().primitive_data()) {
    Eigen::Vector3f edge[3];

    for (auto i = 0; i < 3; ++i) {
      edge[i] = domain().vertex_data()[primitive[(i + 1) % 3]] -
                domain().vertex_data()[primitive[i]];
    }

    const float dot_product = edge[0].dot(edge[2]);
    const float area =
        0.5 * std::sqrt(edge[0].squaredNorm() * edge[2].squaredNorm() -
                        dot_product * dot_product);
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