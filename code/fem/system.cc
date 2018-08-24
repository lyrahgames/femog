#include "system.h"
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

#include "conjugate_gradient.h"
#include "gpu_solver.h"
#include "gpu_wave_solver.h"

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
  Eigen::VectorXf rhs = (1.0f - dt() * gamma) * mass_matrix * y -
                        dt() * c * c * stiffness_matrix * x;

  // Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
  // solver.compute(mass_matrix);
  // y = solver.solve(rhs);
  // std::cout << "iterations = " << solver.iterations()
  //           << "\terror = " << solver.error() << std::endl;

  Cg::conjugate_gradient_custom(mass_matrix, evolution().data(), rhs);

  x = x + dt() * y;

  return *this;
}

System& System::gpu_wave_solve() {
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

  const float c = 20.0f;
  const float gamma = 0.0f;
  // Eigen::VectorXf rhs = (1.0f - dt() * gamma) * mass_matrix * y -
  //                       dt() * c * c * stiffness_matrix * x;

  // Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> solver;
  // solver.compute(mass_matrix);
  // y = solver.solve(rhs);
  // std::cout << "iterations = " << solver.iterations()
  //           << "\terror = " << solver.error() << std::endl;

  // Cg::conjugate_gradient_custom(mass_matrix, evolution().data(), rhs);

  // x = x + dt() * y;

  static Wave_solver solver(
      domain().vertex_data().size(), mass_matrix.valuePtr(),
      stiffness_matrix.valuePtr(), mass_matrix.outerIndexPtr(),
      mass_matrix.innerIndexPtr(), wave().data(), evolution().data());
  solver(c, dt());

  static int count = 0;
  ++count;

  if (count == 3) {
    solver.copy_wave(wave().data());
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