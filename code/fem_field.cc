#include <fem_field.h>
#include <Eigen/Eigen>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace Femog {

Fem_field& Fem_field::add_vertex(const vertex_type& vertex) {
  vertex_data_.push_back(vertex);
  values_.push_back(0);
  volume_force_.push_back(0);
  return *this;
}

Fem_field& Fem_field::add_primitive(const primitive_type& primitive) {
  if (!is_valid(primitive))
    throw std::invalid_argument(
        "Primitive is referencing one or more vertices that do not exist!");

  ++edge_data_[edge_type{primitive[0], primitive[1]}];
  ++edge_data_[edge_type{primitive[0], primitive[2]}];
  ++edge_data_[edge_type{primitive[1], primitive[2]}];

  primitive_data_.push_back(primitive);
  return *this;
}

Fem_field& Fem_field::add_quad(const quad_type& quad) {
  primitive_type primitive_1{quad[0], quad[1], quad[2]};
  primitive_type primitive_2{quad[0], quad[2], quad[3]};

  if (!is_valid(primitive_1) || !is_valid(primitive_2))
    throw std::invalid_argument(
        "Quad is referencing one or more vertices that do not exist!");

  add_primitive(primitive_1);
  add_primitive(primitive_2);

  return *this;
}

int Fem_field::boundary_size() const {
  int size{0};
  for (const auto& pair : edge_data_) {
    if (std::abs(pair.second) == 1) ++size;
  }
  return size;
}

int Fem_field::dirichlet_boundary_size() const {
  int size{0};
  for (const auto& pair : edge_data_) {
    if (pair.second == 1) ++size;
  }
  return size;
}

int Fem_field::neumann_boundary_size() const {
  int size{0};
  for (const auto& pair : edge_data_) {
    if (pair.second == -1) ++size;
  }
  return size;
}

Fem_field& Fem_field::set_neumann_boundary(const edge_type& edge) {
  auto& value = edge_data_.at(edge);
  if (std::abs(value) != 1)
    throw std::invalid_argument(
        "Could not set Neumann boundary! Given edge is an inner edge.");
  value = -1;
  return *this;
}

Fem_field& Fem_field::set_dirichlet_boundary(const edge_type& edge) {
  auto& value = edge_data_.at(edge);
  if (std::abs(value) != 1)
    throw std::invalid_argument(
        "Could not set Dirichlet boundary! Given edge is an inner edge.");
  value = 1;
  return *this;
}

bool Fem_field::is_neumann_boundary(const edge_type& edge) const {
  auto search = edge_data_.find(edge);
  if (search == edge_data_.end()) return false;
  return (search->second == -1);
}

bool Fem_field::is_dirichlet_boundary(const edge_type& edge) const {
  auto search = edge_data_.find(edge);
  if (search == edge_data_.end()) return false;
  return (search->second == 1);
}

bool Fem_field::is_valid(const primitive_type& primitive) const {
  bool invalid = false;

  for (int i = 0; i < 3; ++i)
    invalid =
        invalid || (primitive[i] < 0 || primitive[i] >= vertex_data().size());

  return !invalid;
}

Fem_field& Fem_field::subdivide() {
  std::unordered_map<edge_type, int, edge_hash, edge_equal> edge_subdivide_map;
  edge_subdivide_map.swap(edge_data_);

  std::vector<primitive_type> old_primitive_data;
  old_primitive_data.swap(primitive_data_);

  for (auto& pair : edge_subdivide_map) {
    pair.second = vertex_data_.size();
    add_vertex(0.5f *
               (vertex_data_[pair.first[0]] + vertex_data_[pair.first[1]]));
  }

  for (auto& primitive : old_primitive_data) {
    const int index_01 = edge_subdivide_map.at({primitive[0], primitive[1]});
    const int index_12 = edge_subdivide_map.at({primitive[1], primitive[2]});
    const int index_20 = edge_subdivide_map.at({primitive[2], primitive[0]});

    add_primitive(primitive_type{primitive[0], index_01, index_20});
    add_primitive(primitive_type{index_01, index_12, index_20});
    add_primitive(primitive_type{index_01, primitive[1], index_12});
    add_primitive(primitive_type{index_12, primitive[2], index_20});
  }

  return *this;
}

Fem_field& Fem_field::solve_poisson_equation() {
  using real_type = float;

  const auto inner_start = std::chrono::system_clock::now();
  std::vector<int> is_boundary(vertex_data_.size(), 0);
  for (const auto& pair : edge_data_) {
    if (pair.second != 1 && pair.second != -1) continue;
    is_boundary[pair.first[0]] = 1;
    is_boundary[pair.first[1]] = 1;
  }

  std::vector<int> inner_vertices;
  for (auto i = 0; i < vertex_data_.size(); ++i) {
    if (is_boundary[i] == 0) inner_vertices.push_back(i);
  }
  const auto inner_end = std::chrono::system_clock::now();
  std::cout << "inner vertices time = "
            << std::chrono::duration<float>(inner_end - inner_start).count()
            << " s" << std::endl;

  const auto primitive_start = std::chrono::system_clock::now();
  std::vector<Eigen::Triplet<real_type>> stiffness_triplets;
  std::vector<Eigen::Triplet<real_type>> mass_triplets;
  Eigen::Matrix<real_type, Eigen::Dynamic, 1> approx_rhs =
      Eigen::Matrix<real_type, Eigen::Dynamic, 1>::Zero(vertex_data_.size());

  for (const auto& primitive : primitive_data_) {
    Fem_field::vertex_type edge[3];

    for (auto i = 0; i < 3; ++i) {
      edge[i] =
          vertex_data()[primitive[(i + 1) % 3]] - vertex_data()[primitive[i]];
    }

    const real_type area =
        0.5 * std::abs(-edge[0].x() * edge[2].y() + edge[0].y() * edge[2].x());
    const real_type inverse_area_4 = 0.25 / area;

    for (unsigned int i = 0; i < 3; ++i) {
      for (unsigned int j = 0; j < 3; ++j) {
        const real_type stiffness_value =
            inverse_area_4 * edge[(i + 1) % 3].dot(edge[(j + 1) % 3]);
        stiffness_triplets.push_back(
            {primitive[i], primitive[j], stiffness_value});

        const real_type mass_value = ((i == j) ? (2.0) : (1.0)) * area / 12.0;
        mass_triplets.push_back({primitive[i], primitive[j], mass_value});
      }

      const real_type mean_force =
          (volume_force()[primitive[0]] + volume_force()[primitive[1]] +
           volume_force()[primitive[2]]) /
          3.0f;
      approx_rhs[primitive[i]] += (area * mean_force);
    }
  }
  const auto primitive_end = std::chrono::system_clock::now();
  std::cout
      << "primitive time = "
      << std::chrono::duration<float>(primitive_end - primitive_start).count()
      << " s" << std::endl;

  const auto assemble_start = std::chrono::system_clock::now();
  Eigen::SparseMatrix<real_type> stiffness_matrix(vertex_data().size(),
                                                  vertex_data().size());
  stiffness_matrix.setFromTriplets(stiffness_triplets.begin(),
                                   stiffness_triplets.end());

  Eigen::SparseMatrix<real_type> mass_matrix(vertex_data_.size(),
                                             vertex_data_.size());
  mass_matrix.setFromTriplets(mass_triplets.begin(), mass_triplets.end());
  const auto assemble_end = std::chrono::system_clock::now();
  std::cout
      << "direct assemble time = "
      << std::chrono::duration<float>(assemble_end - assemble_start).count()
      << " s" << std::endl;

  Eigen::Map<Eigen::VectorXf> force(volume_force_.data(), volume_force_.size());
  Eigen::Matrix<real_type, Eigen::Dynamic, 1> rhs = 3.0 * mass_matrix * force;
  // Eigen::Matrix<real_type, Eigen::Dynamic,1> rhs = approx_rhs;

  Eigen::Matrix<real_type, Eigen::Dynamic, 1> inner_rhs =
      Eigen::Matrix<real_type, Eigen::Dynamic, 1>::Zero(inner_vertices.size());
  for (auto i = 0; i < inner_vertices.size(); ++i) {
    inner_rhs[i] = rhs[inner_vertices[i]];
  }

  Eigen::SparseMatrix<real_type> inner_stiffness_matrix(inner_vertices.size(),
                                                        inner_vertices.size());
  Eigen::SparseMatrix<real_type> inner_mass_matrix(inner_vertices.size(),
                                                   inner_vertices.size());

  const auto misc_start = std::chrono::system_clock::now();
  for (auto i = 0; i < inner_vertices.size(); ++i) {
    for (auto j = 0; j < inner_vertices.size(); ++j) {
      inner_stiffness_matrix.insert(i, j) =
          stiffness_matrix.coeffRef(inner_vertices[i], inner_vertices[j]);
      inner_mass_matrix.insert(i, j) =
          mass_matrix.coeffRef(inner_vertices[i], inner_vertices[j]);
    }
  }
  const auto misc_end = std::chrono::system_clock::now();
  std::cout << "misc time = "
            << std::chrono::duration<float>(misc_end - misc_start).count()
            << " s" << std::endl;

  const auto start = std::chrono::system_clock::now();
  // Eigen::SimplicialLDLT<Eigen::SparseMatrix<real_type>, Eigen::Upper> solver;
  Eigen::ConjugateGradient<Eigen::SparseMatrix<real_type>> solver;
  solver.compute(inner_stiffness_matrix);

  Eigen::Matrix<real_type, Eigen::Dynamic, 1> x =
      Eigen::Matrix<real_type, Eigen::Dynamic, 1>::Zero(inner_vertices.size());
  x = solver.solve(inner_rhs);
  const auto end = std::chrono::system_clock::now();
  std::cout << "sparse solver time = "
            << std::chrono::duration<float>(end - start).count() << " s"
            << std::endl;

  for (auto i = 0; i < values_.size(); ++i) {
    values_[i] = 0;
  }

  for (auto i = 0; i < inner_vertices.size(); ++i) {
    values_[inner_vertices[i]] = x[i];
  }

  return *this;
}

Fem_field& Fem_field::solve_heat_equation() {
  std::vector<int> is_boundary(vertex_data_.size(), 0);
  for (const auto& pair : edge_data_) {
    if (pair.second != 1 && pair.second != -1) continue;
    is_boundary[pair.first[0]] = 1;
    is_boundary[pair.first[1]] = 1;
  }

  std::vector<int> inner_vertices;
  for (auto i = 0; i < vertex_data_.size(); ++i) {
    if (is_boundary[i] == 0) inner_vertices.push_back(i);
  }

  std::vector<Eigen::Triplet<double>> stiffness_triplets;
  std::vector<Eigen::Triplet<double>> mass_triplets;
  Eigen::VectorXd approx_rhs = Eigen::VectorXd::Zero(vertex_data_.size());

  for (const auto& primitive : primitive_data_) {
    Fem_field::vertex_type edge[3];

    for (auto i = 0; i < 3; ++i) {
      edge[i] =
          vertex_data()[primitive[(i + 1) % 3]] - vertex_data()[primitive[i]];
    }

    const double area =
        0.5 * std::abs(-edge[0].x() * edge[2].y() + edge[0].y() * edge[2].x());
    const double inverse_area_4 = 0.25 / area;

    for (unsigned int i = 0; i < 3; ++i) {
      for (unsigned int j = 0; j < 3; ++j) {
        const double stiffness_value =
            inverse_area_4 * edge[(i + 1) % 3].dot(edge[(j + 1) % 3]);
        stiffness_triplets.push_back(
            {primitive[i], primitive[j], stiffness_value});

        const double mass_value = ((i == j) ? (2.0) : (1.0)) * area / 12.0;
        mass_triplets.push_back({primitive[i], primitive[j], mass_value});
      }

      const double mean_force =
          (volume_force()[primitive[0]] + volume_force()[primitive[1]] +
           volume_force()[primitive[2]]) /
          3.0f;
      approx_rhs[primitive[i]] += (area * mean_force);
    }
  }

  Eigen::SparseMatrix<double> stiffness_matrix(vertex_data().size(),
                                               vertex_data().size());
  stiffness_matrix.setFromTriplets(stiffness_triplets.begin(),
                                   stiffness_triplets.end());

  Eigen::SparseMatrix<double> mass_matrix(vertex_data_.size(),
                                          vertex_data_.size());
  mass_matrix.setFromTriplets(mass_triplets.begin(), mass_triplets.end());

  Eigen::Map<Eigen::VectorXf> force(volume_force_.data(), volume_force_.size());
  Eigen::VectorXd rhs = 3.0 * mass_matrix * force.cast<double>();
  // Eigen::VectorXd rhs = approx_rhs;

  Eigen::VectorXd inner_rhs = Eigen::VectorXd::Zero(inner_vertices.size());
  for (auto i = 0; i < inner_vertices.size(); ++i) {
    inner_rhs[i] = rhs[inner_vertices[i]];
  }

  Eigen::SparseMatrix<double> inner_stiffness_matrix(inner_vertices.size(),
                                                     inner_vertices.size());
  Eigen::SparseMatrix<double> inner_mass_matrix(inner_vertices.size(),
                                                inner_vertices.size());

  for (auto i = 0; i < inner_vertices.size(); ++i) {
    for (auto j = 0; j < inner_vertices.size(); ++j) {
      inner_stiffness_matrix.insert(i, j) =
          stiffness_matrix.coeffRef(inner_vertices[i], inner_vertices[j]);
      inner_mass_matrix.insert(i, j) =
          mass_matrix.coeffRef(inner_vertices[i], inner_vertices[j]);
    }
  }

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper> solver;
  solver.compute(inner_stiffness_matrix);

  Eigen::VectorXd x = Eigen::VectorXd::Zero(inner_vertices.size());
  x = solver.solve(inner_rhs);

  for (auto i = 0; i < values_.size(); ++i) {
    values_[i] = 0;
  }

  for (auto i = 0; i < inner_vertices.size(); ++i) {
    values_[inner_vertices[i]] = x[i];
  }

  return *this;
}

}  // namespace Femog