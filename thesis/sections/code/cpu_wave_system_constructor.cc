namespace Fem {

template <typename Domain>
Cpu_wave_system<Domain>::Cpu_wave_system(const Domain& domain) {
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

  // construct triplets for inner and boundary matrices
  std::vector<Eigen::Triplet<value_type>> stiffness_triplets;
  std::vector<Eigen::Triplet<value_type>> boundary_stiffness_triplets;
  std::vector<Eigen::Triplet<value_type>> mass_triplets;
  std::vector<Eigen::Triplet<value_type>> boundary_mass_triplets;

  for (const auto& primitive : domain().primitive_data()) {
    Eigen::Vector2f edge[3];

    for (auto i = 0; i < 3; ++i) {
      edge[i] = domain().vertex_data()[primitive[(i + 1) % 3]] -
                domain().vertex_data()[primitive[i]];
    }

    const value_type area =
        0.5 * std::abs(-edge[0].x() * edge[2].y() + edge[0].y() * edge[2].x());
    const value_type inverse_area_4 = 0.25 / area;

    // diagonal entries
    for (auto i = 0; i < 3; ++i) {
      if (is_boundary[primitive[i]]) continue;
      const value_type stiffness_value =
          inverse_area_4 * edge[(i + 1) % 3].squaredNorm();
      const value_type mass_value = area / 6.0;
      const int index = inverse_permutation[primitive[i]];
      stiffness_triplets.push_back({index, index, stiffness_value});
      mass_triplets.push_back({index, index, mass_value});
    }
    // lower triangle
    for (unsigned int i = 0; i < 3; ++i) {
      if (is_boundary[primitive[i]]) continue;
      const int index_i = inverse_permutation[primitive[i]];

      for (unsigned int j = 0; j < i; ++j) {
        const value_type stiffness_value =
            inverse_area_4 * edge[(i + 1) % 3].dot(edge[(j + 1) % 3]);
        const value_type mass_value = area / 12.0;

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
  stiffness_matrix = Eigen::SparseMatrix<value_type, Eigen::RowMajor>(
      inner_vertex_count, inner_vertex_count);
  stiffness_matrix.setFromTriplets(stiffness_triplets.begin(),
                                   stiffness_triplets.end());

  boundary_stiffness_matrix = Eigen::SparseMatrix<value_type, Eigen::RowMajor>(
      inner_vertex_count, boundary_vertex_count);
  boundary_stiffness_matrix.setFromTriplets(boundary_stiffness_triplets.begin(),
                                            boundary_stiffness_triplets.end());

  mass_matrix = Eigen::SparseMatrix<value_type, Eigen::RowMajor>(
      inner_vertex_count, inner_vertex_count);
  mass_matrix.setFromTriplets(mass_triplets.begin(), mass_triplets.end());

  boundary_mass_matrix = Eigen::SparseMatrix<value_type, Eigen::RowMajor>(
      inner_vertex_count, boundary_vertex_count);
  boundary_mass_matrix.setFromTriplets(boundary_mass_triplets.begin(),
                                       boundary_mass_triplets.end());
}

}  // namespace Fem