namespace Fem {

template <class Vertex>
Domain<Vertex>& Domain<Vertex>::subdivide() {
  decltype(edge_map_) edge_subdivide_map;
  edge_subdivide_map.swap(edge_map_);

  decltype(primitive_data_) old_primitive_data;
  old_primitive_data.swap(primitive_data_);
  primitive_map_.clear();

  std::vector<typename decltype(edge_map_)::value_type> boundary_edges;

  for (auto& pair : edge_subdivide_map) {
    if (pair.second.insertions == 1) {
      boundary_edges.push_back(pair);
      boundary_edges.back().second.insertions = vertex_data_.size();
    }

    pair.second.insertions = vertex_data_.size();
    add_vertex(0.5f *
               (vertex_data_[pair.first[0]] + vertex_data_[pair.first[1]]));
  }

  for (auto& primitive : old_primitive_data) {
    const int index_01 =
        edge_subdivide_map.at({primitive[0], primitive[1]}).insertions;
    const int index_12 =
        edge_subdivide_map.at({primitive[1], primitive[2]}).insertions;
    const int index_20 =
        edge_subdivide_map.at({primitive[2], primitive[0]}).insertions;

    add_primitive(Primitive{primitive[0], index_01, index_20});
    add_primitive(Primitive{index_01, index_12, index_20});
    add_primitive(Primitive{index_01, primitive[1], index_12});
    add_primitive(Primitive{index_12, primitive[2], index_20});
  }

  for (auto& pair : boundary_edges) {
    if (pair.second.is_neumann_boundary) {
      set_neumann_boundary(Edge{pair.first[0], pair.second.insertions});
      set_neumann_boundary(Edge{pair.first[1], pair.second.insertions});
    } else {
      set_dirichlet_boundary(Edge{pair.first[0], pair.second.insertions});
      set_dirichlet_boundary(Edge{pair.first[1], pair.second.insertions});
    }
  }

  return *this;
}

}  // namespace Fem