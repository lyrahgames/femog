template <class Vertex>
Domain<Vertex>& Domain<Vertex>::add_vertex(const Vertex& vertex) {
  vertex_data_.push_back(vertex);

  add_values_to_fields();

  return *this;
}

template <class Vertex>
Domain<Vertex>& Domain<Vertex>::add_primitive(const Primitive& primitive) {
  validate(primitive);

  ++edge_map_[Edge{primitive[0], primitive[1]}].insertions;
  ++edge_map_[Edge{primitive[0], primitive[2]}].insertions;
  ++edge_map_[Edge{primitive[1], primitive[2]}].insertions;

  ++primitive_map_[primitive].insertions;
  primitive_data_.push_back(primitive);

  return *this;
}

template <class Vertex>
Domain<Vertex>& Domain<Vertex>::add_quad(const Quad& quad) {
  Primitive primitive_1{quad[0], quad[1], quad[2]};
  Primitive primitive_2{quad[0], quad[2], quad[3]};

  validate(primitive_1);
  validate(primitive_2);

  add_primitive(primitive_1);
  add_primitive(primitive_2);

  return *this;
}

template <class Vertex>
Domain<Vertex>& Domain<Vertex>::subdivide() {
  decltype(edge_map_) edge_subdivide_map;
  edge_subdivide_map.swap(edge_map_);

  decltype(primitive_data_) old_primitive_data;
  old_primitive_data.swap(primitive_data_);

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

template <typename Vertex>
Domain<Vertex>& Domain<Vertex>::set_dirichlet_boundary(const Edge& edge) {
  auto& info = edge_map_.at(edge);
  if (info.insertions != 1)
    throw std::invalid_argument(
        "Could not set Dirichlet boundary! Given edge is an inner edge.");
  info.is_neumann_boundary = false;
  return *this;
}

template <typename Vertex>
Domain<Vertex>& Domain<Vertex>::set_neumann_boundary(const Edge& edge) {
  auto& info = edge_map_.at(edge);
  if (info.insertions != 1)
    throw std::invalid_argument(
        "Could not set Dirichlet boundary! Given edge is an inner edge.");
  info.is_neumann_boundary = true;
  return *this;
}