namespace Fem {

template <class Vertex>
Domain<Vertex>& Domain<Vertex>::add_vertex(const Vertex& vertex) {
  vertex_data_.push_back(vertex);
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

}  // namespace Fem