namespace Fem {

template <class T>
class Domain : public Domain_base {
 public:
  using Vertex = T;
  using Edge = Domain_base::Edge;
  using Primitive = Domain_base::Primitive;
  using Quad = std::array<int, 4>;

  Domain() = default;
  virtual ~Domain() = default;
  Domain(Domain&&) = default;
  Domain& operator=(Domain&&) = default;
  Domain(const Domain&) = default;
  Domain& operator=(const Domain&) = default;

  const auto& vertex_data() const { return vertex_data_; }
  const auto& primitive_data() const { return primitive_data_; }
  const auto& primitive_map() const { return primitive_map_; }
  const auto& edge_map() const { return edge_map_; }

  auto error_code(const Primitive& primitive) const;
  bool is_valid(const Primitive& primitive) const;
  void validate(const Primitive& primitive) const;

  Domain& add_vertex(const Vertex& vertex);
  Domain& operator<<(const Vertex& vertex) { return add_vertex(vertex); }
  Domain& add_primitive(const Primitive& primitive);
  Domain& operator<<(const Primitive& primitive) {
    return add_primitive(primitive);
  }
  Domain& add_quad(const Quad& quad);
  Domain& operator<<(const Quad& quad) { return add_quad(quad); }

  Domain& set_dirichlet_boundary(const Edge& edge);
  Domain& set_neumann_boundary(const Edge& edge);

  Domain& subdivide();

 private:
  std::vector<Vertex> vertex_data_;
  std::vector<Primitive> primitive_data_;
  std::unordered_map<Edge, typename Edge::Info, typename Edge::Hash> edge_map_;
  std::unordered_map<Primitive, typename Primitive::Info,
                     typename Primitive::Hash>
      primitive_map_;
};

}  // namespace Fem