#ifndef FEMOG_FEM_DOMAIN_H_
#define FEMOG_FEM_DOMAIN_H_

#include <array>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Femog::Fem {

template <class T>
class Domain {
 public:
  using Vertex = T;
  using Quad = std::array<int, 4>;
  struct Edge;
  struct Primitive;

  Domain() = default;
  ~Domain() = default;
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
  Domain& operator<<(const Vertex& vertex) { return this->add_vertex(vertex); }
  Domain& add_primitive(const Primitive& primitive);
  Domain& operator<<(const Primitive& primitive) {
    return this->add_primitive(primitive);
  }
  Domain& add_quad(const Quad& quad);
  Domain& operator<<(const Quad& quad) { return this->add_quad(quad); }

  Domain& subdivide();

 private:
  std::vector<Vertex> vertex_data_;
  std::vector<Primitive> primitive_data_;
  std::unordered_map<Edge, typename Edge::Info, typename Edge::Hash> edge_map_;
  std::unordered_map<Primitive, typename Primitive::Info,
                     typename Primitive::Hash>
      primitive_map_;
};

template <class Vertex>
struct Domain<Vertex>::Edge : public std::array<int, 2> {
  struct Hash {
    std::size_t operator()(const Edge& edge) const {
      return edge[0] ^ (edge[1] << 1);
    }
  };

  struct Info {
    int insertions = 0;
  };

  Edge(int v1, int v2) {
    if (v1 > v2) std::swap(v1, v2);
    front() = v1;
    back() = v2;
  }
};

template <class Vertex>
struct Domain<Vertex>::Primitive : public std::array<int, 3> {
  struct Hash {
    std::size_t operator()(const Primitive& primitive) const {
      return primitive[0] ^ (primitive[1] << 1) ^ (primitive[2] << 2);
    }
  };

  struct Info {
    int insertions = 0;
  };

  enum class Error {
    VALID = 0,
    INDICES_DO_NOT_EXIST,
    INDICES_ARE_EQUAL,
    EDGE_CONNECTS_TWO_PRIMITIVES,
    PRIMITIVE_ALREADY_EXISTS
  };

  Primitive(int v1, int v2, int v3) {
    // insertion sort indices
    if (v1 > v2) std::swap(v1, v2);
    if (v2 > v3) std::swap(v2, v3);
    if (v1 > v2) std::swap(v1, v2);
    data()[0] = v1;
    data()[1] = v2;
    data()[2] = v3;
  }
};

template <class Vertex>
auto Domain<Vertex>::error_code(const Primitive& primitive) const {
  for (int i = 0; i < 3; ++i) {
    // indices must exist in vertex_data
    if (primitive[i] < 0 || primitive[i] >= vertex_data().size())
      return Primitive::Error::INDICES_DO_NOT_EXIST;

    for (int j = 0; j < i; ++j) {
      // indices have to pairwise different
      if (primitive[i] == primitive[j])
        return Primitive::Error::INDICES_ARE_EQUAL;

      // edge must not connect more than two primitives
      auto search = edge_map_.find(Edge{primitive[i], primitive[j]});
      if (search != edge_map_.end() && search->second.insertions >= 2)
        return Primitive::Error::EDGE_CONNECTS_TWO_PRIMITIVES;
    }
  }

  // test for duplication of whole primitive
  auto search = primitive_map_.find(primitive);
  if (search != primitive_map_.end())
    return Primitive::Error::PRIMITIVE_ALREADY_EXISTS;

  return Primitive::Error::VALID;
}

template <class Vertex>
bool Domain<Vertex>::is_valid(const Primitive& primitive) const {
  return error_code(primitive) == Primitive::Error::VALID;
}

template <class Vertex>
void Domain<Vertex>::validate(const Primitive& primitive) const {
  const auto error = error_code(primitive);

  switch (error) {
    case Primitive::Error::INDICES_DO_NOT_EXIST:
      throw std::invalid_argument(
          "Primitive is invalid! One or more of the vertices of the given "
          "indices do not exist.");
      break;

    case Primitive::Error::INDICES_ARE_EQUAL:
      throw std::invalid_argument(
          "Primitive is invalid! The given indices are not pairwise "
          "different.");
      break;

    case Primitive::Error::EDGE_CONNECTS_TWO_PRIMITIVES:
      throw std::invalid_argument(
          "Primitive is invalid! An edge is already connecting two other "
          "primitives.");
      break;

    case Primitive::Error::PRIMITIVE_ALREADY_EXISTS:
      throw std::invalid_argument(
          "Primitive is invalid! The primitive does already exist.");
      break;
  }
}

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

template <class Vertex>
Domain<Vertex>& Domain<Vertex>::subdivide() {
  decltype(edge_map_) edge_subdivide_map;
  edge_subdivide_map.swap(edge_map_);

  decltype(primitive_data_) old_primitive_data;
  old_primitive_data.swap(primitive_data_);

  for (auto& pair : edge_subdivide_map) {
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
  return *this;
}

}  // namespace Femog::Fem

#endif  // FEMOG_FEM_DOMAIN_H_