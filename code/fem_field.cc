#include <fem_field.h>
#include <stdexcept>

namespace Femog {

Fem_field& Fem_field::add_vertex(const vertex_type& vertex) {
  vertex_data_.push_back(vertex);
  values_.push_back(0);
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

}  // namespace Femog