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

  // edge_data_.try_emplace(edge_type{primitive[0], primitive[1]}, false);
  // edge_data_.try_emplace(edge_type{primitive[2], primitive[1]}, false);
  // edge_data_.try_emplace(edge_type{primitive[0], primitive[2]}, false);

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

  // ++edge_data_[edge_type{quad[0], quad[1]}];
  // ++edge_data_[edge_type{quad[1], quad[2]}];
  // ++edge_data_[edge_type{quad[2], quad[3]}];
  // ++edge_data_[edge_type{quad[3], quad[0]}];
  // ++edge_data_[edge_type{quad[0], quad[2]}];

  // primitive_data_.push_back(primitive_1);
  // primitive_data_.push_back(primitive_2);

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
  for (auto& primitive : primitive_data_) {
    const Eigen::Vector2f subdivision_01 =
        0.5f * (vertex_data_[primitive[0]] + vertex_data_[primitive[1]]);
    const Eigen::Vector2f subdivision_12 =
        0.5f * (vertex_data_[primitive[2]] + vertex_data_[primitive[1]]);
    const Eigen::Vector2f subdivision_20 =
        0.5f * (vertex_data_[primitive[0]] + vertex_data_[primitive[2]]);

    vertex_data_.push_back(subdivision_01);
    values_.push_back(0);
    vertex_data_.push_back(subdivision_12);
    values_.push_back(0);
    vertex_data_.push_back(subdivision_20);
    values_.push_back(0);

    const int index_01 = vertex_data_.size() - 3;
    const int index_12 = vertex_data_.size() - 2;
    const int index_20 = vertex_data_.size() - 1;
    const int index_1 = primitive[1];
    const int index_2 = primitive[2];

    primitive[1] = index_01;
    primitive[2] = index_20;

    primitive_data_.push_back(Eigen::Vector3i(index_01, index_1, index_12));
    primitive_data_.push_back(Eigen::Vector3i(index_01, index_12, index_20));
    primitive_data_.push_back(Eigen::Vector3i(index_12, index_2, index_20));
  }
  return *this;
}

}  // namespace Femog