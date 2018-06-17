#ifndef FEMOG_FEM_FIELD_H_
#define FEMOG_FEM_FIELD_H_

#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

namespace Femog {

class Fem_field {
 public:
  using vertex_type = Eigen::Vector2f;
  using primitive_type = Eigen::Vector3i;
  using quad_type = Eigen::Vector4i;
  using edge_type = Eigen::Vector2i;

  struct edge_hash {
    // has to return the same values for keys if edge_equal returns true
    // this hash must be symmetric
    std::size_t operator()(const edge_type& edge) const {
      return (edge[0] ^ (edge[1] << 4)) ^ (edge[1] ^ (edge[0] << 4));
    }
  };

  struct edge_equal {
    bool operator()(const edge_type& edge_1, const edge_type& edge_2) const {
      return (edge_1[0] == edge_2[0] && edge_1[1] == edge_2[1]) ||
             (edge_1[0] == edge_2[1] && edge_1[1] == edge_2[0]);
    }
  };

  Fem_field() = default;
  Fem_field(Fem_field&&) = default;
  Fem_field& operator=(Fem_field&&) = default;
  Fem_field(const Fem_field&) = default;
  Fem_field& operator=(const Fem_field&) = default;

  const std::vector<vertex_type>& vertex_data() const { return vertex_data_; }
  const std::vector<float>& values() const { return values_; }
  const std::vector<primitive_type>& primitive_data() const {
    return primitive_data_;
  }
  const std::unordered_map<edge_type, int, edge_hash, edge_equal>& edge_data()
      const {
    return edge_data_;
  }

  Fem_field& add_vertex(const vertex_type& vertex);
  Fem_field& add_primitive(const primitive_type& primitive);
  Fem_field& add_quad(const quad_type& quad);

  bool is_valid(const primitive_type& primitive) const;

  Fem_field& subdivide();

 private:
  std::vector<vertex_type> vertex_data_;
  std::vector<primitive_type> primitive_data_;
  std::vector<float> values_;
  std::unordered_map<edge_type, int, edge_hash, edge_equal> edge_data_;
};

}  // namespace Femog

#endif  // FEMOG_FEM_FIELD_H_