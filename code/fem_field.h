#ifndef FEMOG_FEM_FIELD_H_
#define FEMOG_FEM_FIELD_H_

#include <Eigen/Dense>
#include <vector>

namespace Femog {

class Fem_field {
 public:
  Fem_field() = default;

  std::vector<Eigen::Vector2f> vertex_data;
  std::vector<Eigen::Vector3i> primitive_data;
  std::vector<float> values;
};

}  // namespace Femog

#endif  // FEMOG_FEM_FIELD_H_