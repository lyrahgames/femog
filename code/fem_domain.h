#ifndef FEMOG_FEM_DOMAIN_H_
#define FEMOG_FEM_DOMAIN_H_

#include <Eigen/Dense>
#include <vector>

namespace Femog {

class Fem_domain {
 public:
  Fem_domain() = default;

  std::vector<Eigen::Vector2f> vertex_data;
  std::vector<Eigen::Vector3i> primitive_data;
};

}  // namespace Femog

#endif  // FEMOG_FEM_DOMAIN_H_