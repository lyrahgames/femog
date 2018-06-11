#include <doctest/doctest.h>

#include <Eigen/Dense>
#include <vector>

namespace Femog {

class Fem_domain {
 public:
  Fem_domain() = default;

 private:
  std::vector<Eigen::Vector2f> vertex_data;
  std::vector<Eigen::Vector3i> primitive_data;
};

}  // namespace Femog

TEST_CASE("The FEM domain") {
  using Femog::Fem_domain;
  Fem_domain femd{};
}