#include "io.h"
#include <array>

namespace Femog::Fem {

Domain<Eigen::Vector2f> domain_from_stream(std::istream& input) {
  using Domain_t = Domain<Eigen::Vector2f>;
  Domain_t domain;

  std::string command;
  while (input >> command) {
    std::string line;
    std::getline(input, line);
    std::stringstream arguments{line};

    if (command == "v") {
      Domain_t::Vertex vertex;
      for (int i = 0; i < 2; ++i) arguments >> vertex[i];
      domain << vertex;
    } else if (command == "p") {
      // Domain_t::Primitive primitive;
      std::array<int, 3> data;
      for (int i = 0; i < 3; ++i) {
        arguments >> data[i];
        data[i] -= 1;
      }
      domain << Domain_t::Primitive{data[0], data[1], data[2]};
    } else if (command == "q") {
      Domain_t::Quad quad;
      for (int i = 0; i < 4; ++i) {
        arguments >> quad[i];
        quad[i] -= 1;
      }
      domain << quad;
    }
  }

  return domain;
}

}  // namespace Femog::Fem