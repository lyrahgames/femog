#include "domain.h"

namespace Femog::Fem {

Domain& Domain::add_vertex(const Vertex& vertex) {
  vertex_data().push_back(vertex);
  return *this;
}

}  // namespace Femog::Fem