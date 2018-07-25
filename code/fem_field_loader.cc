#include "fem_field_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace Femog {

Fem_field fem_field_file(const std::string& file_path) {
  std::fstream file(file_path, std::ios::in);
  if (!file)
    throw std::runtime_error("The file '" + file_path +
                             "' could not be opened! Please check the "
                             "existence and your permissions.");

  Fem_field field;

  std::string command;
  while (file >> command) {
    std::string line;
    std::getline(file, line);
    std::stringstream arguments{line};

    if (command == "v") {
      Fem_field::vertex_type vertex;
      for (int i = 0; i < 2; ++i) arguments >> vertex[i];
      field.add_vertex(vertex);
    } else if (command == "p") {
      Fem_field::primitive_type primitive;
      for (int i = 0; i < 3; ++i) {
        arguments >> primitive[i];
        primitive[i] -= 1;
      }
      field.add_primitive(primitive);
    } else if (command == "q") {
      Fem_field::quad_type quad;
      for (int i = 0; i < 4; ++i) {
        arguments >> quad[i];
        quad[i] -= 1;
      }
      field.add_quad(quad);
    } else if (command == "n") {
      Fem_field::edge_type edge;
      for (int i = 0; i < 2; ++i) {
        arguments >> edge[i];
        edge[i] -= 1;
      }
      field.set_neumann_boundary(edge);
    } else if (command == "d") {
      Fem_field::edge_type edge;
      for (int i = 0; i < 2; ++i) {
        arguments >> edge[i];
        edge[i] -= 1;
      }
      field.set_dirichlet_boundary(edge);
    }
  }

  return field;
}

}  // namespace Femog
