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

void load_domain_from_file(Domain<Eigen::Vector2f>& domain,
                           const std::string& file_path) {
  using Domain_t = Domain<Eigen::Vector2f>;

  std::fstream file(file_path, std::ios::in);
  if (!file)
    throw std::runtime_error("The file '" + file_path +
                             "' could not be opened! Please check the "
                             "existence and your permissions.");

  std::string command;
  while (file >> command) {
    std::string line;
    std::getline(file, line);
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
}

void load_domain_from_obj(Domain<Eigen::Vector3f>& domain,
                          std::vector<Eigen::Vector3f>& normal_data,
                          const std::string& file_path) {
  using Domain_t = Domain<Eigen::Vector3f>;

  std::fstream file(file_path, std::ios::in);
  if (!file)
    throw std::runtime_error("The file '" + file_path +
                             "' could not be opened! Please check the "
                             "existence and your permissions.");

  std::vector<Eigen::Vector3f> tmp_normals;

  std::string command;
  while (file >> command) {
    std::string line;
    std::getline(file, line);
    std::stringstream arguments{line};

    if (command == "v") {
      Domain_t::Vertex vertex;
      for (int i = 0; i < 3; ++i) arguments >> vertex[i];
      domain << vertex;
      normal_data.push_back(Eigen::Vector3f{});
    } else if (command == "vn") {
      Eigen::Vector3f normal;
      for (int i = 0; i < 3; ++i) {
        arguments >> normal[i];
      }
      tmp_normals.push_back(normal);
    } else if (command == "f") {
      std::array<int, 3> data;
      std::array<int, 3> normal;
      for (int i = 0; i < 3; ++i) {
        arguments >> data[i];
        arguments.ignore(2);
        arguments >> normal[i];
        data[i] -= 1;
        normal[i] -= 1;

        normal_data[data[i]] = tmp_normals[normal[i]];
      }
      domain << Domain_t::Primitive{data[0], data[1], data[2]};
    }
  }
}

}  // namespace Femog::Fem