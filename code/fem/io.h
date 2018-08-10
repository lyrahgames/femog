#ifndef FEMOG_FEM_IO_H_
#define FEMOG_FEM_IO_H_

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Dense>

#include "domain.h"

namespace Femog::Fem {

Domain<Eigen::Vector2f> domain_from_stream(std::istream& input);
void load_domain_from_file(Domain<Eigen::Vector2f>& domain,
                           const std::string& file_path);

void load_domain_from_obj(Domain<Eigen::Vector3f>& domain,
                          std::vector<Eigen::Vector3f>& normal_data,
                          const std::string& file_path);

}  // namespace Femog::Fem

#endif  // FEMOG_FEM_IO_H_