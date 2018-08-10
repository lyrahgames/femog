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

}  // namespace Femog::Fem

#endif  // FEMOG_FEM_IO_H_