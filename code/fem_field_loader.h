#ifndef FEM_FIELD_LOADER
#define FEM_FIELD_LOADER

#include <string>

#include "fem_field.h"

namespace Femog {

Fem_field fem_field_file(const std::string& file_path);

}  // namespace Femog

#endif  // FEM_FIELD_LOADER