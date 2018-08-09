#include "domain_field_base.h"

namespace Femog::Fem {

void Domain_base::add_values_to_fields() {
  for (auto field : field_list_) field->add_value();
}

}  // namespace Femog::Fem