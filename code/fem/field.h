#ifndef FEMOG_FEM_FIELD_H_
#define FEMOG_FEM_FIELD_H_

#include <vector>

#include "domain.h"

namespace Femog::Fem {

template <class T>
class Field : public Field_base {
 public:
  using Value_t = T;

  template <class Vertex>
  Field(Domain<Vertex>& domain)
      : Field_base{domain}, data_(domain.vertex_data().size(), 0) {}

  virtual ~Field() = default;
  Field(Field&&) = default;
  Field& operator=(Field&&) = default;
  Field(const Field&) = default;
  Field& operator=(const Field&) = default;

  const auto& values() const { return data_; }
  auto data() { return data_.data(); }
  auto& operator[](int index) { return data_[index]; }
  const auto& operator[](int index) const { return data_[index]; }

 protected:
  void add_value() override { data_.push_back(T{}); }

 private:
  std::vector<Value_t> data_;
};

}  // namespace Femog::Fem

#endif  // FEMOG_FEM_FIELD_H_