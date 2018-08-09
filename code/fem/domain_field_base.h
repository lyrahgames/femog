#ifndef FEMOG_FEM_DOMAIN_FIELD_BASE_H_
#define FEMOG_FEM_DOMAIN_FIELD_BASE_H_

#include <array>
#include <list>

namespace Femog::Fem {

class Field_base;

class Domain_base {
 public:
  using Quad = std::array<int, 4>;
  struct Edge;
  struct Primitive;

  Domain_base() = default;
  virtual ~Domain_base() = default;
  Domain_base(Domain_base&&) = default;
  Domain_base& operator=(Domain_base&&) = default;
  Domain_base(const Domain_base&) = default;
  Domain_base& operator=(const Domain_base&) = default;

  const auto& field_list() const { return field_list_; }

 protected:
  friend class Field_base;
  std::list<Field_base*> field_list_;

  void add_values_to_fields();
};

class Field_base {
 public:
  Field_base(Domain_base& domain) : domain_{&domain} {
    domain.field_list_.push_back(this);
  }

  virtual ~Field_base() { domain_->field_list_.remove(this); }

 protected:
  friend class Domain_base;
  Domain_base* domain_;

  virtual void add_value() = 0;
};

struct Domain_base::Edge : public std::array<int, 2> {
  struct Hash {
    std::size_t operator()(const Edge& edge) const {
      return edge[0] ^ (edge[1] << 1);
    }
  };

  struct Info {
    int insertions = 0;
  };

  Edge(int v1, int v2) {
    if (v1 > v2) std::swap(v1, v2);
    front() = v1;
    back() = v2;
  }
};

struct Domain_base::Primitive : public std::array<int, 3> {
  struct Hash {
    std::size_t operator()(const Primitive& primitive) const {
      return primitive[0] ^ (primitive[1] << 1) ^ (primitive[2] << 2);
    }
  };

  struct Info {
    int insertions = 0;
  };

  enum class Error {
    VALID = 0,
    INDICES_DO_NOT_EXIST,
    INDICES_ARE_EQUAL,
    EDGE_CONNECTS_TWO_PRIMITIVES,
    PRIMITIVE_ALREADY_EXISTS
  };

  Primitive(int v1, int v2, int v3) {
    // insertion sort indices
    if (v1 > v2) std::swap(v1, v2);
    if (v2 > v3) std::swap(v2, v3);
    if (v1 > v2) std::swap(v1, v2);
    data()[0] = v1;
    data()[1] = v2;
    data()[2] = v3;
  }
};

}  // namespace Femog::Fem

#endif  // FEMOG_FEM_DOMAIN_FIELD_BASE_H_