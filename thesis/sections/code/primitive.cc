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