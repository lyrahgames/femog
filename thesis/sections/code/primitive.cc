namespace Fem {

struct Domain_base::Primitive : public std::array<int, 3> {
  struct Hash;
  struct Info;

  Primitive(int v1, int v2, int v3);
};

Domain_base::Primitive::Primitive(int v1, int v2, int v3) {
  // insertion sort indices
  if (v1 > v2) std::swap(v1, v2);
  if (v2 > v3) std::swap(v2, v3);
  if (v1 > v2) std::swap(v1, v2);
  data()[0] = v1;
  data()[1] = v2;
  data()[2] = v3;
}

struct Domain_base::Primitive::Hash {
  std::size_t operator()(const Primitive& primitive) const {
    return primitive[0] ^ (primitive[1] << 1) ^ (primitive[2] << 2);
  }
};

struct Domain_base::Primitive::Info {
  int insertions = 0;
};

}  // namespace Fem