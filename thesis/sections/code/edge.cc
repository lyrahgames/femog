namespace Fem {

struct Domain_base::Edge : public std::array<int, 2> {
  struct Hash;
  struct Info;

  Edge(int v1, int v2);
};

Domain_base::Edge::Edge(int v1, int v2) {
  if (v1 > v2) std::swap(v1, v2);
  front() = v1;
  back() = v2;
}

struct Domain_base::Edge::Hash {
  std::size_t operator()(const Edge& edge) const {
    return edge[0] ^ (edge[1] << 1);
  }
};

struct Domain_base::Edge::Info {
  int insertions = 0;
  bool is_neumann_boundary = false;
};

}  // namespace Fem