struct Domain_base {
 public:
  struct Edge;
  struct Primitive;

  Domain_base() = default;
  virtual ~Domain_base() = default;
  Domain_base(Domain_base&&) = default;
  Domain_base& operator=(Domain_base&&) = default;
  Domain_base(const Domain_base&) = default;
  Domain_base& operator=(const Domain_base&) = default;
};