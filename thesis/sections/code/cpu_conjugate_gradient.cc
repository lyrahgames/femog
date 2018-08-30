template <typename Matrix, typename Vector>
void conjugate_gradient(const Matrix& A, Vector& x, const Vector& b) {
  constexpr float max_error = 1e-6f;

  Vector r = A * x - b;
  Vector p = -r;
  Vector y;
  float res = r.squaredNorm();

  for (auto i = 0; i < b.size(); ++i) {
    y = A * p;
    const float alpha = res / p.dot(y);
    x += alpha * p;
    r += alpha * y;
    const float new_res = r.squaredNorm();
    if (new_res <= max_error) break;
    const float beta = new_res / res;
    res = new_res;
    p = beta * p - r;
  }
}
