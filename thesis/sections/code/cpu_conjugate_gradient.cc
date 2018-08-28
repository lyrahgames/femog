namespace Fem {

void conjugate_gradient(const Eigen::SparseMatrix<float> A, float* data,
                        const Eigen::VectorXf& b) {
  Eigen::Map<Eigen::VectorXf> x(data, b.size());
  Eigen::VectorXf r = A * x - b;
  Eigen::VectorXf p = -r;
  Eigen::VectorXf y;
  float res = r.squaredNorm();
  for (auto i = 0; i < b.size(); ++i) {
    y = A * p;
    const float alpha = res / p.dot(y);
    x += alpha * p;
    r += alpha * y;
    const float new_res = r.squaredNorm();
    if (new_res <= 1e-6f) break;
    const float beta = new_res / res;
    res = new_res;
    p = beta * p - r;
  }
}

}  // namespace Fem