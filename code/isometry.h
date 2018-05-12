#ifndef ISOMETRY_H_
#define ISOMETRY_H_

#include <Eigen/Geometry>

#include <permutation.h>

namespace Femog {

using Isometry_base =
    Eigen::Transform<float, 3, Eigen::AffineCompact, Eigen::RowMajor>;

class Isometry : public Isometry_base {
 public:
  template <int I, int J, int Orientation>
  struct Construction_system {
    static constexpr int i = I;
    static constexpr int j = J;
    static constexpr int k = 3 - I - J;
    static constexpr int sign = Orientation;
  };

  using Positive_zy_construction = Construction_system<2, 1, 1>;

  Isometry() : Isometry_base{Isometry_base::Identity()} {}

  template <class System = Positive_zy_construction>
  Isometry(const Eigen::Vector3f& v0, const Eigen::Vector3f& v1,
           System s = Positive_zy_construction{}) {
    // use Gram-Schmidt process to orthonormalize given vectors
    matrix().col(System::i) = v0;
    matrix().col(System::i).normalize();

    matrix().col(System::j) =
        v1 - matrix().col(System::i).dot(v1) * matrix().col(System::i);
    matrix().col(System::j).normalize();

    matrix().col(System::k) =
        System::sign *
        signum(Permutation<System::i + 1, System::j + 1, System::k + 1>{}) *
        matrix().col(System::i).cross(matrix().col(System::j));
    matrix().col(System::k).normalize();
  }
};

}  // namespace Femog

#endif  // ISOMETRY_H_