#include <doctest/doctest.h>

#include <isometry.h>

namespace {
template <class Integer>
Integer kronecker_delta(Integer i, Integer j) {
  return i == j;
}
}  // namespace

TEST_CASE("The isometry") {
  using doctest::Approx;
  using Femog::Isometry;

  CHECK(Isometry::MatrixType::RowsAtCompileTime == 3);
  CHECK(Isometry::MatrixType::ColsAtCompileTime == 4);
  CHECK(sizeof(Isometry) == 3 * 4 * sizeof(float));
  CHECK(Isometry::MatrixType::IsRowMajor);

  SUBCASE("is an identity if default constructed.") {
    Isometry isometry;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        CHECK(isometry(i, j) == Approx(kronecker_delta(i, j)));
      }
    }
  }

  SUBCASE("makes sure the 3x3-block is orthogonal by construction.") {
    Isometry isometry{{1, 2, 3}, {1, 1, 0}, {1, 0, 1}};

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        CHECK(isometry.matrix().col(i).dot(isometry.matrix().col(j)) ==
              Approx(kronecker_delta(i, j)));
      }
    }
  }

  SUBCASE("makes sure the 3x3-block respects the given orientation.") {
    CHECK(
        Isometry{{}, {1, 2, 3}, {1, 0, 1}, Isometry::Positive_zy_construction{}}
            .matrix()
            .block(0, 0, 3, 3)
            .determinant() == Approx(1));

    CHECK(
        Isometry{
            {}, {1, 0, 0}, {1, 1, 1}, Isometry::Construction_system<2, 0, -1>{}}
            .matrix()
            .block(0, 0, 3, 3)
            .determinant() == Approx(-1));
  }

  SUBCASE("can be used on a 3-dimensional vector.") {
    Isometry isometry{{1, 2, 3},
                      {0, 1, 0},
                      {0, 0, 1},
                      Isometry::Construction_system<2, 1, -1>{}};
    Eigen::Vector3f v(2, 1, 6);
    REQUIRE(isometry * v == Eigen::Vector3f{3, 8, 4});
  }
}