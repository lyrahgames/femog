#include <doctest/doctest.h>

#include <permutation.h>

TEST_CASE("The 3-dimensional permutations") {
  using Femog::Permutation;
  using Femog::signum;

  CHECK(signum(Permutation<1, 2, 3>{}) == 1);
  CHECK(signum(Permutation<1, 0, 1>{}) == 0);
  CHECK(signum(Permutation<1, 3, 2>{}) == -1);
  CHECK(signum(Permutation<3, 2, 1>{}) == -1);
  CHECK(signum(Permutation<3, 1, 2>{}) == 1);
}