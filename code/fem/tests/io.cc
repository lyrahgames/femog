#include <doctest/doctest.h>

#include <fem/io.h>

SCENARIO("Loading a domain from stream") {
  GIVEN("an input stream like a string stream") {
    std::stringstream input{"v 0 0\nv 1 0\nv 0 1"};

    WHEN("the stream is used as input for a domain") {
      Femog::Fem::Domain<Eigen::Vector2f> domain =
          Femog::Fem::domain_from_stream(input);

      THEN("the stream is read according to some rules") {
        CHECK(domain.vertex_data().size() == 3);
      }
    }
  }
}