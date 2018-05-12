#include <doctest/doctest.h>

#include <camera.h>

TEST_CASE("The camera") {
  using doctest::Approx;
  using Femog::Camera;

  Camera camera;

  SUBCASE("is valid if default constructed.") {
    CHECK(camera.field_of_view() > 0.0f);
    CHECK(camera.field_of_view() < M_PI);
    CHECK(camera.screen_width() > 0);
    CHECK(camera.screen_height() > 0);

    CHECK(camera.frame() * Eigen::Vector3f{1, 0, 0} ==
          Eigen::Vector3f{1, 0, 0});
    CHECK(camera.frame() * Eigen::Vector3f{0, 1, 0} ==
          Eigen::Vector3f{0, 1, 0});
    CHECK(camera.frame() * Eigen::Vector3f{0, 0, 1} ==
          Eigen::Vector3f{0, 0, 1});
  }

  SUBCASE("methods can be used with chaining.") {
    camera.field_of_view(0.1f).screen_resolution(1000, 500).look_at(
        {3, 1, 1}, {1, 1, 1}, {0, 1, 0});

    CHECK(camera.field_of_view() == Approx(0.1f));
    CHECK(camera.screen_width() == 1000);
    CHECK(camera.screen_height() == 500);

    CHECK(camera.frame() * Eigen::Vector3f{1, 0, 0} ==
          Eigen::Vector3f{3, 1, 0});
    CHECK(camera.frame() * Eigen::Vector3f{0, 1, 0} ==
          Eigen::Vector3f{3, 2, 1});
    CHECK(camera.frame() * Eigen::Vector3f{0, 0, 1} ==
          Eigen::Vector3f{4, 1, 1});
  }

  SUBCASE("computes the aspect ratio.") {
    CHECK(camera.aspect_ratio() == Approx(16.0 / 9));

    camera.screen_resolution(1000, 600);
    CHECK(camera.aspect_ratio() == Approx(10.0 / 6));
  }

  SUBCASE("computes the pixel size in the camera space with distance 1.") {
    CHECK(camera.pixel_size() == Approx(2.301186e-3f));
  }

  SUBCASE("throws an exception if screen resolution should be negative.") {
    CHECK_THROWS_AS(camera.screen_resolution(-1, 500), std::invalid_argument);
    CHECK_THROWS_AS(camera.screen_resolution(500, 0), std::invalid_argument);
    CHECK_THROWS_AS(camera.screen_resolution(0, 0), std::invalid_argument);
  }

  SUBCASE(
      "throws an exception if field of view should be negative or too big.") {
    CHECK_THROWS_AS(camera.field_of_view(M_PI), std::invalid_argument);
    CHECK_THROWS_AS(camera.field_of_view(0.0f), std::invalid_argument);
    CHECK_THROWS_AS(camera.horizontal_field_of_view(M_PI),
                    std::invalid_argument);
    CHECK_THROWS_AS(camera.horizontal_field_of_view(-1.0),
                    std::invalid_argument);
  }
}