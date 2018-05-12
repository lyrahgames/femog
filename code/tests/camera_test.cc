#include <doctest/doctest.h>

#include <camera.h>

TEST_CASE("The camera") {
  using Femog::Camera;

  Camera camera;
  MESSAGE("camera test \n" << camera.position());
}