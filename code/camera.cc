#include "camera.h"
#include <cmath>

namespace Femog {

Camera& Camera::frame(const Isometry& frame) {
  frame_ = frame;
  return *this;
}

float Camera::horizontal_field_of_view() const {
  return 2.0 *
         std::atan(std::tan(vertical_field_of_view_ * 0.5f) * aspect_ratio());
}

float Camera::opengl_field_of_view() const {
  return vertical_field_of_view_ * 180 / M_PI;
}

float Camera::pixel_size() const {
  return 2.0f * std::tan(vertical_field_of_view_ * 0.5f) /
         static_cast<float>(screen_height_);
}

float Camera::aspect_ratio() const {
  return static_cast<float>(screen_width_) / static_cast<float>(screen_height_);
}

Camera& Camera::look_at(const Eigen::Vector3f& eye,
                        const Eigen::Vector3f& center,
                        const Eigen::Vector3f& up) {
  frame_ =
      Isometry{eye, eye - center, up, Isometry::Positive_zy_construction{}};
  return *this;
}

Camera& Camera::screen_resolution(int width, int height) {
  screen_height_ = height;
  screen_width_ = width;
  return *this;
}

Camera& Camera::field_of_view(float fov) {
  if (fov > 0.0f || fov < static_cast<float>(M_PI))
    vertical_field_of_view_ = fov;
  return *this;
}

Camera& Camera::vertical_field_of_view(float fov) { return field_of_view(fov); }

Camera& Camera::horizontal_field_of_view(float fov) {
  if (fov > 0.0f || fov < static_cast<float>(M_PI))
    field_of_view(2.0 * std::atan(std::tan(fov * 0.5f) / aspect_ratio()));
  return *this;
}

}  // namespace Femog