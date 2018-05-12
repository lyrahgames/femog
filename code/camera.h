#ifndef FEMOG_CAMERA_H_
#define FEMOG_CAMERA_H_

#include <isometry.h>

namespace Femog {

class Camera {
  using Vector3f = Eigen::Vector3f;

 public:
  Vector3f position() const { return frame_.origin(); }
  Vector3f direction() const { return -frame_.basis_z(); }
  Vector3f right() const { return frame_.basis_x(); }
  Vector3f up() const { return frame_.basis_y(); }

  Camera& frame(const Isometry& frame);
  Isometry& frame() { return frame_; }
  const Isometry& frame() const { return frame_; }

  float field_of_view() const { return vertical_field_of_view_; }
  float vertical_field_of_view() const { return vertical_field_of_view_; }
  float horizontal_field_of_view() const;

  float opengl_field_of_view() const;
  int screen_width() const { return screen_width_; }
  int screen_height() const { return screen_height_; }
  float pixel_size() const;
  float aspect_ratio() const;

  Camera& look_at(const Vector3f& eye, const Vector3f& center,
                  const Vector3f& up);
  Camera& screen_resolution(int width, int height);
  Camera& field_of_view(float fov);
  Camera& vertical_field_of_view(float fov);
  Camera& horizontal_field_of_view(float fov);

 private:
  Isometry frame_;
  float vertical_field_of_view_;
  int screen_height_;
  int screen_width_;
};

}  // namespace Femog

#endif  // FEMOG_CAMERA_H_