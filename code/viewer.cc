#include "viewer.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace Femog {

Viewer::Viewer(QWidget* parent) : QOpenGLWidget(parent) {
  setMouseTracking(true);
}

void Viewer::load(const std::string& file_path) {
  std::fstream file(file_path, std::ios::binary | std::ios::in);
  if (!file.is_open())
    throw std::runtime_error(std::string{"The file '"} + file_path +
                             "' could not be opened! Does it exist?");

  file.ignore(80);
  std::uint32_t primitive_count;
  file.read(reinterpret_cast<char*>(&primitive_count), 4);

  stl_data.resize(3 * primitive_count);

  for (auto i = 0; i < primitive_count; ++i) {
    file.ignore(12);
    file.read(reinterpret_cast<char*>(&stl_data[3 * i]), 36);
    file.ignore(2);
  }

  compute_bounding_box();
  const float scene_radius =
      0.5f * (bounding_box_max - bounding_box_min).norm();
  eye_distance = scene_radius / std::sin(0.5f * camera.field_of_view());
  world.translation() = 0.5f * (bounding_box_min + bounding_box_max);
}

void Viewer::initializeGL() {
  glClearColor(1.0, 1.0, 1.0, 1.0);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  // glLoadIdentity();
  // gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0);
  // position = Eigen::Vector3f(0, 0, 10);
  compute_look_at();
}

void Viewer::resizeGL(int width, int height) {
  camera.screen_resolution(width, height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glViewport(0, 0, width, height);
  gluPerspective(45, camera.aspect_ratio(), 0.1, 1000);
  glMatrixMode(GL_MODELVIEW);
}

void Viewer::paintGL() {
  // glLoadIdentity();
  // gluLookAt(position[0], position[1], position[2], 0, 0, 0, 0, 1, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glBegin(GL_TRIANGLES);
  glColor3f(0, 0, 0);
  for (const auto& vertex : stl_data) glVertex3fv(vertex.data());
  glEnd();
}

void Viewer::mousePressEvent(QMouseEvent* event) {}

void Viewer::mouseReleaseEvent(QMouseEvent* event) {}

void Viewer::mouseMoveEvent(QMouseEvent* event) {
  const int x_difference = event->x() - old_mouse_x;
  const int y_difference = event->y() - old_mouse_y;

  if (event->buttons() == Qt::LeftButton) {
    constexpr float eye_altitude_max_abs = M_PI_2 - 0.0001f;
    eye_azimuth += x_difference * 0.01;
    eye_altitude += y_difference * 0.01;
    if (eye_altitude > eye_altitude_max_abs)
      eye_altitude = eye_altitude_max_abs;
    if (eye_altitude < -eye_altitude_max_abs)
      eye_altitude = -eye_altitude_max_abs;

    compute_look_at();
  } else if (event->buttons() == Qt::RightButton) {
    world.translation() =
        world.origin() -
        camera.right() * x_difference * camera.pixel_size() * eye_distance +
        camera.up() * y_difference * camera.pixel_size() * eye_distance;
    compute_look_at();
  }

  old_mouse_x = event->x();
  old_mouse_y = event->y();
}

void Viewer::wheelEvent(QWheelEvent* event) {
  eye_distance += -0.003 * event->angleDelta().y() * eye_distance;
  constexpr float eye_distance_min = 1e-5f;
  if (eye_distance < eye_distance_min) eye_distance = eye_distance_min;
  compute_look_at();
}

void Viewer::keyPressEvent(QKeyEvent* event) {
  if (event->text() == 'b') {
    eye_azimuth = eye_altitude = 0.0f;
    world = Isometry{
        0.5f * (bounding_box_min + bounding_box_max), {0, -1, 0}, {0, 0, 1}};
  } else if (event->text() == 'g') {
    eye_azimuth = eye_altitude = 0.0f;
    world = Isometry{
        0.5f * (bounding_box_min + bounding_box_max), {0, 0, 1}, {0, 1, 0}};
  }

  compute_look_at();
}

void Viewer::keyReleaseEvent(QKeyEvent* event) {}

void Viewer::compute_look_at() {
  Eigen::Vector3f position;
  position[0] = eye_distance * cosf(eye_azimuth) * cosf(eye_altitude);
  position[1] = eye_distance * sinf(eye_altitude);
  position[2] = eye_distance * sinf(eye_azimuth) * cosf(eye_altitude);
  position = world * position;

  camera.look_at(position, world.origin(), world.basis_y());

  makeCurrent();
  glLoadIdentity();
  gluLookAt(position[0], position[1], position[2], world.origin()[0],
            world.origin()[1], world.origin()[2], world.basis_y()[0],
            world.basis_y()[1], world.basis_y()[2]);

  update();
}

void Viewer::compute_bounding_box() {
  if (stl_data.size() == 0) return;
  bounding_box_min = stl_data[0];
  bounding_box_max = stl_data[0];

  for (int i = 1; i < stl_data.size(); ++i) {
    bounding_box_max =
        bounding_box_max.array().max(stl_data[i].array()).matrix();
    bounding_box_min =
        bounding_box_min.array().min(stl_data[i].array()).matrix();
  }
}

}  // namespace Femog