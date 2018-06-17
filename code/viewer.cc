#include "viewer.h"
#include <QApplication>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include "fem_field_loader.h"

namespace Femog {

Viewer::Viewer(QWidget* parent) : QOpenGLWidget(parent) {
  setMouseTracking(true);
  world = Isometry{{}, {0, -1, 0}, {0, 0, 1}};
  eye_azimuth = eye_altitude = M_PI_4;
}

void Viewer::load(const std::string& file_path) {
  // std::fstream file(file_path, std::ios::binary | std::ios::in);
  // if (!file.is_open())
  //   throw std::runtime_error(std::string{"The file '"} + file_path +
  //                            "' could not be opened! Does it exist?");

  // file.ignore(80);
  // std::uint32_t primitive_count;
  // file.read(reinterpret_cast<char*>(&primitive_count), 4);

  // stl_data.resize(3 * primitive_count);

  // for (auto i = 0; i < primitive_count; ++i) {
  //   file.ignore(12);
  //   file.read(reinterpret_cast<char*>(&stl_data[3 * i]), 36);
  //   file.ignore(2);
  // }

  field = fem_field_file(file_path);
  field.subdivide();
  field.subdivide();
  field.subdivide();
  field.subdivide();
  // field.subdivide();

  auto f = [](const Fem_field::vertex_type& vertex) {
    return std::sin(3.0f * vertex.x()) * std::cos(vertex.y());
  };

  for (auto i = 0; i < field.vertex_data().size(); ++i) {
    field.values()[i] = f(field.vertex_data()[i]);
  }

  compute_automatic_view();
}

// void Viewer::generate() {
//   constexpr int count = 100;
//   std::mt19937 rng{std::random_device{}()};
//   std::normal_distribution<float> distribution(0, 0.2);

//   auto f = [](float x, float y) {
//     // return 0.001f * x * y * std::sin(0.5f * x) * std::cos(0.7f * y);
//     return 2.0 * std::sin(0.5 * x) + y;
//   };

//   for (int i = 0; i <= count; ++i) {
//     for (int j = 0; j <= count; ++j) {
//       const float x = static_cast<float>(i) + distribution(rng);
//       const float y = static_cast<float>(j) + distribution(rng);
//       field.vertex_data().push_back({x, y});
//       field.values().push_back(f(x, y));
//     }
//   }

//   for (int i = 0; i < count; ++i) {
//     for (int j = 0; j < count; ++j) {
//       const int index_00 = i * (count + 1) + j;
//       const int index_10 = (i + 1) * (count + 1) + j;
//       field.primitive_data().push_back({index_00, index_10, index_10 + 1});
//       field.primitive_data().push_back({index_00, index_10 + 1, index_00 +
//       1});
//     }
//   }

//   compute_automatic_view();
// }

void Viewer::initializeGL() {
  glClearColor(1.0, 1.0, 1.0, 1.0);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
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
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glBegin(GL_LINES);
  for (const auto& pair : field.edge_data()) {
    if (pair.second == 1)
      glColor3f(1, 0, 0);
    else
      glColor3f(0, 0, 0);

    glVertex3f(field.vertex_data()[pair.first[0]].x(),
               field.vertex_data()[pair.first[0]].y(),
               field.values()[pair.first[0]]);
    glVertex3f(field.vertex_data()[pair.first[1]].x(),
               field.vertex_data()[pair.first[1]].y(),
               field.values()[pair.first[1]]);
  }
  glEnd();

  // glBegin(GL_TRIANGLES);
  // glColor3f(0, 0, 0);

  // for (const auto& primitive : field.primitive_data()) {
  //   for (int i = 0; i < 3; ++i)
  //     glVertex3f(field.vertex_data()[primitive[i]].x(),
  //                field.vertex_data()[primitive[i]].y(),
  //                field.values()[primitive[i]]);
  // }
  // glEnd();
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
  if (event->key() == Qt::Key_Escape) {
    QCoreApplication::quit();
  } else if (event->text() == 'b') {
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
  if (field.vertex_data().size() == 0) return;
  bounding_box_min =
      Eigen::Vector3f(field.vertex_data()[0].x(), field.vertex_data()[0].y(),
                      field.values()[0]);
  bounding_box_max =
      Eigen::Vector3f(field.vertex_data()[0].x(), field.vertex_data()[0].y(),
                      field.values()[0]);

  for (int i = 1; i < field.vertex_data().size(); ++i) {
    bounding_box_max =
        bounding_box_max.array()
            .max(Eigen::Array3f(field.vertex_data()[i].x(),
                                field.vertex_data()[i].y(), field.values()[i]))
            .matrix();
    bounding_box_min =
        bounding_box_min.array()
            .min(Eigen::Array3f(field.vertex_data()[i].x(),
                                field.vertex_data()[i].y(), field.values()[i]))
            .matrix();
  }
}

void Viewer::compute_automatic_view() {
  compute_bounding_box();
  const float scene_radius =
      0.5f * (bounding_box_max - bounding_box_min).norm();
  eye_distance = scene_radius / std::sin(0.5f * camera.field_of_view());
  world.translation() = 0.5f * (bounding_box_min + bounding_box_max);
}

}  // namespace Femog