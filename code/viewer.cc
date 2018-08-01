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
  field = fem_field_file(file_path);

  std::cout << "domain:" << std::endl
            << "vertex count = " << field.vertex_data().size() << std::endl
            << "primitive count = " << field.primitive_data().size()
            << std::endl
            << "edge_count = " << field.edge_data().size() << std::endl
            << std::endl;

  // const int max = 20;

  // for (int i = 0; i < max; ++i) {
  //   for (int j = 0; j < max; ++j) {
  //     field.add_vertex(
  //         {static_cast<float>(i) / max, static_cast<float>(j) / max});
  //   }
  // }

  // for (int i = 0; i < max - 1; ++i) {
  //   for (int j = 0; j < max - 1; ++j) {
  //     const int index = max * i + j;
  //     const int index1 = max * (i + 1) + j;
  //     field.add_quad({index, index + 1, index1 + 1, index1});
  //   }
  // }

  compute_automatic_view();
}

void Viewer::subdivide(int count) {
  for (auto i = 0; i < count; ++i) {
    field.subdivide();

    std::cout << "subdivision " << i + 1 << ":" << std::endl
              << "vertex count = " << field.vertex_data().size() << std::endl
              << "primitive count = " << field.primitive_data().size()
              << std::endl
              << "edge_count = " << field.edge_data().size() << std::endl
              << std::endl;
  }

  compute_automatic_view();
}

void Viewer::set_analytic_volume_force() {
  auto f = [](const Fem_field::vertex_type& vertex) {
    // return std::sin(3.0f * vertex.x()) * std::cos(vertex.y());
    return 0;
  };

  auto g = [](const Fem_field::vertex_type& vertex) {
    const float sigma2 = 0.05;
    // return std::exp(-(vertex - Fem_field::vertex_type{0.5,
    // 0.5}).squaredNorm() /
    //                 sigma2) /
    //        std::sqrt(sigma2);

    return std::exp(
               -(vertex - Fem_field::vertex_type{0.25, 0.25}).squaredNorm() /
               sigma2) /
               std::sqrt(sigma2) -
           std::exp(
               -(vertex - Fem_field::vertex_type{0.75, 0.75}).squaredNorm() /
               sigma2) /
               std::sqrt(sigma2);
  };

  for (auto i = 0; i < field.vertex_data().size(); ++i) {
    field.values()[i] = f(field.vertex_data()[i]);
    field.volume_force()[i] = g(field.vertex_data()[i]);
  }

  compute_automatic_view();
}

void Viewer::solve() {
  field.solve_poisson_equation();
  compute_automatic_view();
}

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

  if (render_volume_force) {
    glBegin(GL_LINES);
    for (const auto& pair : field.edge_data()) {
      if (pair.second == 1)
        glColor3f(1, 0, 0);
      else if (pair.second == -1)
        glColor3f(0, 0, 1);
      else
        glColor3f(0, 0, 0);

      glVertex3f(field.vertex_data()[pair.first[0]].x(),
                 field.vertex_data()[pair.first[0]].y(),
                 field.volume_force()[pair.first[0]]);
      glVertex3f(field.vertex_data()[pair.first[1]].x(),
                 field.vertex_data()[pair.first[1]].y(),
                 field.volume_force()[pair.first[1]]);
    }
    glEnd();
    // glBegin(GL_TRIANGLES);
    // for (const auto& primitive : field.primitive_data()) {
    //   const double mean_force = (field.volume_force()[primitive[0]] +
    //                              field.volume_force()[primitive[1]] +
    //                              field.volume_force()[primitive[2]]) /
    //                             3.0f;
    //   for (int i = 0; i < 3; ++i)
    //     glVertex3f(field.vertex_data()[primitive[i]].x(),
    //                field.vertex_data()[primitive[i]].y(), mean_force);
    // }
    // glEnd();
  } else {
    glBegin(GL_LINES);
    for (const auto& pair : field.edge_data()) {
      if (pair.second == 1)
        glColor3f(1, 0, 0);
      else if (pair.second == -1)
        glColor3f(0, 0, 1);
      else
        glColor3f(0, 0, 0);

      constexpr float scale = 1.0f;

      glVertex3f(field.vertex_data()[pair.first[0]].x(),
                 field.vertex_data()[pair.first[0]].y(),
                 scale * field.values()[pair.first[0]]);
      glVertex3f(field.vertex_data()[pair.first[1]].x(),
                 field.vertex_data()[pair.first[1]].y(),
                 scale * field.values()[pair.first[1]]);
    }
    glEnd();
  }

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
  } else if (event->text() == "b") {
    eye_azimuth = eye_altitude = 0.0f;
    world = Isometry{
        0.5f * (bounding_box_min + bounding_box_max), {0, -1, 0}, {0, 0, 1}};
  } else if (event->text() == "g") {
    eye_azimuth = eye_altitude = 0.0f;
    world = Isometry{
        0.5f * (bounding_box_min + bounding_box_max), {0, 0, 1}, {0, 1, 0}};
  } else if (event->text() == "f") {
    render_volume_force = !render_volume_force;
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