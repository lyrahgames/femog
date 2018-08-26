#include "viewer.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

#include <QApplication>
#include <QTimer>

#include <fem/io.h>
#include "fem_field_loader.h"

namespace Femog {

Viewer::Viewer(QWidget* parent) : QOpenGLWidget(parent) {
  setMouseTracking(true);
  world = Isometry{{}, {0, -1, 0}, {0, 0, 1}};
  eye_azimuth = eye_altitude = M_PI_4;

  QTimer* timer = new QTimer(this);
  connect(timer, SIGNAL(timeout()), this, SLOT(repaint()));
  connect(timer, SIGNAL(timeout()), this, SLOT(loop_slot()));
  timer->start(1000.0f / 60.0f);
}

void Viewer::load(const std::string& file_path) {
  obj_switch = (file_path.rfind(".obj") != std::string::npos);

  if (obj_switch) {
    Femog::Fem::load_domain_from_obj(system3.domain(), system_normals,
                                     file_path);
    std::cout << "domain:" << std::endl
              << "vertex count = " << system3.domain().vertex_data().size()
              << std::endl
              << "primitive count = "
              << system3.domain().primitive_data().size() << std::endl
              << "edge_count = " << system3.domain().edge_map().size()
              << std::endl
              << std::endl;
  } else {
    field = fem_field_file(file_path);
    Femog::Fem::load_domain_from_file(system.domain(), file_path);

    // std::cout << "old domain:" << std::endl
    //           << "vertex count = " << field.vertex_data().size() << std::endl
    //           << "primitive count = " << field.primitive_data().size()
    //           << std::endl
    //           << "edge_count = " << field.edge_data().size() << std::endl
    //           << std::endl;

    std::cout << "domain:" << std::endl
              << "vertex count = " << system.domain().vertex_data().size()
              << std::endl
              << "primitive count = " << system.domain().primitive_data().size()
              << std::endl
              << "edge_count = " << system.domain().edge_map().size()
              << std::endl
              << std::endl;
  }

  compute_automatic_view();
}

void Viewer::subdivide(int count) {
  for (auto i = 0; i < count; ++i) {
    // field.subdivide();

    if (obj_switch) {
      system3.domain().subdivide();
      std::cout << "subdivision " << i + 1 << ":" << std::endl
                << "vertex count = " << system.domain().vertex_data().size()
                << std::endl
                << "primitive count = "
                << system.domain().primitive_data().size() << std::endl
                << "edge_count = " << system.domain().edge_map().size()
                << std::endl
                << std::endl;
    } else {
      system.domain().subdivide();
      std::cout << "subdivision " << i + 1 << ":" << std::endl
                << "vertex count = " << system.domain().vertex_data().size()
                << std::endl
                << "primitive count = "
                << system.domain().primitive_data().size() << std::endl
                << "edge_count = " << system.domain().edge_map().size()
                << std::endl
                << std::endl;
    }
  }

  compute_automatic_view();
}

void Viewer::set_analytic_volume_force() {
  if (obj_switch) {
    auto g = [](const Eigen::Vector3f& vertex) {
      const float sigma2 = 0.05;
      return 0.1f * std::exp(-(vertex).squaredNorm() / sigma2) /
             std::sqrt(sigma2);
    };
    const float sigma2 =
        0.05f * 0.5f * (bounding_box_max - bounding_box_min).norm();

    for (auto i = 0; i < system3.domain().vertex_data().size(); ++i) {
      // system3.wave()[i] = g(system3.domain().vertex_data()[i]);
      system3.wave()[i] = 0.1f *
                          std::exp(-(system3.domain().vertex_data()[i] -
                                     system3.domain().vertex_data()[0])
                                        .squaredNorm() /
                                   sigma2) /
                          std::sqrt(sigma2);
      system3.evolution()[i] = 0;
    }
  } else {
    auto f = [](const Fem_field::vertex_type& vertex) {
      return std::cos(3.0f * vertex.x() * M_PI) *
             std::cos(3.0f * vertex.y() * M_PI);
      // return 0;
    };

    auto g = [&](const Fem_field::vertex_type& vertex) {
      const float sigma2 =
          0.01 * 0.5f * (bounding_box_max - bounding_box_min).squaredNorm();
      return 0.05f * (bounding_box_max - bounding_box_min).squaredNorm() *
             std::exp(-(vertex - 0.5f * (bounding_box_max + bounding_box_min)
                                            .block<2, 1>(0, 0))
                           .squaredNorm() /
                      sigma2) /
             std::sqrt(sigma2);

      // return 0.1f *
      //            std::exp(-(vertex - Fem_field::vertex_type{0.25, 0.25})
      //                          .squaredNorm() /
      //                     sigma2) /
      //            std::sqrt(sigma2) -
      //        0.1f *
      //            std::exp(-(vertex - Fem_field::vertex_type{0.75, 0.75})
      //                          .squaredNorm() /
      //                     sigma2) /
      //            std::sqrt(sigma2);
    };

    for (auto i = 0; i < system.domain().vertex_data().size(); ++i) {
      // field.values()[i] = g(field.vertex_data()[i]);
      // field.volume_force()[i] = g(field.vertex_data()[i]);
      // field.volume_force()[i] = 0;

      system.wave()[i] = g(system.domain().vertex_data()[i]);
      // system.wave()[i] = 0.0f;
      // system.evolution()[i] = -10.0f * g(system.domain().vertex_data()[i]);
      system.evolution()[i] = 0.0f;
      // (g(system.domain().vertex_data()[i] + Eigen::Vector2f{0.1f, 0.1f}) -
      //  g(system.domain().vertex_data()[i] - Eigen::Vector2f{0.1f, 0.1f})) /
      // 0.01f;
    }
  }

  system.generate_with_boundary();

  // compute_automatic_view();
}

void Viewer::solve() {
  const auto start = std::chrono::system_clock::now();
  // field.solve_poisson_equation();
  const auto end = std::chrono::system_clock::now();

  const auto time = std::chrono::duration<float>(end - start).count();

  std::cout << "solving time = " << time << " s" << std::endl;

  // for (auto i = 0; i < field.vertex_data().size(); ++i) {
  //   field.values()[i] = 0;
  //   field.volume_force()[i] = 0;
  //   // field.volume_force()[i] = field.values()[i];
  // }

  // field.volume_force()[0] = 1;

  compute_automatic_view();
}

void Viewer::loop_slot() {
  if (!loop_switch) return;

  if (obj_switch) {
    system3.dt() = 0.001f;
    system3.solve();
  } else {
    system.dt() = 0.0001f;
    // system.solve();
    // system.solve_custom();
    system.gpu_wave_solve();
    // system.gpu_solve();
  }
}

void Viewer::initializeGL() {
  if (glewInit() != GLEW_OK)
    throw std::runtime_error("Failed to initialize GLEW!");

  glEnable(GL_DEPTH_TEST);
  // glEnable(GL_MULTISAMPLE);
  // glEnable(GL_POINT_SMOOTH);
  // glEnable(GL_BLEND);
  // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glPointSize(5.0f);
  // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glClearColor(0.95, 0.95, 0.95, 1.0);

  vertex_array = std::unique_ptr<Vertex_array>(new Vertex_array);

  // load, compile, link and validate shaders
  std::fstream vertex_shader_file("../vertex_shader.glsl");
  std::fstream fragment_shader_file("../fragment_shader.glsl");
  Vertex_shader vertex_shader{vertex_shader_file};
  Fragment_shader fragment_shader{fragment_shader_file};
  program = std::unique_ptr<Program>(new Program);
  (*program)
      .attach_shader(vertex_shader)
      .attach_shader(fragment_shader)
      .link()
      .detach_shader(fragment_shader)
      .detach_shader(vertex_shader);

  matrix_id = glGetUniformLocation(program->id(), "model_view_projection");
  light_id = glGetUniformLocation(program->id(), "light_position");

  vertex_buffer = std::unique_ptr<Array_buffer>(new Array_buffer);
  normal_buffer = std::unique_ptr<Array_buffer>(new Array_buffer);
  element_buffer =
      std::unique_ptr<Element_array_buffer>(new Element_array_buffer);
  color_buffer = std::unique_ptr<Array_buffer>(new Array_buffer);

  compute_look_at();
}

void Viewer::resizeGL(int width, int height) {
  camera.screen_resolution(width, height);
  compute_look_at();
}

void Viewer::paintGL() {
  if (render_wireframe_switch) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  } else {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  program->use();

  glUniformMatrix4fv(matrix_id, 1, GL_FALSE, &model_view_projection[0][0]);
  // glUniformMatrix4fv(matrix_id, 1, GL_FALSE, mvp.data());
  glUniform3f(light_id, camera.position().x(), camera.position().y(),
              camera.position().z());

  // field.solve_heat_equation(0.00001f);
  // field.solve_wave_equation(0.001f);

  if (obj_switch) {
    // system3.dt() = 0.001f;
    // system3.solve();

    // std::vector<float> vertex_buffer_data(field.vertex_data().size() * 3);
    std::vector<float> vertex_buffer_data(
        system3.domain().vertex_data().size() * 3);
    float max = -INFINITY;
    float min = INFINITY;
    // for (auto i = 0; i < field.vertex_data().size(); ++i) {
    //   vertex_buffer_data[3 * i + 0] = field.vertex_data()[i].x();
    //   vertex_buffer_data[3 * i + 1] = field.vertex_data()[i].y();
    //   vertex_buffer_data[3 * i + 2] = field.values()[i];

    //   max = std::max(max, field.values()[i]);
    //   min = std::min(min, field.values()[i]);
    // }
    for (auto i = 0; i < system3.domain().vertex_data().size(); ++i) {
      vertex_buffer_data[3 * i + 0] =
          system3.domain().vertex_data()[i].x() +
          0.5f * (bounding_box_max - bounding_box_min).norm() *
              system3.wave()[i] * system_normals[i].x();
      vertex_buffer_data[3 * i + 1] =
          system3.domain().vertex_data()[i].y() +
          0.5f * (bounding_box_max - bounding_box_min).norm() *
              system3.wave()[i] * system_normals[i].y();
      vertex_buffer_data[3 * i + 2] =
          system3.domain().vertex_data()[i].z() +
          0.5f * (bounding_box_max - bounding_box_min).norm() *
              system3.wave()[i] * system_normals[i].z();

      max = std::max(max, system3.wave()[i]);
      min = std::min(min, system3.wave()[i]);
    }
    vertex_buffer->set_data(vertex_buffer_data);

    std::vector<float> normal_buffer_data(
        system3.domain().vertex_data().size() * 3, 0.0f);
    std::vector<int> element_buffer_data(
        system3.domain().primitive_data().size() * 3);
    for (auto i = 0; i < system3.domain().primitive_data().size(); ++i) {
      Eigen::Vector3f vertex[3];
      for (int k = 0; k < 3; ++k) {
        for (int p = 0; p < 3; ++p) {
          vertex[k][p] =
              vertex_buffer_data[3 * system3.domain().primitive_data()[i][k] +
                                 p];
        }
      }

      Eigen::Vector3f normal =
          (vertex[1] - vertex[0]).cross(vertex[2] - vertex[0]);
      normal.normalize();
      if (normal.dot(system_normals[system3.domain().primitive_data()[i][0]]) <
          0)
        normal = -normal;

      for (auto v = 0; v < 3; ++v) {
        element_buffer_data[3 * i + v] =
            system3.domain().primitive_data()[i][v];
        normal_buffer_data[3 * system3.domain().primitive_data()[i][v] + 0] +=
            normal.x();
        normal_buffer_data[3 * system3.domain().primitive_data()[i][v] + 1] +=
            normal.y();
        normal_buffer_data[3 * system3.domain().primitive_data()[i][v] + 2] +=
            normal.z();
      }
    }
    element_buffer->set_data(element_buffer_data);

    std::vector<float> color_buffer_data(system3.domain().vertex_data().size() *
                                         3);
    for (auto i = 0; i < system3.domain().vertex_data().size(); ++i) {
      color_buffer_data[3 * i + 0] = (system3.wave()[i] - min) / (max - min);
      color_buffer_data[3 * i + 1] = color_buffer_data[3 * i + 0];
      color_buffer_data[3 * i + 2] = 1.0f;
    }
    color_buffer->set_data(color_buffer_data);

    for (auto i = 0; i < system3.domain().vertex_data().size(); ++i) {
      Eigen::Vector3f normal(normal_buffer_data[3 * i + 0],
                             normal_buffer_data[3 * i + 1],
                             normal_buffer_data[3 * i + 2]);
      normal.normalize();
      normal_buffer_data[3 * i + 0] = normal.x();
      normal_buffer_data[3 * i + 1] = normal.y();
      normal_buffer_data[3 * i + 2] = normal.z();
    }
    normal_buffer->set_data(normal_buffer_data);

    (*vertex_array)
        .enable_attribute(0, *vertex_buffer)
        .enable_attribute(1, *color_buffer)
        .enable_attribute(2, *normal_buffer);
    element_buffer->draw();
    (*vertex_array).disable_attribute(2).disable_attribute(1);

    vertex_buffer->bind();
    if (render_vertices_switch)
      glDrawArrays(GL_POINTS, 0, vertex_buffer_data.size());
  } else {
    // system.dt() = 0.001f;
    // system.solve();

    // std::vector<float> vertex_buffer_data(field.vertex_data().size() * 3);
    std::vector<float> vertex_buffer_data(system.domain().vertex_data().size() *
                                          3);
    float max = -INFINITY;
    float min = INFINITY;
    // for (auto i = 0; i < field.vertex_data().size(); ++i) {
    //   vertex_buffer_data[3 * i + 0] = field.vertex_data()[i].x();
    //   vertex_buffer_data[3 * i + 1] = field.vertex_data()[i].y();
    //   vertex_buffer_data[3 * i + 2] = field.values()[i];

    //   max = std::max(max, field.values()[i]);
    //   min = std::min(min, field.values()[i]);
    // }
    for (auto i = 0; i < system.domain().vertex_data().size(); ++i) {
      vertex_buffer_data[3 * i + 0] = system.domain().vertex_data()[i].x();
      vertex_buffer_data[3 * i + 1] = system.domain().vertex_data()[i].y();
      vertex_buffer_data[3 * i + 2] = system.wave()[i];

      max = std::max(max, system.wave()[i]);
      min = std::min(min, system.wave()[i]);
    }
    vertex_buffer->set_data(vertex_buffer_data);

    // std::vector<float> normal_buffer_data(field.vertex_data().size() * 3,
    // 0.0f); std::vector<int> element_buffer_data(field.primitive_data().size()
    // * 3); for (auto i = 0; i < field.primitive_data().size(); ++i) {
    //   Eigen::Vector3f vertex[3];
    //   for (int k = 0; k < 3; ++k) {
    //     for (int p = 0; p < 3; ++p) {
    //       vertex[k][p] = vertex_buffer_data[3 * field.primitive_data()[i][k]
    //       + p];
    //     }
    //   }

    //   Eigen::Vector3f normal =
    //       (vertex[1] - vertex[0]).cross(vertex[2] - vertex[0]);
    //   normal.normalize();
    //   if (normal.z() < 0) normal = -normal;

    //   for (auto v = 0; v < 3; ++v) {
    //     element_buffer_data[3 * i + v] = field.primitive_data()[i][v];
    //     normal_buffer_data[3 * field.primitive_data()[i][v] + 0] +=
    //     normal.x(); normal_buffer_data[3 * field.primitive_data()[i][v] + 1]
    //     += normal.y(); normal_buffer_data[3 * field.primitive_data()[i][v] +
    //     2] += normal.z();
    //   }
    // }
    std::vector<float> normal_buffer_data(
        system.domain().vertex_data().size() * 3, 0.0f);
    std::vector<int> element_buffer_data(
        system.domain().primitive_data().size() * 3);
    for (auto i = 0; i < system.domain().primitive_data().size(); ++i) {
      Eigen::Vector3f vertex[3];
      for (int k = 0; k < 3; ++k) {
        for (int p = 0; p < 3; ++p) {
          vertex[k][p] =
              vertex_buffer_data[3 * system.domain().primitive_data()[i][k] +
                                 p];
        }
      }

      Eigen::Vector3f normal =
          (vertex[1] - vertex[0]).cross(vertex[2] - vertex[0]);
      normal.normalize();
      if (normal.z() < 0) normal = -normal;

      for (auto v = 0; v < 3; ++v) {
        element_buffer_data[3 * i + v] = system.domain().primitive_data()[i][v];
        normal_buffer_data[3 * system.domain().primitive_data()[i][v] + 0] +=
            normal.x();
        normal_buffer_data[3 * system.domain().primitive_data()[i][v] + 1] +=
            normal.y();
        normal_buffer_data[3 * system.domain().primitive_data()[i][v] + 2] +=
            normal.z();
      }
    }
    element_buffer->set_data(element_buffer_data);

    // std::vector<float> color_buffer_data(field.vertex_data().size() * 3);
    // for (auto i = 0; i < field.vertex_data().size(); ++i) {
    //   color_buffer_data[3 * i] = (field.values()[i] - min) / (max - min);
    //   color_buffer_data[3 * i + 1] = 1.0f;
    //   color_buffer_data[3 * i + 2] = 0.0f;
    // }
    // color_buffer->set_data(color_buffer_data);
    std::vector<float> color_buffer_data(system.domain().vertex_data().size() *
                                         3);
    for (auto i = 0; i < system.domain().vertex_data().size(); ++i) {
      color_buffer_data[3 * i + 0] = (system.wave()[i] - min) / (max - min);
      color_buffer_data[3 * i + 1] = color_buffer_data[3 * i + 0];
      color_buffer_data[3 * i + 2] = 1.0f;
    }
    color_buffer->set_data(color_buffer_data);

    // for (auto i = 0; i < field.vertex_data().size(); ++i) {
    //   Eigen::Vector3f normal(normal_buffer_data[3 * i + 0],
    //                          normal_buffer_data[3 * i + 1],
    //                          normal_buffer_data[3 * i + 2]);
    //   normal.normalize();
    //   normal_buffer_data[3 * i + 0] = normal.x();
    //   normal_buffer_data[3 * i + 1] = normal.y();
    //   normal_buffer_data[3 * i + 2] = normal.z();
    // }
    // normal_buffer->set_data(normal_buffer_data);
    for (auto i = 0; i < system.domain().vertex_data().size(); ++i) {
      Eigen::Vector3f normal(normal_buffer_data[3 * i + 0],
                             normal_buffer_data[3 * i + 1],
                             normal_buffer_data[3 * i + 2]);
      normal.normalize();
      normal_buffer_data[3 * i + 0] = normal.x();
      normal_buffer_data[3 * i + 1] = normal.y();
      normal_buffer_data[3 * i + 2] = normal.z();
    }
    normal_buffer->set_data(normal_buffer_data);

    (*vertex_array)
        .enable_attribute(0, *vertex_buffer)
        .enable_attribute(1, *color_buffer)
        .enable_attribute(2, *normal_buffer);
    element_buffer->draw();
    (*vertex_array).disable_attribute(2).disable_attribute(1);

    vertex_buffer->bind();
    if (render_vertices_switch)
      glDrawArrays(GL_POINTS, 0, vertex_buffer_data.size());
  }

  const auto current_time = std::chrono::system_clock::now();
  ++frame_count_;
  const auto time_difference =
      std::chrono::duration<float>(current_time - last_time_).count();
  if (time_difference >= 3.0) {
    const auto frame_time = time_difference / static_cast<float>(frame_count_);
    last_time_ = current_time;
    frame_count_ = 0;
    std::cout << "frame time = " << frame_time << " s\t"
              << "fps = " << 1.0f / frame_time << std::endl;
  }
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
  } else if (event->text() == "v") {
    render_vertices_switch = !render_vertices_switch;
  } else if (event->key() == Qt::Key_Space) {
    loop_switch = !loop_switch;
  } else if (event->text() == "r") {
    set_analytic_volume_force();
  } else if (event->text() == "w") {
    render_wireframe_switch = !render_wireframe_switch;
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

  camera.look_at(position, world.origin(), world.basis_y())
      .field_of_view(M_PI_4)
      .near_and_far_distance(0.001f, 10000.0f);
  mvp = (camera.projection_matrix().transpose() *
         camera.view_matrix().transpose())
            .transpose();

  glm::mat4 projection = glm::perspective(
      glm::radians(45.0f), (float)width() / (float)height(), 0.001f, 10000.0f);
  glm::mat4 view = glm::lookAt(
      glm::vec3(position[0], position[1], position[2]),
      glm::vec3(world.origin()[0], world.origin()[1], world.origin()[2]),
      glm::vec3(world.basis_y()[0], world.basis_y()[1], world.basis_y()[2]));
  glm::mat4 model = glm::mat4(1.0f);
  model_view_projection = projection * view * model;

  // std::cout << world.matrix() << std::endl << std::endl;
  // std::cout << camera.view_matrix() << std::endl << std::endl;
  // for (int i = 0; i < 4; ++i) {
  //   for (int j = 0; j < 4; ++j) {
  //     std::cout << view[i][j] << "\t";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  // std::cout << camera.projection_matrix() << std::endl << std::endl;
  // for (int i = 0; i < 4; ++i) {
  //   for (int j = 0; j < 4; ++j) {
  //     std::cout << projection[i][j] << "\t";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  update();
}

void Viewer::compute_bounding_box() {
  if (obj_switch) {
    if (system3.domain().vertex_data().size() == 0) return;
    bounding_box_min = system3.domain().vertex_data()[0];
    bounding_box_max = system3.domain().vertex_data()[0];

    for (int i = 1; i < system3.domain().vertex_data().size(); ++i) {
      bounding_box_max = bounding_box_max.array()
                             .max(system3.domain().vertex_data()[i].array())
                             .matrix();
      bounding_box_min = bounding_box_min.array()
                             .min(system3.domain().vertex_data()[i].array())
                             .matrix();
    }
    return;
  }

  if (system.domain().vertex_data().size() == 0) return;
  bounding_box_min =
      Eigen::Vector3f(system.domain().vertex_data()[0].x(),
                      system.domain().vertex_data()[0].y(), 0.0f);
  bounding_box_max =
      Eigen::Vector3f(system.domain().vertex_data()[0].x(),
                      system.domain().vertex_data()[0].y(), 0.0f);

  for (int i = 1; i < system.domain().vertex_data().size(); ++i) {
    bounding_box_max =
        bounding_box_max.array()
            .max(Eigen::Array3f(system.domain().vertex_data()[i].x(),
                                system.domain().vertex_data()[i].y(), 0.0f))
            .matrix();
    bounding_box_min =
        bounding_box_min.array()
            .min(Eigen::Array3f(system.domain().vertex_data()[i].x(),
                                system.domain().vertex_data()[i].y(), 0.0f))
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