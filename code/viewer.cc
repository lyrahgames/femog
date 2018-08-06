#include "viewer.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

#include <QApplication>

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
  const auto start = std::chrono::system_clock::now();
  field.solve_poisson_equation();
  const auto end = std::chrono::system_clock::now();

  const auto time = std::chrono::duration<float>(end - start).count();

  std::cout << "solving time = " << time << " s" << std::endl;

  compute_automatic_view();
}

void Viewer::initializeGL() {
  if (glewInit() != GLEW_OK)
    throw std::runtime_error("Failed to initialize GLEW!");

  glEnable(GL_DEPTH_TEST);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glClearColor(1.0, 1.0, 1.0, 1.0);

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

  matrix_id = glGetUniformLocation(program->id(), "MVP");
  glm::mat4 projection = glm::perspective(
      glm::radians(45.0f), (float)width() / (float)height(), 0.1f, 100.0f);
  glm::mat4 view =
      glm::lookAt(glm::vec3(4, 3, -3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  glm::mat4 model = glm::mat4(1.0f);
  model_view_projection = projection * view * model;

  std::vector<float> vertex_buffer_data{-1, -1, -1, -1, -1, 1,  -1, 1,
                                        1,  -1, 1,  -1, 1,  -1, -1, 1,
                                        -1, 1,  1,  1,  1,  1,  1,  -1};
  vertex_buffer = std::unique_ptr<Array_buffer>(new Array_buffer);
  vertex_buffer->set_data(vertex_buffer_data);

  std::vector<int> element_buffer_data{0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7};
  element_buffer =
      std::unique_ptr<Element_array_buffer>(new Element_array_buffer);
  element_buffer->set_data(element_buffer_data);

  std::vector<float> color_buffer_data{
      0.583f, 0.771f, 0.014f, 0.609f, 0.115f, 0.436f, 0.327f, 0.483f, 0.844f,
      0.822f, 0.569f, 0.201f, 0.435f, 0.602f, 0.223f, 0.310f, 0.747f, 0.185f,
      0.597f, 0.770f, 0.761f, 0.559f, 0.436f, 0.730f, 0.359f, 0.583f, 0.152f,
      0.483f, 0.596f, 0.789f, 0.559f, 0.861f, 0.639f, 0.195f, 0.548f, 0.859f,
      0.014f, 0.184f, 0.576f, 0.771f, 0.328f, 0.970f, 0.406f, 0.615f, 0.116f,
      0.676f, 0.977f, 0.133f, 0.971f, 0.572f, 0.833f, 0.140f, 0.616f, 0.489f,
      0.997f, 0.513f, 0.064f, 0.945f, 0.719f, 0.592f, 0.543f, 0.021f, 0.978f,
      0.279f, 0.317f, 0.505f, 0.167f, 0.620f, 0.077f, 0.347f, 0.857f, 0.137f,
      0.055f, 0.953f, 0.042f, 0.714f, 0.505f, 0.345f, 0.783f, 0.290f, 0.734f,
      0.722f, 0.645f, 0.174f, 0.302f, 0.455f, 0.848f, 0.225f, 0.587f, 0.040f,
      0.517f, 0.713f, 0.338f, 0.053f, 0.959f, 0.120f, 0.393f, 0.621f, 0.362f,
      0.673f, 0.211f, 0.457f, 0.820f, 0.883f, 0.371f, 0.982f, 0.099f, 0.879f};
  color_buffer = std::unique_ptr<Array_buffer>(new Array_buffer);
  color_buffer->set_data(color_buffer_data);

  compute_look_at();
}

void Viewer::resizeGL(int width, int height) {
  camera.screen_resolution(width, height);

  // glMatrixMode(GL_PROJECTION);
  // glLoadIdentity();
  // glViewport(0, 0, width, height);
  // gluPerspective(45, camera.aspect_ratio(), 0.1, 1000);
  // glMatrixMode(GL_MODELVIEW);

  // matrix_id = glGetUniformLocation(program->id(), "MVP");
  glm::mat4 projection = glm::perspective(
      glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
  glm::mat4 view =
      glm::lookAt(glm::vec3(4, 3, -3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  glm::mat4 model = glm::mat4(1.0f);
  model_view_projection = projection * view * model;
}

void Viewer::paintGL() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  program->use();
  glUniformMatrix4fv(matrix_id, 1, GL_FALSE, &model_view_projection[0][0]);

  std::vector<float> vertex_buffer_data(field.vertex_data().size() * 3);
  float max = -INFINITY;
  float min = INFINITY;
  for (auto i = 0; i < field.vertex_data().size(); ++i) {
    vertex_buffer_data[3 * i + 0] = field.vertex_data()[i].x();
    vertex_buffer_data[3 * i + 1] = field.vertex_data()[i].y();
    vertex_buffer_data[3 * i + 2] = field.values()[i];

    max = std::max(max, field.values()[i]);
    min = std::min(min, field.values()[i]);
  }
  vertex_buffer->set_data(vertex_buffer_data);

  std::vector<int> element_buffer_data(field.primitive_data().size() * 3);
  for (auto i = 0; i < field.primitive_data().size(); ++i) {
    for (auto v = 0; v < 3; ++v)
      element_buffer_data[3 * i + v] = field.primitive_data()[i][v];
  }
  element_buffer->set_data(element_buffer_data);

  std::vector<float> color_buffer_data(field.vertex_data().size() * 3);
  for (auto i = 0; i < field.vertex_data().size(); ++i) {
    color_buffer_data[3 * i] = (field.values()[i] - min) / (max - min);
    color_buffer_data[3 * i + 1] = 1.0f;
    color_buffer_data[3 * i + 2] = 0.0f;
  }
  color_buffer->set_data(color_buffer_data);

  (*vertex_array)
      .enable_attribute(0, *vertex_buffer)
      .enable_attribute(1, *color_buffer);
  element_buffer->draw();
  (*vertex_array).disable_attribute(1).disable_attribute(0);

  // if (render_volume_force) {
  //   glBegin(GL_LINES);
  //   for (const auto& pair : field.edge_data()) {
  //     if (pair.second == 1)
  //       glColor3f(1, 0, 0);
  //     else if (pair.second == -1)
  //       glColor3f(0, 0, 1);
  //     else
  //       glColor3f(0, 0, 0);

  //     glVertex3f(field.vertex_data()[pair.first[0]].x(),
  //                field.vertex_data()[pair.first[0]].y(),
  //                field.volume_force()[pair.first[0]]);
  //     glVertex3f(field.vertex_data()[pair.first[1]].x(),
  //                field.vertex_data()[pair.first[1]].y(),
  //                field.volume_force()[pair.first[1]]);
  //   }
  //   glEnd();
  // } else {
  //   glBegin(GL_LINES);
  //   for (const auto& pair : field.edge_data()) {
  //     if (pair.second == 1)
  //       glColor3f(1, 0, 0);
  //     else if (pair.second == -1)
  //       glColor3f(0, 0, 1);
  //     else
  //       glColor3f(0, 0, 0);

  //     constexpr float scale = 1.0f;

  //     glVertex3f(field.vertex_data()[pair.first[0]].x(),
  //                field.vertex_data()[pair.first[0]].y(),
  //                scale * field.values()[pair.first[0]]);
  //     glVertex3f(field.vertex_data()[pair.first[1]].x(),
  //                field.vertex_data()[pair.first[1]].y(),
  //                scale * field.values()[pair.first[1]]);
  //   }
  //   glEnd();
  // }

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

  // makeCurrent();
  // glLoadIdentity();
  // gluLookAt(position[0], position[1], position[2], world.origin()[0],
  //           world.origin()[1], world.origin()[2], world.basis_y()[0],
  //           world.basis_y()[1], world.basis_y()[2]);

  glm::mat4 projection = glm::perspective(
      glm::radians(45.0f), (float)width() / (float)height(), 0.1f, 100.0f);
  glm::mat4 view = glm::lookAt(
      glm::vec3(position[0], position[1], position[2]),
      glm::vec3(world.origin()[0], world.origin()[1], world.origin()[2]),
      glm::vec3(world.basis_y()[0], world.basis_y()[1], world.basis_y()[2]));
  glm::mat4 model = glm::mat4(1.0f);
  model_view_projection = projection * view * model;

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