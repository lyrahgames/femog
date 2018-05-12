#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <GL/glu.h>
#include <Eigen/Dense>

#include <QApplication>
#include <QMouseEvent>
#include <QOpenGLWidget>

class Viewer : public QOpenGLWidget {
 public:
  Viewer(QWidget* parent = nullptr) : QOpenGLWidget(parent) {
    setMouseTracking(true);
  }
  void load(const std::string& file_path) {
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
  }

 protected:
  void initializeGL() override {
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    // glLoadIdentity();
    // gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0);
    position = Eigen::Vector3f(0, 0, 10);
    compute_look_at();
  }

  void resizeGL(int width, int height) override {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, width, height);
    gluPerspective(45, static_cast<float>(width) / height, 0.1, 1000);
    glMatrixMode(GL_MODELVIEW);
  }

  void paintGL() override {
    // glLoadIdentity();
    // gluLookAt(position[0], position[1], position[2], 0, 0, 0, 0, 1, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBegin(GL_TRIANGLES);
    glColor3f(0, 0, 0);
    for (const auto& vertex : stl_data) glVertex3fv(vertex.data());
    glEnd();
  }

  void mousePressEvent(QMouseEvent* event) { press = true; }

  void mouseReleaseEvent(QMouseEvent* event) { press = false; }

  void mouseMoveEvent(QMouseEvent* event) {
    const int x_difference = event->x() - old_mouse_x;
    const int y_difference = event->y() - old_mouse_y;

    if (press) {
      constexpr float eye_altitude_max_abs = M_PI_2 - 0.0001f;
      eye_azimuth += x_difference * 0.01;
      eye_altitude += y_difference * 0.01;
      if (eye_altitude > eye_altitude_max_abs)
        eye_altitude = eye_altitude_max_abs;
      if (eye_altitude < -eye_altitude_max_abs)
        eye_altitude = -eye_altitude_max_abs;
    }

    old_mouse_x = event->x();
    old_mouse_y = event->y();
    compute_look_at();
  }

 private:
  bool press = false;
  int old_mouse_x = 0;
  int old_mouse_y = 0;
  std::vector<Eigen::Vector3f> stl_data;
  Eigen::Vector3f position;
  float eye_distance = 10.0f;
  float eye_azimuth = 0.0f;
  float eye_altitude = 0.0f;

  void compute_look_at() {
    position[0] = eye_distance * cosf(eye_azimuth) * cosf(eye_altitude);
    position[1] = eye_distance * sinf(eye_altitude);
    position[2] = eye_distance * sinf(eye_azimuth) * cosf(eye_altitude);

    makeCurrent();
    glLoadIdentity();
    gluLookAt(position[0], position[1], position[2], 0, 0, 0, 0, 1, 0);
    update();
  }
};

int main(int argc, char* argv[]) {
  if (2 != argc) {
    std::cout << "usage:" << std::endl
              << argv[0] << " <path to stl file>" << std::endl;
    return -1;
  }

  QApplication application(argc, argv);
  Viewer viewer;
  viewer.load(argv[1]);
  viewer.show();
  return application.exec();
}