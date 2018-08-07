#ifndef FEMOG_VIEWER_H_
#define FEMOG_VIEWER_H_

#include <memory>
#include <string>
#include <vector>

#include <GL/glew.h>
#include <GL/glu.h>

#include <glm/gtc/matrix_transform.hpp>

#include <Eigen/Dense>

#include <QKeyEvent>
#include <QMouseEvent>
#include <QOpenGLWidget>
#include <QWheelEvent>

#include <camera.h>
#include <fem_field.h>
#include <isometry.h>

#include "glsxx/element_array_buffer.h"
#include "glsxx/program.h"
#include "glsxx/vertex_array.h"

namespace Femog {

class Viewer : public QOpenGLWidget {
 public:
  Viewer(QWidget* parent = nullptr);

  void load(const std::string& file_path);
  void subdivide(int count);
  void set_analytic_volume_force();
  void compute_automatic_view();
  void solve();

 protected:
  void initializeGL() override;
  void resizeGL(int width, int height) override;
  void paintGL() override;
  void mousePressEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void wheelEvent(QWheelEvent* event) override;
  void keyPressEvent(QKeyEvent* event) override;
  void keyReleaseEvent(QKeyEvent* event) override;

 private:
  glm::mat4 model_view_projection;
  GLuint matrix_id;
  GLuint light_id;

  std::unique_ptr<Vertex_array> vertex_array;
  std::unique_ptr<Program> program;
  std::unique_ptr<Array_buffer> vertex_buffer;
  std::unique_ptr<Array_buffer> normal_buffer;
  std::unique_ptr<Array_buffer> color_buffer;
  std::unique_ptr<Element_array_buffer> element_buffer;

  int old_mouse_x = 0;
  int old_mouse_y = 0;
  float eye_distance = 10.0f;
  float eye_azimuth = 0.0f;
  float eye_altitude = 0.0f;
  Camera camera;
  Isometry world;
  Fem_field field;
  Eigen::Vector3f bounding_box_min;
  Eigen::Vector3f bounding_box_max;
  bool render_volume_force = false;

  void compute_look_at();
  void compute_bounding_box();
};

}  // namespace Femog

#endif  // FEMOG_VIEWER_H_