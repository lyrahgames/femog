#ifndef FEMOG_VIEWER_H_
#define FEMOG_VIEWER_H_

#include <string>
#include <vector>

#include <GL/glu.h>
#include <Eigen/Dense>

#include <QKeyEvent>
#include <QMouseEvent>
#include <QOpenGLWidget>
#include <QWheelEvent>

#include <camera.h>
#include <isometry.h>

namespace Femog {

class Viewer : public QOpenGLWidget {
 public:
  Viewer(QWidget* parent = nullptr);

  void load(const std::string& file_path);

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
  int old_mouse_x = 0;
  int old_mouse_y = 0;
  float eye_distance = 10.0f;
  float eye_azimuth = 0.0f;
  float eye_altitude = 0.0f;
  Camera camera;
  Isometry world;
  std::vector<Eigen::Vector3f> stl_data;
  Eigen::Vector3f bounding_box_min;
  Eigen::Vector3f bounding_box_max;

  void compute_look_at();
  void compute_bounding_box();
};

}  // namespace Femog

#endif  // FEMOG_VIEWER_H_