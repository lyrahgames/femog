#include <iostream>
#include <sstream>

#include <QApplication>
#include <QSurfaceFormat>

#include <viewer.h>

#include <fem/io.h>

int main(int argc, char* argv[]) {
  if (2 > argc || 3 < argc) {
    std::cout << "usage:" << std::endl
              << argv[0] << " <path to stl file> [<subdivision count>]"
              << std::endl;
    return -1;
  }

  auto domain = Femog::Fem::domain_from_stream(std::cin);
  std::cout << "vertex count = " << domain.vertex_data().size() << std::endl;

  std::string file_path{argv[1]};
  int subdivision_count{0};
  if (argc == 3) std::stringstream{argv[2]} >> subdivision_count;

  QSurfaceFormat surface_format;
  surface_format.setVersion(3, 3);
  surface_format.setSamples(10);
  surface_format.setProfile(QSurfaceFormat::CoreProfile);
  QSurfaceFormat::setDefaultFormat(surface_format);

  QApplication application(argc, argv);
  Femog::Viewer viewer;
  viewer.load(file_path);
  viewer.subdivide(subdivision_count);
  viewer.set_analytic_volume_force();
  viewer.solve();
  viewer.show();
  return application.exec();
}