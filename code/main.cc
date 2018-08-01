#include <iostream>
#include <sstream>

#include <QApplication>

#include <viewer.h>

int main(int argc, char* argv[]) {
  if (2 > argc || 3 < argc) {
    std::cout << "usage:" << std::endl
              << argv[0] << " <path to stl file> [<subdivision count>]"
              << std::endl;
    return -1;
  }

  std::string file_path{argv[1]};
  int subdivision_count{0};
  if (argc == 3) std::stringstream{argv[2]} >> subdivision_count;

  QApplication application(argc, argv);
  Femog::Viewer viewer;
  viewer.load(file_path);
  viewer.subdivide(subdivision_count);
  viewer.set_analytic_volume_force();
  viewer.solve();
  viewer.show();
  return application.exec();
}