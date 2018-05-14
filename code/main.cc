#include <iostream>

#include <QApplication>

#include <viewer.h>

int main(int argc, char* argv[]) {
  if (2 != argc) {
    std::cout << "usage:" << std::endl
              << argv[0] << " <path to stl file>" << std::endl;
    return -1;
  }

  QApplication application(argc, argv);
  Femog::Viewer viewer;
  // viewer.load(argv[1]);
  viewer.generate();
  viewer.show();
  return application.exec();
}