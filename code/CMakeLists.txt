cmake_minimum_required(VERSION 3.10)
project(fem_on_gpu VERSION 0.1.0 LANGUAGES CXX)

enable_testing()
find_package(doctest REQUIRED)

find_package(OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(viewer viewer.cc)
target_link_libraries(viewer PRIVATE
  Qt5::Widgets
  Eigen3::Eigen
  OpenGL::GLU
)

add_library(femog INTERFACE)
target_include_directories(femog INTERFACE .)
target_link_libraries(femog INTERFACE Eigen3::Eigen)
add_library(femog::femog ALIAS femog)

add_subdirectory(tests)