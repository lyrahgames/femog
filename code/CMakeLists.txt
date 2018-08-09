cmake_minimum_required(VERSION 3.10)
project(fem_on_gpu VERSION 0.1.0 LANGUAGES CXX)

enable_testing()
find_package(doctest REQUIRED)

find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(GLEW REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Qt5Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)

add_library(femog
  camera.cc
  viewer.cc
  fem_field.cc
  fem_field_loader.cc
)
target_include_directories(femog PUBLIC .)
target_link_libraries(femog PUBLIC
  Eigen3::Eigen
  Qt5::Widgets
  OpenGL::GL
  OpenGL::GLU
  GLEW::GLEW
  glm
)
target_compile_features(femog PUBLIC cxx_std_17)
add_library(femog::femog ALIAS femog)

add_subdirectory(tests)

add_executable(viewer main.cc)
target_link_libraries(viewer PRIVATE femog::femog)