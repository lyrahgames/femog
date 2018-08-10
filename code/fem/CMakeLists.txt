cmake_minimum_required(VERSION 3.10)
project(femog.fem VERSION 0.1.0 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)

add_library(femog.fem domain_field_base.cc io.cc system.cc)
target_include_directories(femog.fem PUBLIC ..)
target_link_libraries(femog.fem PUBLIC Eigen3::Eigen)
target_compile_features(femog.fem PUBLIC cxx_std_17)
add_library(Femog::Fem ALIAS femog.fem)

enable_testing()
add_subdirectory(tests)