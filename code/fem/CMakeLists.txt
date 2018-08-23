cmake_minimum_required(VERSION 3.10)
project(femog.fem VERSION 0.1.0 LANGUAGES CXX CUDA)

include(ExternalProject)
find_package(Git REQUIRED)
ExternalProject_Add(cusp-project
  PREFIX ${CMAKE_BINARY_DIR}/cusp
  # DOWNLOAD_DIR cusp
  GIT_REPOSITORY https://github.com/cusplibrary/cusplibrary.git
  TIMEOUT 10
  # UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
)
ExternalProject_Get_Property(cusp-project download_dir)
message(${download_dir})
add_library(cusp INTERFACE)
target_include_directories(cusp INTERFACE ${download_dir}/cusp-project)
target_compile_features(cusp INTERFACE cxx_std_14)
add_dependencies(cusp cusp-project)
add_library(cusp::cusp ALIAS cusp)

find_package(Eigen3 REQUIRED)

add_library(femog.fem.gpu gpu_solver.cu)
target_include_directories(femog.fem.gpu PUBLIC .)
target_link_libraries(femog.fem.gpu PRIVATE cusp::cusp)
target_link_libraries(femog.fem.gpu PUBLIC Eigen3::Eigen)
set_property(TARGET femog.fem.gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)


add_library(femog.fem domain_field_base.cc io.cc system.cc)
target_include_directories(femog.fem PUBLIC ..)
target_link_libraries(femog.fem PUBLIC Eigen3::Eigen femog.fem.gpu)
target_compile_features(femog.fem PUBLIC cxx_std_14)
# set_property(TARGET femog.fem PROPERTY CUDA_STANDARD 11)
# target_compile_options(femog.fem PUBLIC -fPIC)
add_library(Femog::Fem ALIAS femog.fem)

enable_testing()
add_subdirectory(tests)