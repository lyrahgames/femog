add_executable(main_test
  doctest_main.cc
  permutation_test.cc
  isometry_test.cc
  camera_test.cc
  fem_field_test.cc
)

target_link_libraries(main_test
  doctest::doctest
  Eigen3::Eigen
  femog::femog
)

add_test(permutation_test
  main_test --source-file=${CMAKE_CURRENT_SOURCE_DIR}/permutation_test.cc
)

add_test(isometry_test
  main_test --source-file=${CMAKE_CURRENT_SOURCE_DIR}/isometry_test.cc
)

add_test(camera_test
  main_test --source-file=${CMAKE_CURRENT_SOURCE_DIR}/camera_test.cc
)

add_test(fem_field_test
  main_test --source-file=${CMAKE_CURRENT_SOURCE_DIR}/fem_field_test.cc
)