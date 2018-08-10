find_package(doctest REQUIRED)

add_executable(fem_test
  doctest_main.cc
  domain.cc
  field.cc
  io.cc
)

target_link_libraries(fem_test
  doctest::doctest
  Femog::Fem
)

add_test(domain_test
  fem_test --source-file=${CMAKE_CURRENT_SOURCE_DIR}/domain.cc
)

add_test(field_test
  fem_test --source-file=${CMAKE_CURRENT_SOURCE_DIR}/field.cc
)

add_test(io_test
  fem_test --source-file=${CMAKE_CURRENT_SOURCE_DIR}/io.cc
)