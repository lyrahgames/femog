#include <doctest/doctest.h>

#include <fem_field.h>
#include <stdexcept>

TEST_CASE("The FEM field") {
  using Femog::Fem_field;

  Fem_field field{};
  CHECK(field.vertex_data().size() == 0);
  CHECK(field.primitive_data().size() == 0);
  CHECK(field.values().size() == 0);

  SUBCASE("can add vertices to the domain.") {
    field.add_vertex(Fem_field::vertex_type{1, 2});

    CHECK(field.vertex_data().size() == 1);
    CHECK(field.values().size() == 1);
    CHECK(field.vertex_data()[0] == Fem_field::vertex_type{1, 2});

    SUBCASE("The size of the value vector is the same as the vertex count.") {
      field.add_vertex(Fem_field::vertex_type{2, 3});
      CHECK(field.vertex_data().size() == 2);
      CHECK(field.values().size() == field.vertex_data().size());
    }
  }

  SUBCASE("can add primitives to the domain.") {
    field.add_vertex(Fem_field::vertex_type{0, 0});
    field.add_vertex(Fem_field::vertex_type{0, 1});
    field.add_vertex(Fem_field::vertex_type{1, 0});

    SUBCASE("The primitive is added if the referenced vertices do exist.") {
      field.add_primitive(Fem_field::primitive_type{0, 1, 2});

      CHECK(field.vertex_data().size() == 3);
      CHECK(field.primitive_data().size() == 1);
      CHECK(field.primitive_data()[0] == Fem_field::primitive_type{0, 1, 2});
    }

    SUBCASE("An exception is thrown if the referenced vertices do not exist.") {
      CHECK_THROWS_AS(field.add_primitive(Fem_field::primitive_type{1, 2, 3}),
                      std::invalid_argument);
      CHECK(field.vertex_data().size() == 3);
      CHECK(field.primitive_data().size() == 0);
    }

    SUBCASE("Appending a quad will result in adding two primitives.") {
      field.add_vertex(Fem_field::vertex_type{1, 1});
      CHECK(field.vertex_data().size() == 4);

      field.add_quad(Fem_field::quad_type{0, 1, 2, 3});
      CHECK(field.vertex_data().size() == 4);
      CHECK(field.primitive_data().size() == 2);
      CHECK(field.primitive_data()[0] == Fem_field::primitive_type{0, 1, 2});
      CHECK(field.primitive_data()[1] == Fem_field::primitive_type{0, 2, 3});

      SUBCASE(
          "No primitives will be added if there are one or more wrong "
          "references to vertices.") {
        CHECK_THROWS_AS(field.add_quad(Fem_field::quad_type{1, 2, 3, 4}),
                        std::invalid_argument);
        CHECK(field.vertex_data().size() == 4);
        CHECK(field.primitive_data().size() == 2);
      }
    }
  }

  SUBCASE("will add edges for every primitive if they do not exist already.") {
    field.add_vertex(Fem_field::vertex_type{0, 0});
    field.add_vertex(Fem_field::vertex_type{0, 1});
    field.add_vertex(Fem_field::vertex_type{1, 0});
    field.add_vertex(Fem_field::vertex_type{1, 1});
    CHECK(field.edge_data().size() == 0);

    field.add_primitive(Fem_field::primitive_type{0, 1, 2});
    CHECK(field.edge_data().size() == 3);
    for (const auto& pair : field.edge_data()) CHECK(pair.second == 1);

    field.add_primitive(Fem_field::primitive_type{0, 2, 3});
    CHECK(field.edge_data().size() == 5);
    CHECK(field.edge_data().at({0, 2}) == 2);
    CHECK(field.edge_data().at({0, 1}) == 1);
    CHECK(field.edge_data().at({2, 1}) == 1);
    CHECK(field.edge_data().at({2, 3}) == 1);
    CHECK(field.edge_data().at({0, 3}) == 1);

    field.add_vertex(Fem_field::vertex_type{2, 0});
    field.add_vertex(Fem_field::vertex_type{2, 1});
    field.add_quad(Fem_field::quad_type{2, 4, 5, 3});
    CHECK(field.edge_data().size() == 9);
    CHECK(field.edge_data().at({2, 3}) == 2);
    CHECK(field.edge_data().at({2, 4}) == 1);
    CHECK(field.edge_data().at({2, 5}) == 2);
    CHECK(field.edge_data().at({5, 3}) == 1);
    CHECK(field.edge_data().at({4, 5}) == 1);
  }

  SUBCASE("can be subdivided.") {
    field.add_vertex(Fem_field::vertex_type{0, 0});
    field.add_vertex(Fem_field::vertex_type{0, 1});
    field.add_vertex(Fem_field::vertex_type{1, 0});
    field.add_vertex(Fem_field::vertex_type{1, 1});
    field.add_quad(Fem_field::quad_type{0, 1, 2, 3});
    field.subdivide();

    CHECK(field.vertex_data().size() == 9);
    CHECK(field.values().size() == 9);
    CHECK(field.primitive_data().size() == 8);
    CHECK(field.edge_data().size() == 16);
  }
}