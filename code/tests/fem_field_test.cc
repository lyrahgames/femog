#include <doctest/doctest.h>

#include <fem_field.h>
#include <Eigen/Sparse>
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

SCENARIO("The FEM field can define Neumann and Dirichlet boundaries.") {
  using Femog::Fem_field;

  GIVEN("an FEM field with a domain built from primitives") {
    Fem_field field;
    field.add_vertex({0, 0});
    field.add_vertex({0, 1});
    field.add_vertex({1, 0});
    field.add_vertex({1, 1});
    field.add_quad({0, 1, 2, 3});

    WHEN("no boundary is specified") {
      THEN("every boundary is assumed to be a Dirichlet boundary.") {
        CHECK(field.edge_data().size() == 5);
        CHECK(field.boundary_size() == 4);
        CHECK(field.dirichlet_boundary_size() == field.boundary_size());
        CHECK(field.neumann_boundary_size() == 0);

        CHECK(field.is_dirichlet_boundary({0, 1}));
        CHECK(field.is_dirichlet_boundary({1, 2}));
        CHECK(field.is_dirichlet_boundary({2, 3}));
        CHECK(field.is_dirichlet_boundary({3, 0}));
        CHECK_FALSE(field.is_dirichlet_boundary({0, 2}));
        CHECK_FALSE(field.is_dirichlet_boundary({1, 3}));
      }
    }

    WHEN("an existing non-inner edge is specified as Neumann boundary") {
      field.set_neumann_boundary({0, 1});
      THEN("the edge will be marked as Neumann boundary.") {
        CHECK(field.is_neumann_boundary({0, 1}));
        CHECK(field.boundary_size() == 4);
        CHECK(field.dirichlet_boundary_size() == 3);
        CHECK(field.neumann_boundary_size() == 1);
      }
    }
    AND_WHEN("a new primitive is added and connected by this boundary") {
      field.add_vertex({-1, 1});
      field.add_primitive({0, 1, 4});
      THEN("the edge will be marked as an inner edge.") {
        CHECK_FALSE(field.is_neumann_boundary({0, 1}));
        CHECK_FALSE(field.is_dirichlet_boundary({0, 1}));

        CHECK(field.edge_data().size() == 7);
        CHECK(field.boundary_size() == 5);
        CHECK(field.dirichlet_boundary_size() == field.boundary_size());
        CHECK(field.neumann_boundary_size() == 0);
      }
    }

    WHEN(
        "an inner or non-existing edge is specified as Neumann or Dirichlet "
        "boundary") {
      THEN(
          "the function throws an 'invalid_argument' or 'out_of_range' "
          "exception.") {
        CHECK_THROWS_AS(field.set_neumann_boundary({1, 3}), std::out_of_range);
        CHECK_THROWS_AS(field.set_neumann_boundary({0, 2}),
                        std::invalid_argument);
        CHECK_THROWS_AS(field.set_dirichlet_boundary({1, 3}),
                        std::out_of_range);
        CHECK_THROWS_AS(field.set_dirichlet_boundary({0, 2}),
                        std::invalid_argument);
      }
    }
  }
}

TEST_CASE("The FEM Field can construct a stiffness matrix.") {
  using Femog::Fem_field;
  Fem_field field;
  field.add_vertex({0, 0});
  field.add_vertex({0, 1});
  field.add_vertex({1, 0});
  field.add_vertex({1, 1});
  field.add_quad({0, 1, 2, 3});
  field.add_vertex({-1, 1});
  field.add_primitive({0, 1, 4});
  field.volume_force()[3] = 1.0;

  std::vector<Eigen::Triplet<float>> triplets;
  Eigen::VectorXf rhs(field.vertex_data().size());

  for (const auto& primitive : field.primitive_data()) {
    Fem_field::vertex_type edge[3];

    for (auto i = 0; i < 3; ++i) {
      edge[i] = field.vertex_data()[primitive[(i + 1) % 3]] -
                field.vertex_data()[primitive[i]];
    }

    const float area =
        0.5f * std::abs(-edge[0].x() * edge[2].y() + edge[0].y() * edge[2].x());
    const float inverse_area_4 = 0.25f / area;

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const float value =
            inverse_area_4 * edge[(i + 1) % 3].dot(edge[(j + 1) % 3]);
        triplets.push_back({primitive[i], primitive[j], value});
      }

      const float mean_force = (field.volume_force()[primitive[0]] +
                                field.volume_force()[primitive[1]] +
                                field.volume_force()[primitive[2]]) /
                               3.0f;
      rhs[primitive[i]] += (area * mean_force);
    }
  }

  Eigen::SparseMatrix<float> matrix(field.vertex_data().size(),
                                    field.vertex_data().size());

  matrix.setFromTriplets(triplets.begin(), triplets.end());

  MESSAGE("Matrix:\n" << matrix);
  MESSAGE("force:\n" << rhs);
}