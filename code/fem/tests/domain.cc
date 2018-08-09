#include <doctest/doctest.h>

#include <fem/domain.h>
#include <Eigen/Dense>

TEST_CASE("The FEM domain") {
  using Domain2f = Femog::Fem::Domain<Eigen::Vector2f>;

  Domain2f domain{};

  CHECK(domain.vertex_data().size() == 0);

  SUBCASE(
      "can push back vertices to the domain with standard, chain and stream "
      "notation.") {
    // domain.add_vertex(Domain2f::Vertex{1, 2});
    domain << Domain2f::Vertex{1, 2};

    CHECK(domain.vertex_data().size() == 1);
    CHECK(domain.vertex_data()[0] == Domain2f::Vertex{1, 2});

    domain.add_vertex({0, 0}).add_vertex({1, 1});

    CHECK(domain.vertex_data().size() == 3);
    CHECK(domain.vertex_data()[1] == Domain2f::Vertex{0, 0});
    CHECK(domain.vertex_data()[2] == Domain2f::Vertex{1, 1});
  }

  SUBCASE("can add primitives to the domain.") {
    using Vertex = Domain2f::Vertex;
    using Primitive = Domain2f::Primitive;
    using Quad = Domain2f::Quad;
    using Edge = Domain2f::Edge;
    // domain.add_vertex({0, 0}).add_vertex({0, 1}).add_vertex({1, 0});
    domain << Vertex{0, 0} << Vertex{0, 1} << Vertex{1, 0};

    SUBCASE("The primitive is added if the referenced vertices do exist.") {
      // domain.add_primitive({0, 1, 2});
      domain << Primitive{0, 1, 2};

      CHECK(domain.vertex_data().size() == 3);
      CHECK(domain.primitive_data().size() == 1);
      CHECK(domain.primitive_data()[0] == Domain2f::Primitive{0, 1, 2});
    }

    SUBCASE("An exception is thrown if the referenced vertices do not exist.") {
      CHECK_THROWS_AS(domain.add_primitive({1, 2, 3}), std::invalid_argument);
      CHECK_THROWS_AS(domain.add_primitive({4, 2, 3}), std::invalid_argument);
      CHECK_THROWS_AS(domain.add_primitive({4, 2, 0}), std::invalid_argument);
      CHECK(domain.vertex_data().size() == 3);
      CHECK(domain.primitive_data().size() == 0);
    }

    SUBCASE(
        "An exception is thrown if two or more referenced vertices are the "
        "same.") {
      CHECK_THROWS_AS(domain.add_primitive({1, 1, 2}), std::invalid_argument);
      CHECK_THROWS_AS(domain.add_primitive({0, 1, 0}), std::invalid_argument);
    }

    SUBCASE("Primitives will be the same if indices are permutated.") {
      CHECK(Primitive{0, 1, 2} == Primitive{0, 2, 1});
      CHECK(Primitive{0, 1, 2} == Primitive{0, 2, 1});
      CHECK(Primitive{0, 1, 2} == Primitive{1, 0, 2});
      CHECK(Primitive{0, 1, 2} == Primitive{1, 2, 0});
      CHECK(Primitive{0, 1, 2} == Primitive{2, 0, 1});
      CHECK(Primitive{0, 1, 2} == Primitive{2, 1, 0});
    }

    SUBCASE(
        "An exception is thrown if the same primitive is added a second "
        "time.") {
      CHECK(domain.vertex_data().size() == 3);
      CHECK(domain.primitive_data().size() == 0);
      CHECK(domain.primitive_map().size() == 0);
      CHECK(domain.edge_map().size() == 0);

      domain << Primitive{0, 1, 2};

      CHECK(domain.primitive_data().size() == 1);
      CHECK(domain.primitive_map().size() == 1);
      CHECK(domain.primitive_map().at({0, 1, 2}).insertions == 1);
      CHECK(domain.edge_map().size() == 3);
      CHECK(domain.edge_map().at(Edge{0, 1}).insertions == 1);
      CHECK(domain.edge_map().at(Edge{0, 2}).insertions == 1);
      CHECK(domain.edge_map().at(Edge{1, 2}).insertions == 1);

      // domain << Primitive{0, 1, 2};
      CHECK_THROWS_AS(domain.add_primitive({0, 1, 2}), std::invalid_argument);

      CHECK(domain.primitive_data().size() == 1);
      CHECK(domain.primitive_map().size() == 1);
      CHECK(domain.primitive_map().at({0, 1, 2}).insertions == 1);
      CHECK(domain.edge_map().size() == 3);
      CHECK(domain.edge_map().at(Edge{0, 1}).insertions == 1);
      CHECK(domain.edge_map().at(Edge{0, 2}).insertions == 1);
      CHECK(domain.edge_map().at(Edge{1, 2}).insertions == 1);
    }

    SUBCASE(
        "An exception is thrown if an edge of the given primitive is already "
        "connecting two primitives.") {
      domain << Vertex{1, 1} << Quad{0, 1, 3, 2} << Vertex{0.5, 2};

      CHECK(domain.vertex_data().size() == 5);
      CHECK(domain.edge_map().size() == 5);
      CHECK(domain.primitive_data().size() == 2);

      CHECK_THROWS_AS(domain.add_primitive({0, 3, 4}), std::invalid_argument);
    }

    SUBCASE("Appending a quad will result in adding two primitives.") {
      // domain.add_vertex({1, 1});
      domain << Vertex{1, 1};

      CHECK(domain.vertex_data().size() == 4);

      // domain.add_quad({0, 1, 2, 3});
      domain << Quad{0, 1, 2, 3};

      CHECK(domain.vertex_data().size() == 4);
      CHECK(domain.primitive_data().size() == 2);
      CHECK(domain.primitive_data()[0] == Domain2f::Primitive{0, 1, 2});
      CHECK(domain.primitive_data()[1] == Domain2f::Primitive{0, 2, 3});

      SUBCASE(
          "No primitives will be added if there are one or more wrong "
          "references to vertices.") {
        CHECK_THROWS_AS(domain.add_quad({1, 2, 3, 4}), std::invalid_argument);
        CHECK(domain.vertex_data().size() == 4);
        CHECK(domain.primitive_data().size() == 2);
      }
    }
  }

  SUBCASE("will add edges for every primitive if they do not exist already.") {
    // domain.add_vertex({0, 0}).add_vertex({0, 1}).add_vertex({1,
    // 0}).add_vertex( {1, 1});
    domain << Domain2f::Vertex{0, 0} << Domain2f::Vertex{0, 1}
           << Domain2f::Vertex{1, 0} << Domain2f::Vertex{1, 1};

    CHECK(domain.edge_map().size() == 0);

    // domain.add_primitive({0, 1, 2});
    domain << Domain2f::Primitive{0, 1, 2};

    CHECK(domain.edge_map().size() == 3);

    for (const auto& pair : domain.edge_map())
      CHECK(pair.second.insertions == 1);

    domain << Domain2f::Primitive{0, 2, 3};

    CHECK(domain.edge_map().size() == 5);
    CHECK(domain.edge_map().at({0, 2}).insertions == 2);
    CHECK(domain.edge_map().at({0, 1}).insertions == 1);
    CHECK(domain.edge_map().at({2, 1}).insertions == 1);
    CHECK(domain.edge_map().at({2, 3}).insertions == 1);
    CHECK(domain.edge_map().at({0, 3}).insertions == 1);

    // domain.add_vertex({2, 0}).add_vertex({2, 1}).add_quad({2, 4, 5, 3});
    domain << Domain2f::Vertex{2, 0} << Domain2f::Vertex{2, 1}
           << Domain2f::Quad{2, 4, 5, 3};

    CHECK(domain.edge_map().size() == 9);
    CHECK(domain.edge_map().at({2, 3}).insertions == 2);
    CHECK(domain.edge_map().at({2, 4}).insertions == 1);
    CHECK(domain.edge_map().at({2, 5}).insertions == 2);
    CHECK(domain.edge_map().at({5, 3}).insertions == 1);
    CHECK(domain.edge_map().at({4, 5}).insertions == 1);
  }

  SUBCASE("can be subdivided.") {
    // domain.add_vertex({0, 0})
    //     .add_vertex({0, 1})
    //     .add_vertex({1, 0})
    //     .add_vertex({1, 1})
    //     .add_quad({0, 1, 2, 3})
    //     .subdivide();

    domain << Domain2f::Vertex{0, 0} << Domain2f::Vertex{0, 1}
           << Domain2f::Vertex{1, 0} << Domain2f::Vertex{1, 1}
           << Domain2f::Quad{0, 1, 2, 3};

    CHECK(domain.vertex_data().size() == 4);
    CHECK(domain.primitive_data().size() == 2);
    CHECK(domain.edge_map().size() == 5);

    domain.subdivide();

    CHECK(domain.vertex_data().size() == 9);
    CHECK(domain.primitive_data().size() == 8);
    CHECK(domain.edge_map().size() == 16);
  }
}