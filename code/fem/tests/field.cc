#include <doctest/doctest.h>

#include <Eigen/Dense>

#include <fem/domain.h>
#include <fem/field.h>

SCENARIO("The FEM field construction") {
  using namespace Femog::Fem;

  GIVEN("an FEM domain") {
    Domain<Eigen::Vector2f> domain;
    CHECK(domain.field_list().size() == 0);

    THEN("an FEM field can be constructed.") {
      Field<float> field{domain};
      CHECK(domain.field_list().size() == 1);
    }
    AND_THEN("with deletion the field will be unregistered.") {
      CHECK(domain.field_list().size() == 0);
    }

    WHEN("the domain is empty") {
      THEN("the constructed field contains no values.") {
        Field<float> field{domain};
        CHECK(field.values().size() == 0);
      }
    }

    WHEN("the domain contains some vertices") {
      domain << Eigen::Vector2f{0, 0} << Eigen::Vector2f{0, 1}
             << Eigen::Vector2f{1, 1} << Eigen::Vector2f{1, 0};

      THEN("the constructed field will contain the same amount of values.") {
        Field<float> field{domain};
        CHECK(field.values().size() == domain.vertex_data().size());
      }
    }
  }
}

SCENARIO("The FEM field and domain interaction") {
  using namespace Femog::Fem;

  GIVEN("a field based on an empty domain") {
    Domain<Eigen::Vector2f> domain;
    Field<float> field{domain};

    WHEN("vertices are added to the domain") {
      domain << Eigen::Vector2f{0, 0} << Eigen::Vector2f{0, 1}
             << Eigen::Vector2f{1, 1} << Eigen::Vector2f{1, 0};

      THEN("new values are added to the field") {
        CHECK(field.values().size() == 4);
      }
    }
  }

  GIVEN("a field based on a quad domain") {
    Domain<Eigen::Vector2f> domain;
    domain << Eigen::Vector2f{0, 0} << Eigen::Vector2f{0, 1}
           << Eigen::Vector2f{1, 1} << Eigen::Vector2f{1, 0}
           << Domain_base::Quad{0, 1, 2, 3};
    Field<float> field{domain};

    CHECK(field.values().size() == 4);
    CHECK(domain.vertex_data().size() == 4);
    CHECK(domain.primitive_data().size() == 2);

    WHEN("the domain is subdivided") {
      domain.subdivide();

      CHECK(domain.vertex_data().size() == 9);
      CHECK(domain.primitive_data().size() == 8);

      THEN("the same amount of values is appended to the field.") {
        CHECK(field.values().size() == domain.vertex_data().size());
      }

      WHEN("the domain is subdivided a second time") {
        domain.subdivide();

        CHECK(domain.vertex_data().size() == 25);
        CHECK(domain.primitive_data().size() == 32);

        THEN("again the same amount of values is appended to the field.") {
          CHECK(field.values().size() == domain.vertex_data().size());
        }
      }
    }
  }
}