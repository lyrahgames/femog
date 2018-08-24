#ifndef FEMOG_FEM_SYSTEM_H_
#define FEMOG_FEM_SYSTEM_H_

#include <Eigen/Dense>

#include "domain.h"
#include "field.h"

namespace Femog::Fem {

class System {
 public:
  System() : domain_{}, wave_{domain_}, evolution_{domain_} {}

  auto& domain() { return domain_; }
  auto& wave() { return wave_; }
  auto& evolution() { return evolution_; }
  auto& dt() { return dt_; }

  System& solve();
  System& solve_custom();
  System& gpu_solve();
  System& gpu_wave_solve();

 private:
  Domain<Eigen::Vector2f> domain_;
  Field<float> wave_;
  Field<float> evolution_;
  float dt_;
};

class System3 {
 public:
  System3() : domain_{}, wave_{domain_}, evolution_{domain_} {}

  auto& domain() { return domain_; }
  auto& wave() { return wave_; }
  auto& evolution() { return evolution_; }
  auto& dt() { return dt_; }

  System3& solve();

 private:
  Domain<Eigen::Vector3f> domain_;
  Field<float> wave_;
  Field<float> evolution_;
  float dt_;
};

}  // namespace Femog::Fem

#endif  // FEMOG_FEM_SYSTEM_H_