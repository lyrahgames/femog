#version 330 core

in vec3 fragment_position;
in vec3 fragment_normal;
in vec3 fragment_color;

out vec3 color;

uniform vec3 light_position;

void main() {
  float specular_coeff = 0.3;
  float diffuse_coeff = 1.0;
  float ambient_coeff = 0.3;

  vec3 normal = normalize(fragment_normal);
  vec3 light_direction = normalize(light_position - fragment_position);
  float cos_diffuse = abs(dot(normal, light_direction));
  float cos_specular =
      clamp(dot(light_direction, reflect(-light_direction, normal)), 0, 1);

  color = fragment_color * (diffuse_coeff * cos_diffuse + ambient_coeff) +
          specular_coeff * pow(cos_specular, 10);
}