#version 330 core

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_color;
layout(location = 2) in vec3 vertex_normal;

out vec3 fragment_position;
out vec3 fragment_normal;
out vec3 fragment_color;

uniform mat4 model_view_projection;

void main() {
  gl_Position = model_view_projection * vec4(vertex_position, 1);
  fragment_position = vertex_position;
  fragment_normal = vertex_normal;
  fragment_color = vertex_color;
}