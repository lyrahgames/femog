#ifndef SHADER_H_
#define SHADER_H_

#include <iostream>
#include <stdexcept>
#include <string>

#include <GL/glew.h>

template <int Shader_type>
class Shader {
 public:
  Shader(std::istream& input) {
    const std::string source_code{std::istreambuf_iterator<char>{input}, {}};
    const char* source_code_pointer = source_code.c_str();

    id_ = glCreateShader(Shader_type);

    glShaderSource(id_, 1, &source_code_pointer, nullptr);
    glCompileShader(id_);

    GLint success;
    glGetShaderiv(id_, GL_COMPILE_STATUS, &success);

    if (!success) {
      int info_log_length;
      glGetShaderiv(id_, GL_INFO_LOG_LENGTH, &info_log_length);
      std::string info_log(info_log_length + 1, '\0');
      glGetShaderInfoLog(id_, info_log_length, nullptr, info_log.data());
      throw std::runtime_error("Shader of type '" +
                               ((Shader_type == GL_VERTEX_SHADER)
                                    ? (std::string{"GL_VERTEX_SHADER"})
                                    : (std::string{"GL_FRAGMENT_SHADER"})) +
                               "' could not be constructed!\nInfo Log:\n" +
                               info_log);
    }
  }
  virtual ~Shader() { glDeleteShader(id_); }

  GLuint id() const { return id_; }

 private:
  GLuint id_;
};

class Vertex_shader : public Shader<GL_VERTEX_SHADER> {
 public:
  Vertex_shader(std::istream& input) : Shader<GL_VERTEX_SHADER>(input) {}
};

class Fragment_shader : public Shader<GL_FRAGMENT_SHADER> {
 public:
  Fragment_shader(std::istream& input) : Shader<GL_FRAGMENT_SHADER>(input) {}
};

#endif  // SHADER_H_