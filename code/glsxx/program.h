#ifndef PROGAM_H_
#define PROGAM_H_

#include <iostream>
#include <stdexcept>

#include "shader.h"

class Program {
 public:
  Program() { id_ = glCreateProgram(); }
  ~Program() { glDeleteProgram(id_); }
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;
  Program(Program&& in) : id_(in.id_) { in.id_ = 0; }
  Program& operator=(Program&& in) {
    id_ = in.id_;
    in.id_ = 0;
    return *this;
  }

  GLuint id() const { return id_; }

  template <int Shader_type>
  Program& attach_shader(const Shader<Shader_type>& shader) {
    glAttachShader(id_, shader.id());
    return *this;
  }

  template <int Shader_type>
  Program& detach_shader(const Shader<Shader_type>& shader) {
    glDetachShader(id_, shader.id());
    return *this;
  }

  Program& validate() {
    glValidateProgram(id_);
    int success;
    glGetProgramiv(id_, GL_VALIDATE_STATUS, &success);
    if (!success) {
      const std::string log = info_log();
      throw std::runtime_error(
          "GLSL shader program could not be validated!\nInfo Log:\n" + log);
    }
    return *this;
  }

  Program& link() {
    glLinkProgram(id_);
    int success;
    glGetProgramiv(id_, GL_LINK_STATUS, &success);
    if (!success) {
      const std::string log = info_log();
      throw std::runtime_error(
          "GLSL shader program could not be linked!\nInfo Log:\n" + log);
    }
    validate();
    return *this;
  }

  Program& use() {
    glUseProgram(id_);
    return *this;
  }

 private:
  GLuint id_;

  std::string info_log() const {
    int info_log_length;
    glGetProgramiv(id_, GL_INFO_LOG_LENGTH, &info_log_length);
    std::string result(info_log_length + 1, '\0');
    glGetProgramInfoLog(id_, info_log_length, nullptr, result.data());
    return result;
  }
};

#endif  // PROGAM_H_