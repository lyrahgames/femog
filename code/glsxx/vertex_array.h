#ifndef VERTEX_ARRAY_H_
#define VERTEX_ARRAY_H_

#include <GL/glew.h>

#include "array_buffer.h"

class Vertex_array {
 public:
  Vertex_array() { glGenVertexArrays(1, &id_); }
  ~Vertex_array() { glDeleteVertexArrays(1, &id_); }

  GLuint id() const { return id_; }

  Vertex_array& bind() {
    glBindVertexArray(id_);
    return *this;
  }

  Vertex_array& enable_attribute(GLuint index, const Array_buffer& buffer) {
    bind();
    glEnableVertexAttribArray(index);
    buffer.bind();
    glVertexAttribPointer(index, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    return *this;
  }

  Vertex_array& disable_attribute(GLuint index) {
    bind();
    glDisableVertexAttribArray(index);
    return *this;
  }

 private:
  GLuint id_;
};

#endif  // VERTEX_ARRAY_H_