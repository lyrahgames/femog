#ifndef ARRAY_BUFFER_H_
#define ARRAY_BUFFER_H_

#include <vector>

#include <GL/glew.h>

class Array_buffer {
 public:
  Array_buffer() { glGenBuffers(1, &id_); }
  ~Array_buffer() {}

  GLuint id() const { return id_; }

  const Array_buffer& bind() const {
    glBindBuffer(GL_ARRAY_BUFFER, id_);
    return *this;
  }

  Array_buffer& set_data(const std::vector<float>& data) {
    bind();
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(),
                 GL_STATIC_DRAW);
    return *this;
  }

 private:
  GLuint id_;
};

#endif  // ARRAY_BUFFER_H_