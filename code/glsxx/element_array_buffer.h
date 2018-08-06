#ifndef ELEMENT_ARRAY_BUFFER_H_
#define ELEMENT_ARRAY_BUFFER_H_

#include <vector>

#include <GL/glew.h>

class Element_array_buffer {
 public:
  Element_array_buffer() : size_{0} { glGenBuffers(1, &id_); }
  ~Element_array_buffer() {}

  GLuint id() const { return id_; }

  Element_array_buffer& bind() {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id_);
    return *this;
  }

  Element_array_buffer& set_data(const std::vector<int>& data) {
    bind();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.size() * sizeof(int),
                 data.data(), GL_STATIC_DRAW);
    size_ = data.size();
    return *this;
  }

  Element_array_buffer& draw() {
    bind();
    glDrawElements(GL_TRIANGLES, size_, GL_UNSIGNED_INT, nullptr);
    return *this;
  }

 private:
  GLuint id_;
  size_t size_;
};

#endif  // ELEMENT_ARRAY_BUFFER_H_