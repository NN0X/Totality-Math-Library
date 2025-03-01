#ifndef VECTOR_CPU_HPP
#define VECTOR_CPU_HPP

#include "tensor_cpu.hpp"

// TODO: implement vector operations:
// - vector dot product
// - vector cross product
// - vector norm
// - vector normalization
// - vector angle

// TODO: optimize functions with multitheading and other

template <typename T, uint64_t S>
class Vector : public Tensor<T, 1, S>
{
public:
        Vector() : Tensor<T, 1, S>() {}
        Vector(const T &value) : Tensor<T, 1, S>(value) {}
        Vector(const HeapArray<T, S> &vector) : Tensor<T, 1, S>(vector) {}

        ~Vector() = default;
};

#endif // VECTOR_CPU_HPP
