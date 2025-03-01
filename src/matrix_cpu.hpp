#ifndef MATRIX_CPU_HPP
#define MATRIX_CPU_HPP

#include "tensor_cpu.hpp"

// TODO: implement matrix operations:
// - matrix determinant
// - matrix inverse

// TODO: determin at which size to use Strassen algorithm for matrix multiplication
// TODO: implement Strassen algorithm for matrix multiplication

// TODO: optimize functions with multitheading and other

template <typename T, uint64_t R, uint64_t C>
class Matrix : public Tensor<T, 2, R * C>
{
public:
        Matrix() : Tensor<T, 2, R * C>() {}
        Matrix(const T &value) : Tensor<T, 2, R * C>(value) {}
        Matrix(const HeapArray<T, R * C> &matrix) : Tensor<T, 2, R * C>(matrix) {}

        ~Matrix() = default;

        template <uint64_t R2, uint64_t C2>
        Matrix<T, R, C2> operator*(const Matrix<T, R2, C2> &matrix) const
        {
                static_assert(C == R2, "Matrix multiplication is only possible if the number of columns of the first matrix is equal to the number of rows of the second matrix.");
                Matrix<T, R, C2> result;
                for (uint64_t i = 0; i < R; i++)
                {
                        for (uint64_t j = 0; j < C2; j++)
                        {
                                T sum = 0;
                                for (uint64_t k = 0; k < C; k++)
                                {
                                        sum += (*this)(i, k) * matrix(k, j);
                                }
                                result(i, j) = sum;
                        }
                }
                return result;
        }

        Matrix<T, R, C> operator*=(const Matrix<T, R, C> &matrix)
        {
                *this = *this * matrix;
                return *this;
        }

        Matrix<T, C, R> transpose() const
        {
                Matrix<T, C, R> result;
                for (uint64_t i = 0; i < R; i++)
                {
                        for (uint64_t j = 0; j < C; j++)
                        {
                                result(j, i) = (*this)(i, j);
                        }
                }
                return result;
        }
};

#endif // MATRIX_CPU_HPP
