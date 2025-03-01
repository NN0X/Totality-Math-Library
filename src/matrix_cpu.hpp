#ifndef MATRIX_CPU_HPP
#define MATRIX_CPU_HPP

#include "tensor_cpu.hpp"

// TODO: determine at which size to use Strassen algorithm for matrix multiplication
// TODO: implement Strassen algorithm for matrix multiplication

// TODO: optimize functions with multitheading and other


#define BIG_MATRIX_SIZE 500000000

template <typename T, uint64_t R, uint64_t C, bool = (R * C < BIG_MATRIX_SIZE)>
class Matrix;

template <typename T, uint64_t R, uint64_t C>
class Matrix<T, R, C, true> : public Tensor<T, 2, R * C>
{
public:
        Matrix() : Tensor<T, 2, R * C>({R, C}) {}
        Matrix(const T &value) : Tensor<T, 2, R * C>({R, C}, value) {}
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

        Matrix<T, R, C>& operator*=(const Matrix<T, R, C> &matrix)
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

        T determinant() const
        {
                static_assert(R == C, "Determinant is only possible for square matrices.");
                if constexpr (R == 1)
                {
                        return (*this)(0, 0);
                }
                else if constexpr (R == 2)
                {
                        return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
                }
                else
                {
                        T det = 0;
                        for (uint64_t i = 0; i < C; i++)
                        {
                                Matrix<T, R - 1, C - 1> submatrix;
                                for (uint64_t j = 1; j < R; j++)
                                {
                                        for (uint64_t k = 0; k < C; k++)
                                        {
                                                if (k < i)
                                                {
                                                        submatrix(j - 1, k) = (*this)(j, k);
                                                }
                                                else if (k > i)
                                                {
                                                        submatrix(j - 1, k - 1) = (*this)(j, k);
                                                }
                                        }
                                }
                                det += ((*this)(0, i) * submatrix.determinant() * ((i % 2 == 0) ? 1 : -1));
                        }
                        return det;
                }
        }

        Matrix<T, R, C> inverse() const
        {
                static_assert(R == C, "Inverse is only possible for square matrices.");
                Matrix<T, R, C> result;
                T det = determinant();
                if (det == 0)
                {
                        throw std::runtime_error("Matrix is singular.");
                }
                if constexpr (R == 1)
                {
                        result(0, 0) = 1 / (*this)(0, 0);
                }
                else if constexpr (R == 2)
                {
                        result(0, 0) = (*this)(1, 1) / det;
                        result(0, 1) = -(*this)(0, 1) / det;
                        result(1, 0) = -(*this)(1, 0) / det;
                        result(1, 1) = (*this)(0, 0) / det;
                }
                else
                {
                        for (uint64_t i = 0; i < R; i++)
                        {
                                for (uint64_t j = 0; j < C; j++)
                                {
                                        Matrix<T, R - 1, C - 1> submatrix;
                                        for (uint64_t k = 0; k < R; k++)
                                        {
                                                for (uint64_t l = 0; l < C; l++)
                                                {
                                                        if (k < i && l < j)
                                                        {
                                                                submatrix(k, l) = (*this)(k, l);
                                                        }
                                                        else if (k < i && l > j)
                                                        {
                                                                submatrix(k, l - 1) = (*this)(k, l);
                                                        }
                                                        else if (k > i && l < j)
                                                        {
                                                                submatrix(k - 1, l) = (*this)(k, l);
                                                        }
                                                        else if (k > i && l > j)
                                                        {
                                                                submatrix(k - 1, l - 1) = (*this)(k, l);
                                                        }
                                                }
                                        }
                                        result(j, i) = submatrix.determinant() * (((i + j) % 2 == 0) ? 1 : -1) / det;
                                }
                        }
                }
                return result;
        }
};

template <typename T, uint64_t R, uint64_t C>
class Matrix<T, R, C, false> : public Tensor<T, 2, R * C>
{
public:
        Matrix() : Tensor<T, 2, R * C>({R, C}) {}
        Matrix(const T &value) : Tensor<T, 2, R * C>({R, C}, value) {}
        Matrix(const HeapArray<T, R * C> &matrix) : Tensor<T, 2, R * C>(matrix) {}

        ~Matrix() = default;

        template <uint64_t R2, uint64_t C2>
        Matrix<T, R, C2> operator*(const Matrix<T, R2, C2> &matrix) const
        {
                static_assert(C == R2, "Matrix multiplication is only possible if the number of columns of the first matrix is equal to the number of rows of the second matrix.");
                Matrix<T, R, C2> result;
                // TODO: implement Strassen algorithm for matrix multiplication
                return result;
        }

        Matrix<T, R, C>& operator*=(const Matrix<T, R, C> &matrix)
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

        T determinant() const
        {
                static_assert(R == C, "Determinant is only possible for square matrices.");
                if constexpr (R == 1)
                {
                        return (*this)(0, 0);
                }
                else if constexpr (R == 2)
                {
                        return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
                }
                else
                {
                        T det = 0;
                        for (uint64_t i = 0; i < C; i++)
                        {
                                Matrix<T, R - 1, C - 1> submatrix;
                                for (uint64_t j = 1; j < R; j++)
                                {
                                        for (uint64_t k = 0; k < C; k++)
                                        {
                                                if (k < i)
                                                {
                                                        submatrix(j - 1, k) = (*this)(j, k);
                                                }
                                                else if (k > i)
                                                {
                                                        submatrix(j - 1, k - 1) = (*this)(j, k);
                                                }
                                        }
                                }
                                det += ((*this)(0, i) * submatrix.determinant() * ((i % 2 == 0) ? 1 : -1));
                        }
                        return det;
                }
        }

        Matrix<T, R, C> inverse() const
        {
                static_assert(R == C, "Inverse is only possible for square matrices.");
                Matrix<T, R, C> result;
                T det = determinant();
                if (det == 0)
                {
                        throw std::runtime_error("Matrix is singular.");
                }
                if constexpr (R == 1)
                {
                        result(0, 0) = 1 / (*this)(0, 0);
                }
                else if constexpr (R == 2)
                {
                        result(0, 0) = (*this)(1, 1) / det;
                        result(0, 1) = -(*this)(0, 1) / det;
                        result(1, 0) = -(*this)(1, 0) / det;
                        result(1, 1) = (*this)(0, 0) / det;
                }
                else
                {
                        for (uint64_t i = 0; i < R; i++)
                        {
                                for (uint64_t j = 0; j < C; j++)
                                {
                                        Matrix<T, R - 1, C - 1> submatrix;
                                        for (uint64_t k = 0; k < R; k++)
                                        {
                                                for (uint64_t l = 0; l < C; l++)
                                                {
                                                        if (k < i && l < j)
                                                        {
                                                                submatrix(k, l) = (*this)(k, l);
                                                        }
                                                        else if (k < i && l > j)
                                                        {
                                                                submatrix(k, l - 1) = (*this)(k, l);
                                                        }
                                                        else if (k > i && l < j)
                                                        {
                                                                submatrix(k - 1, l) = (*this)(k, l);
                                                        }
                                                        else if (k > i && l > j)
                                                        {
                                                                submatrix(k - 1, l - 1) = (*this)(k, l);
                                                        }
                                                }
                                        }
                                        result(j, i) = submatrix.determinant() * (((i + j) % 2 == 0) ? 1 : -1) / det;
                                }
                        }
                }
                return result;
        }
};

#endif // MATRIX_CPU_HPP
