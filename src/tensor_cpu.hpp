#ifndef TENSOR_CPU_HPP
#define TENSOR_CPU_HPP

#include <iostream>
#include <cstdint>
#include <array>
#include <stdexcept>
#include <string>
#include <cmath>
#include <iomanip>

#include "heaparray.hpp"
#include "assert_utils.hpp"

// TODO: implement tensor operations:
// - squeeze (remove dimensions with size 1)
// - unsqueeze (add dimension with size 1)
// - combine (combine two tensors along a dimension)
// - product (generalized kronecker product)

// TODO: optimize functions with multitheading and other

template <typename T, uint64_t R, uint64_t S>
class Tensor
{
private:
        HeapArray<T, S> mTensor;
        std::array<uint64_t, R> mShape;
        std::array<uint64_t, R> mStrides; // INFO: row-major order

public:
        Tensor() : mTensor()
        {
                static_assert(hasExactNthRoot(S, R), "Size must have an exact root of the rank of the tensor. Hint: Provide shape explicitly.");
                mShape.fill(static_cast<uint64_t>(std::pow(S, 1.0 / R)));
                mStrides.fill(1);
                for (uint64_t i = 0; i < R; i++)
                {
                        for (uint64_t j = i + 1; j < R; j++)
                        {
                                mStrides[i] *= mShape[j];
                        }
                }
        };

        Tensor(const T &value) : mTensor(value)
        {
                static_assert(hasExactNthRoot(S, R), "Size must have an exact root of the rank of the tensor. Hint: Provide shape explicitly.");
                mShape.fill(static_cast<uint64_t>(std::pow(S, 1.0 / R)));
                mStrides.fill(1);
                for (uint64_t i = 0; i < R; i++)
                {
                        for (uint64_t j = i + 1; j < R; j++)
                        {
                                mStrides[i] *= mShape[j];
                        }
                }
        }

        Tensor(const HeapArray<T, S> &tensor) : mTensor(tensor)
        {
                static_assert(hasExactNthRoot(S, R), "Size must have an exact root of the rank of the tensor. Hint: Provide shape explicitly.");
                mShape.fill(static_cast<uint64_t>(std::pow(S, 1.0 / R)));
                mStrides.fill(1);
                for (uint64_t i = 0; i < R; i++)
                {
                        for (uint64_t j = i + 1; j < R; j++)
                        {
                                mStrides[i] *= mShape[j];
                        }
                }
        }

        Tensor(const Tensor<T, R, S> &tensor) : mTensor(tensor.mTensor)
        {
                mShape = tensor.mShape;
                mStrides = tensor.mStrides;
        }

        Tensor(const std::array<uint64_t, R> &shape) : mTensor()
        {
                if (shape.size() != R)
                        throw std::runtime_error("Shape must have the same rank as the tensor.");
                uint64_t shapeSize = 1;
                for (uint64_t i = 0; i < R; i++)
                        shapeSize *= shape[i];
                if (shapeSize != S)
                        throw std::runtime_error("Size inferred from shape must match the size of the tensor.");
                mShape = shape;
                mStrides.fill(1);
                for (uint64_t i = 0; i < R; i++)
                {
                        for (uint64_t j = i + 1; j < R; j++)
                        {
                                mStrides[i] *= mShape[j];
                        }
                }
        }

        Tensor(const std::array<uint64_t, R> &shape, const T &value) : mTensor(value)
        {
                if (shape.size() != R)
                        throw std::runtime_error("Shape must have the same rank as the tensor.");
                uint64_t shapeSize = 1;
                for (uint64_t i = 0; i < R; i++)
                        shapeSize *= shape[i];
                if (shapeSize != S)
                        throw std::runtime_error("Size inferred from shape must match the size of the tensor.");
                mShape = shape;
                mStrides.fill(1);
                for (uint64_t i = 0; i < R; i++)
                {
                        for (uint64_t j = i + 1; j < R; j++)
                        {
                                mStrides[i] *= mShape[j];
                        }
                }
        }

        ~Tensor() = default;

        Tensor<T, R, S> &operator=(const Tensor<T, R, S> &tensor)
        {
                mTensor = tensor.mTensor;
                mShape = tensor.mShape;
                mStrides = tensor.mStrides;
                return *this;
        }

        template <typename... Args>
        T &operator()(Args... args)
        {
                static_assert(sizeof...(args) == R, "Number of arguments must match the rank of the tensor.");
                std::array<uint64_t, R> indices{static_cast<uint64_t>(args)...};
                uint64_t index = 0;
                for (uint64_t i = 0; i < R; i++)
                {
                        index += mStrides[i] * indices[i];
                }
                return mTensor[index];
        }

        T &operator()(const std::array<uint64_t, R> &indices)
        {
                if (indices.size() != R)
                        throw std::runtime_error("Number of arguments must match the rank of the tensor.");
                uint64_t index = 0;
                for (uint64_t i = 0; i < R; i++)
                {
                        index += mStrides[i] * indices[i];
                }
                return mTensor[index];
        }

        template <typename... Args>
        const T &operator()(Args... args) const
        {
                static_assert(sizeof...(args) == R, "Number of arguments must match the rank of the tensor.");
                std::array<uint64_t, R> indices{static_cast<uint64_t>(args)...};
                uint64_t index = 0;
                for (uint64_t i = 0; i < R; i++)
                {
                        index += mStrides[i] * indices[i];
                }
                return mTensor[index];
        }

        const T &operator()(const std::array<uint64_t, R> &indices) const
        {
                if (indices.size() != R)
                        throw std::runtime_error("Number of arguments must match the rank of the tensor.");
                uint64_t index = 0;
                for (uint64_t i = 0; i < R; i++)
                {
                        index += mStrides[i] * indices[i];
                }
                return mTensor[index];
        }

        T &operator[](const uint64_t index)
        {
                return mTensor[index];
        }

        const T &operator[](const uint64_t index) const
        {
                return mTensor[index];
        }

        T &at(const uint64_t index)
        {
                if (index >= S)
                        throw std::out_of_range("Index out of range.");
                return mTensor[index];
        }

        const T &at(const uint64_t index) const
        {
                if (index >= S)
                        throw std::out_of_range("Index out of range.");
                return mTensor[index];
        }

        Tensor<T, R, S> operator+(const Tensor<T, R, S> &tensor) const
        {
                Tensor<T, R, S> result = *this;
                for (uint64_t i = 0; i < S; i++)
                        result[i] += tensor[i];
                return result;
        }

        Tensor<T, R, S> &operator+=(const Tensor<T, R, S> &tensor)
        {
                for (uint64_t i = 0; i < S; i++)
                        mTensor[i] += tensor[i];
                return *this;
        }

        Tensor<T, R, S> operator-(const Tensor<T, R, S> &tensor) const
        {
                Tensor<T, R, S> result = *this;
                for (uint64_t i = 0; i < S; i++)
                        result[i] -= tensor[i];
                return result;
        }

        Tensor<T, R, S> &operator-=(const Tensor<T, R, S> &tensor)
        {
                for (uint64_t i = 0; i < S; i++)
                        mTensor[i] -= tensor[i];
                return *this;
        }

        Tensor<T, R, S> operator*(const T &scalar) const
        {
                Tensor<T, R, S> result = *this; 
                for (uint64_t i = 0; i < S; i++)
                        result[i] *= scalar;
                return result;
        }

        Tensor<T, R, S> &operator*=(const T &scalar)
        {
                for (uint64_t i = 0; i < S; i++)
                        mTensor[i] *= scalar;
                return *this;
        }

        Tensor<T, R, S> operator/(const T &scalar) const
        {
                Tensor<T, R, S> result = *this; 
                for (uint64_t i = 0; i < S; i++)
                        result[i] /= scalar;
                return result;
        }

        Tensor<T, R, S> &operator/=(const T &scalar)
        {
                for (uint64_t i = 0; i < S; i++)
                        mTensor[i] /= scalar;
                return *this;
        }

        template <uint64_t R2, uint64_t S2>
        Tensor<T, R, S*S2> product(const Tensor<T, R2, S2> &tensor) const
        {
                // TODO: implement generalized kronecker product
                return Tensor<T, R, S*S2>();
        }

        Tensor<T, R, S> hadamard(const Tensor<T, R, S> &tensor) const
        {
                Tensor<T, R, S> result;
                for (uint64_t i = 0; i < S; i++)
                        result[i] = mTensor[i] * tensor[i];
                return result;
        }

        template <uint64_t R2, uint64_t S2>
        Tensor<T, R2, S2> slice(uint64_t start, uint64_t end, uint64_t step) const
        {
                static_assert(R2 <= R, "Rank of the slice must be less than or equal to the rank of the tensor.");
                static_assert(S2 <= S, "Size of the slice must be less than or equal to the size of the tensor.");

                if (start >= S || end > S || start >= end)
                        throw std::out_of_range("Invalid slice range.");
                if (step == 0)
                        throw std::invalid_argument("Step must be greater than zero.");
                if ((end - start) % step != 0)
                        throw std::invalid_argument("Slice range must be divisible by the step.");

                if ((end - start) / step != S2)
                        throw std::runtime_error("Size of the slice must match the size of the tensor.");

                Tensor<T, R2, S2> result;
                uint64_t sliceIndex = 0;
                for (uint64_t i = start; i < end; i += step)
                {
                        result[sliceIndex++] = mTensor[i];
                }
                return result;
        }

        template <uint64_t R2>
        Tensor<T, R2, S> reshape(const std::array<uint64_t, R2> &shape) const
        {
                if (shape.size() != R2)
                        throw std::runtime_error("Shape must have the same rank as the tensor.");
                Tensor<T, R2, S> result(shape);
                for (uint64_t i = 0; i < S; i++)
                {
                        result[i] = mTensor[i];
                }
                return result;
        }

        Tensor<T, R, S> transpose(uint64_t dimA, uint64_t dimB) const
        {
                if (dimA >= R || dimB >= R)
                        throw std::out_of_range("Invalid dimensions for transpose.");
                if (dimA == dimB)
                        return *this;
                std::array<uint64_t, R> newShape = mShape;
                std::swap(newShape[dimA], newShape[dimB]);
                Tensor<T, R, S> result(newShape);
                for (uint64_t i = 0; i < S; i++)
                {
                        std::array<uint64_t, R> indices;
                        for (uint64_t j = 0; j < R; j++)
                        {
                                indices[j] = i / mStrides[j] % mShape[j];
                        }
                        std::array<uint64_t, R> newIndices = indices;
                        std::swap(newIndices[dimA], newIndices[dimB]);
                        result(newIndices) = (*this)(indices);
                }

                return result;
        }

        void fill(const T &value)
        {
                mTensor.fill(value);
        }

        void zero()
        {
                mTensor.zero();
        }

        void print(uint8_t precision) const
        {
                static_assert(R <= 3, "Printing is only supported for tensors with rank <= 3.");

                if constexpr (R == 0)
                {
                        std::cout << std::fixed << std::setprecision(precision) << mTensor[0] << "\n";
                }
                else if constexpr (R == 1)
                {
                        std::cout << "| ";
                        for (uint64_t elementIndex = 0; elementIndex < S; elementIndex++)
                        {
                                std::cout << std::fixed << std::setprecision(precision) << mTensor[elementIndex] << " ";
                        }
                        std::cout << "|\n";
                }
                else if constexpr (R == 2)
                {
                        for (uint64_t rowIndex = 0; rowIndex < mShape[0]; rowIndex++)
                        {
                                std::cout << "| ";
                                for (uint64_t colIndex = 0; colIndex < mShape[1]; colIndex++)
                                {
                                        std::cout << std::fixed << std::setprecision(precision) << mTensor[rowIndex * mShape[1] + colIndex] << " ";
                                }
                                std::cout << "|\n";
                        }
                }
                else if constexpr (R == 3)
                {
                        for (uint64_t matrixIndex = 0; matrixIndex < mShape[0]; matrixIndex++)
                        {
                                for (uint64_t rowIndex = 0; rowIndex < mShape[1]; rowIndex++)
                                {
                                        std::cout << "| ";
                                        for (uint64_t colIndex = 0; colIndex < mShape[2]; colIndex++)
                                        {
                                                uint64_t tensorIndex = matrixIndex * mShape[1] * mShape[2] + rowIndex * mShape[2] + colIndex;
                                                std::cout << std::fixed << std::setprecision(precision) << mTensor[tensorIndex] << " ";
                                        }
                                        std::cout << "| ";
                                }
                                std::cout << "\n";
                        }
                }
        }

        void print() const
        {
                print(6);
        }

        uint64_t size() const
        {
                return S;
        }

        uint64_t rank() const
        {
                return R;
        }

        std::array<uint64_t, R> shape() const
        {
                return mShape;
        }

        std::array<uint64_t, R> strides() const
        {
                return mStrides;
        }
};

#endif // TENSOR_CPU_HPP
