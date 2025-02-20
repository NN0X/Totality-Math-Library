#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <cstdint>
#include <array>
#include <stdexcept>
#include <string>
#include <cmath>
#include <iomanip>

#include "heaparray.hpp"
#include "assert_utils.hpp"

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

        void fill(const T &value)
        {
                mTensor.fill(value);
        }

        void zero()
        {
                mTensor.zero();
        }

        void print() const
        {
                static_assert(R <= 3, "Printing is only supported for tensors with rank <= 3.");

                if constexpr (R == 1)
                {
                        std::cout << "| ";
                        for (uint64_t elementIndex = 0; elementIndex < S; elementIndex++)
                        {
                                std::cout << std::fixed << std::setprecision(6) << mTensor[elementIndex] << " ";
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
                                        std::cout << std::fixed << std::setprecision(6) << mTensor[rowIndex * mShape[1] + colIndex] << " ";
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
                                                std::cout << std::fixed << std::setprecision(6) << mTensor[tensorIndex] << " ";
                                        }
                                        std::cout << "| ";
                                }
                                std::cout << "\n";
                        }
                }
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

#endif // TENSOR_HPP
