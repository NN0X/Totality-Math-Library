#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <cstdint>
#include <array>
#include <stdexcept>

#include "heaparray.hpp"

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
                mShape.fill(1);
                mStrides.fill(1);
        };

        Tensor(const T &value) : mTensor(value)
        {
                mShape.fill(1);
                mStrides.fill(1);
        }

        Tensor(const HeapArray<T, S> &tensor) : mTensor(tensor)
        {
                mShape.fill(1);
                mStrides.fill(1);
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
                mShape = shape;
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
                if (sizeof...(args) != R)
                        throw std::runtime_error("Number of arguments must match the rank of the tensor.");
                uint64_t index = 0;
                for (uint64_t i = 0; i < R; i++)
                {
                        index += mStrides[i] * args[i];
                }
                return mTensor[index];
        }

        template <typename... Args>
        const T &operator()(Args... args) const
        {
                if (sizeof...(args) != R)
                        throw std::runtime_error("Number of arguments must match the rank of the tensor.");
                uint64_t index = 0;
                for (uint64_t i = 0; i < R; i++)
                {
                        index += mStrides[i] * args[i];
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
                for (uint64_t i = 0; i < S; i++)
                        std::cout << mTensor[i] << ((i + 1 < S) ? " " : "");
                std::cout << "\n";
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
