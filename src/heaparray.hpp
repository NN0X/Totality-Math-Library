#ifndef HEAPARRAY_HPP
#define HEAPARRAY_HPP

#include <iostream>
#include <cstdint>
#include <array>
#include <memory>
#include <stdexcept>

template <typename T, uint64_t N>
class HeapArray
{
private:
        std::unique_ptr<std::array<T, N>> mArray;

public:
        HeapArray() : mArray(std::make_unique<std::array<T, N>>())
        {
                mArray->fill(0);
        };

        HeapArray(const T &value) : mArray(std::make_unique<std::array<T, N>>())
        {
                mArray->fill(value);

        }

        HeapArray(const std::array<T, N> &array) : mArray(std::make_unique<std::array<T, N>>())
        {
                *mArray = array;
        }

        HeapArray(const HeapArray<T, N> &array) : mArray(std::make_unique<std::array<T, N>>())
        {
                *mArray = *array.mArray;
        }

        ~HeapArray() = default;

        HeapArray<T, N> &operator=(const HeapArray<T, N> &array)
        {
                *mArray = *array.mArray;
                return *this;
        }

        T &operator[](uint64_t index)
        {
                if (index >= N)
                        throw std::out_of_range("Index out of range.");
                return (*mArray)[index];
        }

        const T &operator[](uint64_t index) const
        {
                if (index >= N)
                        throw std::out_of_range("Index out of range.");
                return (*mArray)[index];
        }

        void fill(const T &value)
        {
                mArray->fill(value);
        }

        void zero()
        {
                mArray->fill(0);
        }

        void print() const
        {
                for (uint64_t i = 0; i < N; i++)
                        std::cout << (*mArray)[i] << ((i + 1 < N) ? " " : "");
                std::cout << "\n";
        }

        uint64_t size() const
        {
                return N;
        }

        T sum() const
        {
                uint64_t sum = 0;
                for (uint64_t i = 0; i < N; i++)
                        sum += (*mArray)[i];
                return sum;
        }

        T mul() const
        {
                uint64_t mul = 1;
                for (uint64_t i = 0; i < N; i++)
                        mul *= (*mArray)[i];
                return mul;
        }
};

#endif // HEAPARRAY_HPP
