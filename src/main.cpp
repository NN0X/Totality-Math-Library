#include <iostream>
#include <cstdint>
#include <array>
#include <random>
#include <chrono>

#include "tensor_cpu.hpp"
#include "matrix_cpu.hpp"
#include "vector_cpu.hpp"
#include "heaparray.hpp"

template <typename T, uint64_t N>
HeapArray<T, N> fillRandom()
{
        HeapArray<T, N> array;
        for (uint64_t i = 0; i < N; i++)
        {
                array[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        }
        return array;
}

int main()
{
        Matrix<float, 20000, 20000> matrix1(fillRandom<float, 20000*20000>());
        Matrix<float, 20000, 20000> matrix2(fillRandom<float, 20000*20000>());
        auto start = std::chrono::high_resolution_clock::now();
        Matrix<float, 20000, 20000> matrix3 = matrix1 * matrix2;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << matrix3(0, 0) << "\n";
        std::cout << "Matrix multiplication: " << diff.count() << " s\n";
}
