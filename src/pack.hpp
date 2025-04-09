#ifndef PACK_HPP
#define PACK_HPP
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <cassert>
#include <array>
#include <variant>

#include "heaparray.hpp"

template <typename T, uint64_t S>
class Pack
{
private:
        HeapArray<T, S> data;

public:
        template <typename... Args>
        Pack(Args... args)
        {
                static_assert(sizeof...(args) == S, "Number of arguments must match the size of the pack.");

                data = HeapArray<T, S>({args...});
        }

        T &operator[](size_t index)
        {
                return data[index];
        }

        const T &operator[](size_t index) const
        {
                return data[index];
        }

        T &at(size_t index)
        {
                if (index >= S)
                {
                        throw std::out_of_range("Index out of range");
                }
                return data[index];
        }

        const T &at(size_t index) const
        {
                if (index >= S)
                {
                        throw std::out_of_range("Index out of range");
                }
                return data[index];
        }

        size_t size() const
        {
                return S;
        }
};

// TODO: implement varied typed pack

#endif // PACK_HPP
