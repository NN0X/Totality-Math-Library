#include <iostream>

#include "tensor_cpu.hpp"

int main()
{
        Tensor<float, 3, 27> tensor;
        std::cout << "Tensor 3x3x3\n";
        tensor.print();
        Tensor<float, 2, 9> tensor2;
        std::cout << "Tensor 3x3\n";
        tensor2.print();
        Tensor<float, 1, 3> tensor3;
        std::cout << "Tensor vector 3\n";
        tensor3.print();
        tensor(0, 0, 0) = 1;
        tensor(0, 0, 1) = 2;
        tensor(0, 0, 2) = 3;
        std::cout << "Tensor 3x3x3 after setting values\n";
        tensor.print();
}
