#include <iostream>

#include "tensor_cpu.hpp"
#include "matrix_cpu.hpp"
#include "vector_cpu.hpp"

int main()
{
        // TENSOR

        // scalar tensor
        Tensor<int, 0, 1> scalar;
        std::cout << "Tensor scalar:\n";
        scalar.print();

        // vector tensor
        Tensor<float, 3, 27> tensor;
        std::cout << "Tensor 3x3x3:\n";
        tensor.print();

        // matrix tensor
        Tensor<float, 2, 9> matrix;
        std::cout << "Tensor 3x3:\n";
        matrix.print();

        // vector tensor assigment test
        Tensor<float, 1, 3> vector;
        std::cout << "Tensor vector 3:\n";
        vector.print();
        tensor(0, 0, 0) = 1;
        tensor(0, 0, 1) = 2;
        tensor(0, 0, 2) = 3;
        std::cout << "Tensor 3x3x3 after setting values:\n";
        tensor.print();
        Tensor<float, 2, 4> matrix1; // 2x2
        Tensor<float, 2, 4> matrix2; // 2x2
        matrix1(0, 0) = 1;
        matrix1(0, 1) = 2;
        matrix1(1, 0) = 3;
        matrix1(1, 1) = 4;
        std::cout << "Matrix 1 2x2:\n";
        matrix1.print();
        matrix2(0, 0) = 0;
        matrix2(0, 1) = 5;
        matrix2(1, 0) = 6;
        matrix2(1, 1) = 7;
        std::cout << "Matrix 2 2x2:\n";
        matrix2.print();

        //std::cout << "Tensor product of 2 2x2 matricies:\n";
        //matrix1.product(matrix2).print();
        //Tensor<float, 3, 27> tensorP(2.0f);
        //std::cout << "Tensor product of 3x3x3 and 3x3x3:\n";
        //tensor.product(tensorP).print();
        //std::cout << "Tensor hadamard multiplication of 2 2x2 matricies:\n";

        // hadamard multiplication test
        matrix1.hadamard(matrix2).print();
        std::cout << "Tensor addition of 2 2x2 matricies:\n";
        (matrix1 + matrix2).print();
        std::cout << "Tensor subtraction of 2 2x2 matricies:\n";
        (matrix1 - matrix2).print();

        std::cout << "Tensor hadamard multiplication of 3x3x3 and 3x3x3:\n";
        tensor.hadamard(tensor).print();
        std::cout << "Tensor addition of 3x3x3 and 3x3x3:\n";
        (tensor + tensor).print();
        std::cout << "Tensor subtraction of 3x3x3 and 3x3x3:\n";
        (tensor - tensor).print();

        // scalar operations test
        std::cout << "Tensor scalar multiplication of 3x3x3:\n";
        tensor *= 2.0f;
        tensor.print();
        std::cout << "Tensor scalar division of 3x3x3:\n";
        tensor /= 4.0f;
        tensor.print();

        // slice test
        Tensor<float, 5, 3125> tensorSlicable(5.0f);
        Tensor<float, 2, 4> slice = tensorSlicable.slice<2, 4>(10, 18, 2);
        std::cout << "Tensor slice of 5x5x5x5x5 from 10 to 18 with step 2 (2x2 matrix):\n";
        slice.print();

        // reshape test
        const std::array<uint64_t, 10> shape = {10, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        Tensor<float, 10, 10> tensorReshapable(shape, 5.0f);
        Tensor<float, 2, 10> reshaped = tensorReshapable.reshape<2>({2, 5});
        std::cout << "Tensor reshaped from 10x1x1x1x1x1x1x1x1x1 to 2x5:\n";
        reshaped.print();

        // transpose test
        Tensor<float, 2, 4> matrixTransposable;
        matrixTransposable(0, 0) = 1;
        matrixTransposable(0, 1) = 2;
        matrixTransposable(1, 0) = 3;
        matrixTransposable(1, 1) = 4;
        std::cout << "Matrix 2x2:\n";
        matrixTransposable.print();
        Tensor<float, 2, 4> transposed = matrixTransposable.transpose(0, 1);
        std::cout << "Matrix 2x2 transposed:\n";
        transposed.print();

        Tensor<float, 3, 27> tensorTransposable;
        tensorTransposable(0, 0, 0) = 1;
        tensorTransposable(0, 0, 1) = 2;
        tensorTransposable(0, 0, 2) = 3;
        tensorTransposable(0, 1, 0) = 4;
        tensorTransposable(0, 1, 1) = 5;
        tensorTransposable(0, 1, 2) = 6;
        tensorTransposable(0, 2, 0) = 7;
        tensorTransposable(0, 2, 1) = 8;
        tensorTransposable(0, 2, 2) = 9;
        std::cout << "Tensor 3x3x3:\n";
        tensorTransposable.print();
        Tensor<float, 3, 27> tensorTransposed = tensorTransposable.transpose(0, 2);
        std::cout << "Tensor 3x3x3 transposed:\n";
        tensorTransposed.print();

        // squeeze test
        Tensor<float, 3, 27> tensorSqueezable({1, 1, 27});
        std::cout << "Tensor 1x1x27:\n";
        tensorSqueezable.print();
        Tensor<float, 2, 27> tensorSqueezed = tensorSqueezable.squeeze();
        std::cout << "Tensor 1x1x27 squeezed to 1x27:\n";
        tensorSqueezed.print();

        // unsqueeze test
        Tensor<float, 3, 27> tensorUnsqueezed = tensorSqueezed.unsqueeze(0);
        std::cout << "Tensor 1x27 unsqueezed to 1x1x27:\n";
        tensorUnsqueezed.print();

        // combine test
        Tensor<float, 2, 4> matrixCombining1(1.0f);
        Tensor<float, 2, 4> matrixCombining2(2.0f);
        std::cout << "Matrix 2x2:\n";
        matrixCombining1.print();
        Tensor<float, 2, 8> matrixCombined = matrixCombining1.combine(matrixCombining2, 0);
        std::cout << "Matrix 2x2 combined with 2x2 along axis 0 (vertical):\n";
        matrixCombined.print();

        // MATRIX

        // matrix multiplication test
        Matrix<float, 3, 3> matrix1M(2.0f);
        std::cout << "Matrix 3x3:\n";
        matrix1M.print();
        Matrix<float, 3, 3> matrix2M(3.0f);
        std::cout << "Matrix 3x3:\n";
        matrix2M.print();
        std::cout << "Matrix product of 2 3x3 matricies:\n";
        (matrix1M * matrix2M).print();

        // matrix transpose test
        Matrix<float, 3, 3> matrixTransposableM;
        matrixTransposableM(0, 0) = 1;
        matrixTransposableM(0, 1) = 2;
        matrixTransposableM(0, 2) = 3;
        matrixTransposableM(1, 0) = 4;
        matrixTransposableM(1, 1) = 5;
        matrixTransposableM(1, 2) = 6;
        matrixTransposableM(2, 0) = 7;
        matrixTransposableM(2, 1) = 8;
        matrixTransposableM(2, 2) = 9;
        std::cout << "Matrix 3x3:\n";
        matrixTransposableM.print();
        Matrix<float, 3, 3> transposedM = matrixTransposableM.transpose();
        std::cout << "Matrix 3x3 transposed:\n";
        transposedM.print();

        // VECTOR

        Vector<float, 3> vector1(1.0f);
        std::cout << "Vector 3:\n";
        vector1.print();
}
