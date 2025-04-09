#include <iostream>

#include "tensor_cpu.hpp"
#include "matrix_cpu.hpp"
#include "vector_cpu.hpp"
#include "heaparray.hpp"
#include "pack.hpp"

Vector<double, 5> forwardSubstitution(Matrix<double, 5, 5> &L, Vector<double, 5> &b)
{
        Vector<double, 5> y;
        for (size_t i = 0; i < 5; i++)
        {
                y[i] = b[i];
                for (size_t j = 0; j < i; ++j)
                {
                        y[i] -= L(i, j) * y[j];
                }
        }
        return y;
}

Vector<double, 5> backwardSubstitution(Matrix<double, 5, 5> &U, Vector<double, 5> &y)
{
        Vector<double, 5> x;
        for (int i = 4; i >= 0; i--)
        {
                x[i] = y[i];
                for (int j = 4; j > i; j--)
                {
                        x[i] -= U(i, j) * x[j];
                }
                x[i] /= U(i, i);
        }
        return x;
}

Vector<double, 5> solveLinearEq(Matrix<double, 5, 5>& A, Vector<double, 5>& b)
{
        Matrix<double, 5, 5> P;
        Matrix<double, 5, 5> L = A.decomposeLU(P);
        Vector<double, 5> Pb = P * b;
        Vector<double, 5> y = forwardSubstitution(L, Pb);
        Vector<double, 5> x = backwardSubstitution(A, y);

        return x;
}

int main()
{
        HeapArray<double, 5 * 5> data1 = HeapArray<double, 5 * 5>({
                5, 4, 3, 2, 1,
                10, 8, 7, 6, 5,
                -1, 2, -3, 4, -5,
                6, 5, -4, 3, -2,
                1, 2, 3, 4, 5
        });
        Matrix<double, 5, 5> matrix1 = Matrix<double, 5, 5>(data1);
        Matrix<double, 5, 5> matrix1copy = matrix1;

        HeapArray<double, 5> data2 = HeapArray<double, 5>({
                37, 99, -9, 12, 53
        });
        Vector<double, 5> vector1 = Vector<double, 5>(data2);

        Vector<double, 5> result = solveLinearEq(matrix1, vector1);
        std::cout << "Result: ";
        result.print();
}
