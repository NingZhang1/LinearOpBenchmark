#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
    MatrixXcf a = MatrixXcf::Random(2, 2); /// random
    cout << "Here is the matrix a\n"
         << a << endl;

    cout << "Here is the matrix a^T\n"
         << a.transpose() << endl; // NOTE that it is a expression not a real solid matrix!

    cout << "Here is the conjugate of a\n"
         << a.conjugate() << endl;

    cout << "Here is the matrix a^*\n"
         << a.adjoint() << endl;

    // do not use
    // m = m.transpose();
    // you can use m = m.transpose().eval(); to force evaluation
    // or m.transposeInPlace(); to do it in place

    MatrixXf b(2, 3);
    b << 1, 2, 3, 4, 5, 6;
    cout << "Here is the initial matrix a:\n"
         << b << endl;

    b.transposeInPlace();
    cout << "and after being transposed:\n"
         << b << endl;
}