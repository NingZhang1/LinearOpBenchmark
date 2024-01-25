#include <iostream>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/Sparse>
#include "../../utils/random_generator.h"

using SparseMat_t = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using triplet_t = Eigen::Triplet<double>;

#define Row 10
#define Col 16

int main()
{
    srand(time(NULL));

    Eigen::MatrixXd m = Eigen::MatrixXd::Random(Row, Col);

    std::cout << "m =" << std::endl
              << m << std::endl;

    // perform ColPivotQR

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(m);
    Eigen::MatrixXd Q, R, R1, R2;

    std::cout << "rank = " << qr.rank() << std::endl;

    auto P = qr.colsPermutation();
    auto indices = P.indices();

    std::cout << "P =" << std::endl
              << indices << std::endl;

    Q = qr.householderQ();
    R = qr.matrixQR().triangularView<Eigen::Upper>();

    std::cout << "Q =" << std::endl
              << Q << std::endl;

    std::cout << "R =" << std::endl
              << R << std::endl;

    std::cout << "Q * R =" << std::endl
              << Q * R << std::endl;

    std::cout << "Q * R * P^{-1} =" << std::endl
              << Q * R * P.inverse() << std::endl;

    R1 = R.block(0, 0, Row, Row);
    R2 = R.block(0, 0, Row, Col);

    std::cout << "R1 =" << std::endl
              << R1 << std::endl;
    std::cout << "R2 =" << std::endl
              << R2 << std::endl;

    auto R3 = R1.inverse().eval();

    std::cout << "R1^-1 =" << std::endl
              << R3 << std::endl;

    std::cout << "R1^-1 * R1 =" << std::endl
              << R3 * R1 << std::endl;

    std::cout << "m =" << std::endl
              << m << std::endl;

    /// check whether the memory address is chagned

    std::cout << "m.data()  = " << m.data() << std::endl;
    std::cout << "Q.data()  = " << Q.data() << std::endl;
    std::cout << "R.data()  = " << R.data() << std::endl;
    std::cout << "R1.data() = " << R1.data() << std::endl;
    std::cout << "R2.data() = " << R2.data() << std::endl;
    std::cout << "R3.data() = " << R3.data() << std::endl;

    std::cout << "block R ^{-1} * R * P^{-1}" << std::endl
              << R1.triangularView<Eigen::Upper>().solve(R2) * P.inverse() << std::endl;

    std::cout << "should equal to" << std::endl
              << R3 * R2 * P.inverse() << std::endl;
}