#include <iostream>
#include <Eigen/Dense>

using namespace std;
int main()
{
    Eigen::MatrixXf mat(2, 4);
    Eigen::VectorXf v(2);

    mat << 1, 2, 6, 9,
        3, 1, 7, 2;

    v << 0,
        1;

    // add v to each column of m
    mat.colwise() += v;

    std::cout << "Broadcasting result: " << std::endl;
    std::cout << mat << std::endl;

    Eigen::VectorXf v2(4);

    mat << 1, 2, 6, 9,
        3, 1, 7, 2;

    v2 << 0, 1, 2, 3;

    // add v2 to each row of m
    mat.rowwise() += v2.transpose();

    std::cout << "Broadcasting result: " << std::endl;
    std::cout << mat << std::endl;
}