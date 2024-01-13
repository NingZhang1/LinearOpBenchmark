#include <iostream>
#include <Eigen/Dense>

int main()
{
    Eigen::MatrixXf m(2, 2);

    m << 1, 2,
        3, 4;

    // get location of maximum
    Eigen::Index maxRow, maxCol;
    float max = m.maxCoeff(&maxRow, &maxCol);

    // get location of minimum
    Eigen::Index minRow, minCol;
    float min = m.minCoeff(&minRow, &minCol);

    std::cout << "Max: " << max << ", at: " << maxRow << "," << maxCol << std::endl;
    std::cout << "Min: " << min << ", at: " << minRow << "," << minCol << std::endl;

    Eigen::MatrixXf mat(2, 4);
    mat << 1, 2, 6, 9,
        3, 1, 7, 2;

    std::cout << "Column's maximum: " << std::endl
              << mat.colwise().maxCoeff() << std::endl;

    std::cout << "Row's maximum: " << std::endl
              << mat.rowwise().maxCoeff() << std::endl;

    Eigen::Index maxIndex;
    float maxNorm = mat.colwise().sum().maxCoeff(&maxIndex);

    std::cout << "Maximum sum at position " << maxIndex << std::endl;

    std::cout << "The corresponding vector is: " << std::endl;
    std::cout << mat.col(maxIndex) << std::endl;
    std::cout << "And its sum is is: " << maxNorm << std::endl;
}