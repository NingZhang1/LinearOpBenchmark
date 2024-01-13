#include <iostream>
#include <Eigen/Dense>

using namespace std;
int main()
{
    Eigen::Matrix2d mat;
    mat << 1, 2,
        3, 4;
    cout << "Here is mat.sum():       " << mat.sum() << endl;
    cout << "Here is mat.prod():      " << mat.prod() << endl;
    cout << "Here is mat.mean():      " << mat.mean() << endl;
    cout << "Here is mat.minCoeff():  " << mat.minCoeff() << endl;
    cout << "Here is mat.maxCoeff():  " << mat.maxCoeff() << endl;
    cout << "Here is mat.trace():     " << mat.trace() << endl;

    ////////////////////// NORM ///////////////////////

    Eigen::VectorXf v(2);
    Eigen::MatrixXf m(2, 2), n(2, 2);

    v << -1,
        2;

    m << 1, -2,
        -3, 4;

    std::cout << "v.squaredNorm() = " << v.squaredNorm() << std::endl;
    std::cout << "v.norm() = " << v.norm() << std::endl;
    std::cout << "v.lpNorm<1>() = " << v.lpNorm<1>() << std::endl;
    std::cout << "v.lpNorm<Infinity>() = " << v.lpNorm<Eigen::Infinity>() << std::endl;

    std::cout << std::endl;
    std::cout << "m.squaredNorm() = " << m.squaredNorm() << std::endl;
    std::cout << "m.norm() = " << m.norm() << std::endl;
    std::cout << "m.lpNorm<1>() = " << m.lpNorm<1>() << std::endl;
    std::cout << "m.lpNorm<Infinity>() = " << m.lpNorm<Eigen::Infinity>() << std::endl;

    //

    std::cout << "1-norm(m)     = " << m.cwiseAbs().colwise().sum().maxCoeff()
              << " == " << m.colwise().lpNorm<1>().maxCoeff() << std::endl;

    std::cout << "infty-norm(m) = " << m.cwiseAbs().rowwise().sum().maxCoeff()
              << " == " << m.rowwise().lpNorm<1>().maxCoeff() << std::endl;

    ///////////////////////// Boolean reduction /////////////////////////
    
    Eigen::ArrayXXf a(2, 2);

    a << 1, 2,
        3, 4;

    std::cout << "(a > 0).all()   = " << (a > 0).all() << std::endl;
    std::cout << "(a > 0).any()   = " << (a > 0).any() << std::endl;
    std::cout << "(a > 0).count() = " << (a > 0).count() << std::endl;
    std::cout << std::endl;
    std::cout << "(a > 2).all()   = " << (a > 2).all() << std::endl;
    std::cout << "(a > 2).any()   = " << (a > 2).any() << std::endl;
    std::cout << "(a > 2).count() = " << (a > 2).count() << std::endl;

    

}