#ifndef RANDOM_GENERATOR_H
#define RANDOM_GENERATOR_H

#include "config.h"

/* >>>>>>>>>>>>>>>>>>> generate random number <<<<<<<<<<<<<<<<<<<<<< */

template <typename Scalar>
Scalar GenerateRandomNumber(Scalar _Input);

template <>
inline double GenerateRandomNumber<double>(double _Input)
{
    return (rand() / (double)(RAND_MAX)-0.5);
}

template <>
inline std::complex<double> GenerateRandomNumber<std::complex<double>>(std::complex<double> _input)
{
    std::complex<double> Res;
    Res.real(rand() / (double)(RAND_MAX)-0.5);
    Res.imag(rand() / (double)(RAND_MAX)-0.5);
    return Res;
}

inline std::vector<double> GenerateRandomDoubleVector(const int _nelmt)
{
    srand(time(NULL));
    std::vector<double> Res;
    for (int i = 0; i < _nelmt; ++i)
    {
        Res.push_back(GenerateRandomNumber<double>(0.0));
    }
    return Res;
}

inline std::vector<std::complex<double>> GenerateRandomComplexDoubleVector(const int _nelmt)
{
    srand(time(NULL));
    std::vector<std::complex<double>> Res;
    for (int i = 0; i < _nelmt; ++i)
    {
        Res.push_back(GenerateRandomNumber<std::complex<double>>(0.0));
    }
    return Res;
}

inline std::vector<double> GenerateRandomMatrix(const int _ndim)
{
    return GenerateRandomDoubleVector(_ndim * _ndim);
}

inline std::vector<double> GenerateRandomMatrix(const int _nrow, const int _ncol)
{
    return GenerateRandomDoubleVector(_nrow * _ncol);
}

/* >>>>>>>>>>>>>>>>>>> SparseMat Generator <<<<<<<<<<<<<<<<<<<<<< */

typedef std::pair<std::vector<double>, std::vector<MKL_INT>> ColIndxMat;

std::vector<MKL_INT> GenerateRandomIndx(const int _DIM, const int _NELMT)
{
    srand(time(NULL));
    std::vector<MKL_INT> Res;
    for (int i = 0; i < _NELMT; ++i)
    {
        Res.push_back(rand() % _DIM);
    }
    std::sort(Res.begin(), Res.end());
    Res.erase(std::unique(Res.begin(), Res.end()), Res.end());
    return Res;
}

ColIndxMat GenerateRandomSparseVector(const int _DIM, const double _DENSITY, const bool _ORDERED)
{
    assert(_DENSITY > 0 - 1e-10);
    assert(_DENSITY < 1 - 1e-10);
    const int NELMT = _DIM * _DENSITY + 1;
    auto Indx = GenerateRandomIndx(_DIM, NELMT);
    auto NELMT_REAL = Indx.size();
    if (!_ORDERED)
    {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::shuffle(Indx.begin(), Indx.end(), rng);
    }
    return ColIndxMat(GenerateRandomDoubleVector(Indx.size()), Indx);
}

#endif // !RANDOM_GENERATOR_H
