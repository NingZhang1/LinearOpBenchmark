/*
 * @Author: Ning Zhang
 * @Date: 2024-01-12 20:54:47
 * @Last Modified by: Ning Zhang
 * @Last Modified time: 2024-01-14 10:46:13
 */

#ifndef CONFIG_H
#define CONFIG_H

/* C lib */

#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <cstdlib>
#include <stddef.h>
#include <unistd.h>
#include <ctype.h>
#include <sys/mman.h> /* mmap munmap */
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <execinfo.h>

/* CXX Lib */

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <set>
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <complex>
#include <map>
#include <type_traits>
#include <memory>
#include <random>

/* 多线程相关 */

#define OMP_NUM_OF_THREADS std::atoi(getenv("OMP_NUM_THREADS"))
#define OMP_THREAD_LABEL omp_get_thread_num()

/* constant */

static const unsigned long long KB = 1000ULL;
static const unsigned long long MB = 1000000ULL;
static const unsigned long long GB = 1000000000ULL;
static const unsigned long long TB = 1000000000000ULL;

/* math lib */

#define iCI_complex_double std::complex<double>
#define iCI_complex_float std::complex<float>
#define USE_ISPC


#ifdef _MKL_
#include <mkl.h>
#else

#include <cblas.h>
#include <lapacke.h>

#define MKL_INT CBLAS_INDEX
#define MKL_Complex16 std::complex<double>
#define MKL_Complex8 std::complex<float>

inline void cblas_daxpyi(const MKL_INT N, const double alpha, const double *X,
                         const MKL_INT *indx, double *Y)
{
    for (MKL_INT i = 0; i < N; ++i)
    {
        Y[indx[i]] += alpha * X[i];
    }
}

inline double cblas_ddoti(const MKL_INT N, const double *X, const MKL_INT *indx,
                          const double *Y)
{
    double res = 0.0;
    for (MKL_INT i = 0; i < N; ++i)
    {
        res += X[i] * Y[indx[i]];
    }
    return res;
}

// #define CHECK_DSCRMV

inline void mkl_dcsrmv(const char *transa, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val,
                       const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta, double *y)
{
    for (auto irow = 0; irow < *m; ++irow)
    {
        double tmp = 0.0;
        for (auto icol = pntrb[irow]; icol < pntre[irow]; ++icol)
        {
            tmp += val[icol] * x[indx[icol]];
        }
        y[irow] = tmp;
    }
}

inline void mkl_zcsrmv(const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val,
                       const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y)
{
    for (auto irow = 0; irow < *m; ++irow)
    {
        MKL_Complex16 tmp = 0.0;
        for (auto icol = pntrb[irow]; icol < pntre[irow]; ++icol)
        {
            tmp += val[icol] * x[indx[icol]];
        }
        y[irow] = tmp;
    }
}

inline void cblas_dsctr(const MKL_INT N, const double *X, const MKL_INT *indx,
                        double *Y)
{
    for (MKL_INT i = 0; i < N; ++i)
    {
        Y[indx[i]] = X[i];
    }
}

#endif

/// 计时

#include <boost/chrono/process_cpu_clocks.hpp>
#include <boost/chrono.hpp>
#include <chrono>

using chrono_time_t = decltype(std::chrono::system_clock::now());
using boost_chrono_time_t = decltype(boost::chrono::process_cpu_clock::now());

inline double get_duration_in_ms(chrono_time_t start, chrono_time_t end)
{
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den * 1000;
}

inline double get_duration_in_ms(boost_chrono_time_t start, boost_chrono_time_t end)
{
    auto duration = boost::chrono::duration_cast<boost::chrono::microseconds>(end - start);
    return double(duration.count()) * boost::chrono::microseconds::period::num / boost::chrono::microseconds::period::den * 1000;
}

#endif // CONFIG_H