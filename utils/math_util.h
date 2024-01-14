/*
 * @Author: Ning Zhang
 * @Date: 2024-01-09 23:33:52
 * @Last Modified by: Ning Zhang
 * @Last Modified time: 2024-01-14 10:46:46
 */

#ifndef MATH_UNROLL_H
#define MATH_UNROLL_H

#include "config.h"
#include "MathUtilISPC.h"

/* 利用模板自动展开代码 */

namespace MathUtil
{
    /* ---------- function overload for BLAS ---------- */

    /* ----- BLAS -1 ----- */

    inline float REAL(const float X) { return X; }
    inline double REAL(const double X) { return X; }
    inline float REAL(const iCI_complex_float X) { return X.real(); }
    inline double REAL(const iCI_complex_double X) { return X.real(); }

    inline float IMAG(const float X) { return 0.0; }
    inline double IMAG(const double X) { return 0.0; }
    inline float IMAG(const iCI_complex_float X) { return X.imag(); }
    inline double IMAG(const iCI_complex_double X) { return X.imag(); }

    /* ----- BLAS 0 ----- */

    /* NORM */

    inline float NORM(const float X) { return fabs(X); }
    inline double NORM(const double X) { return fabs(X); }
    inline float NORM(const iCI_complex_float X) { return std::fabs(X); }
    inline double NORM(const iCI_complex_double X) { return std::fabs(X); }

    /* NORM_SQUARE */

    inline float NORM_SQUARE(const float X) { return X * X; }
    inline double NORM_SQUARE(const double X) { return X * X; }
    inline float NORM_SQUARE(const iCI_complex_float X) { return X.real() * X.real() + X.imag() * X.imag(); }
    inline double NORM_SQUARE(const iCI_complex_double X) { return X.real() * X.real() + X.imag() * X.imag(); }

    // for historical reason

    inline float norm_square(const float X) { return X * X; }
    inline double norm_square(const double X) { return X * X; }
    inline float norm_square(const iCI_complex_float X) { return X.real() * X.real() + X.imag() * X.imag(); }
    inline double norm_square(const iCI_complex_double X) { return X.real() * X.real() + X.imag() * X.imag(); }

    /* COMPLEX CONJUGATE */

    inline float ComplexConjugate(const float X) { return X; }
    inline double ComplexConjugate(const double X) { return X; }
    inline iCI_complex_float ComplexConjugate(const iCI_complex_float X) { return std::conj(X); }
    inline iCI_complex_double ComplexConjugate(const iCI_complex_double X) { return std::conj(X); }

    inline void ComplexConjugate(const int N, double *X) {}

    inline void ComplexConjugate(const int N, iCI_complex_double *X)
    {
        for (int i = 0; i < N; ++i)
        {
            X[i] = std::conj(X[i]);
        }
    }

    /* HERMITIAN CONJUGATE */

    template <typename Scalar>
    std::vector<Scalar> HermitianConjugate(std::vector<Scalar> &_Input, const int nRow, const int nCol)
    {
        std::vector<Scalar> Res(nCol * nRow);
        for (int irow = 0; irow < nRow; ++irow)
        {
            for (int icol = 0; icol < nCol; ++icol)
            {
                Res[icol * nRow + irow] = ComplexConjugate(_Input[irow * nCol + icol]);
            }
        }
        return Res;
    }

    /* ----- BLAS 1 ----- */

    /* SCAL: x = ax */

    inline void SCAL(const MKL_INT n, const float a, float *x, const MKL_INT incx) { cblas_sscal(n, a, x, incx); }

    inline void SCAL(const MKL_INT n, const double a, double *x, const MKL_INT incx) { cblas_dscal(n, a, x, incx); }

    inline void SCAL(const MKL_INT n, const iCI_complex_float a, iCI_complex_float *x, const MKL_INT incx) { cblas_cscal(n, &a, x, incx); }

    inline void SCAL(const MKL_INT n, const iCI_complex_double a, iCI_complex_double *x, const MKL_INT incx) { cblas_zscal(n, &a, x, incx); }

    inline void SCAL(const MKL_INT n, const float a, iCI_complex_float *x, const MKL_INT incx) { cblas_csscal(n, a, x, incx); }

    inline void SCAL(const MKL_INT n, const double a, iCI_complex_double *x, const MKL_INT incx) { cblas_zdscal(n, a, x, incx); }

    /* AXPY: y = ax + y */

    inline void AXPY(const MKL_INT n, const float a, const float *x, const MKL_INT incx, float *y, const MKL_INT incy) { cblas_saxpy(n, a, x, incx, y, incy); }

    inline void AXPY(const MKL_INT n, const double a, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) { cblas_daxpy(n, a, x, incx, y, incy); }

    inline void AXPY(const MKL_INT n, const iCI_complex_float a, const iCI_complex_float *x, const MKL_INT incx, iCI_complex_float *y, const MKL_INT incy)
    {
        cblas_caxpy(n, (const void *)&a, x, incx, y, incy);
    }

    inline void AXPY(const MKL_INT n, const iCI_complex_double a, const iCI_complex_double *x, const MKL_INT incx, iCI_complex_double *y, const MKL_INT incy)
    {
        cblas_zaxpy(n, (const void *)&a, x, incx, y, incy);
    }

#ifdef USE_ISPC
    inline void AXPY(const MKL_INT n, const double a, const double *x, double *y) { ispc::vector_daxpy(y, a, x, n); }
    inline void AXPY(const MKL_INT n, const iCI_complex_double a, const iCI_complex_double *x, iCI_complex_double *y) { ispc::vector_zaxpy((double *)y, (const double *)&a, (const double *)x, n); }
#else
    inline void AXPY(const MKL_INT n, const double a, const double *x, double *y) { AXPY(n, a, x, 1, y, 1); }
    inline void AXPY(const MKL_INT n, const iCI_complex_double a, const iCI_complex_double *x, iCI_complex_double *y) { AXPY(n, a, x, 1, y, 1); }
#endif

    inline void AXPYI(const MKL_INT n, const iCI_complex_double a, const iCI_complex_double *x, const MKL_INT *indx, iCI_complex_double *y);

    inline void AXPYI(const MKL_INT n, const double a, const double *x, const MKL_INT *indx, double *y) { cblas_daxpyi(n, a, x, indx, y); }

    inline void AXPYI(const MKL_INT n, const iCI_complex_double a, const double *x, const MKL_INT *indx,
                      iCI_complex_double *y); /// this is the really intereesting thing!

    /* COPY : y = x */

    inline void COPY(const MKL_INT n, const float *x, const MKL_INT incx, float *y, const MKL_INT incy) { cblas_scopy(n, x, incx, y, incy); }

    inline void COPY(const MKL_INT n, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) { cblas_dcopy(n, x, incx, y, incy); }

    inline void COPY(const MKL_INT n, const iCI_complex_float *x, const MKL_INT incx, iCI_complex_float *y, const MKL_INT incy) { cblas_ccopy(n, x, incx, y, incy); }

    inline void COPY(const MKL_INT n, const iCI_complex_double *x, const MKL_INT incx, iCI_complex_double *y, const MKL_INT incy) { cblas_zcopy(n, x, incx, y, incy); }

    /* Inner Product */

    inline float DOT(const MKL_INT n, const float *x, const MKL_INT incx, const float *y, const MKL_INT incy) { return cblas_sdot(n, x, incx, y, incy); }

    inline double DOT(const MKL_INT n, const double *x, const MKL_INT incx, const double *y, const MKL_INT incy) { return cblas_ddot(n, x, incx, y, incy); }

    inline iCI_complex_float DOT(const MKL_INT n, const iCI_complex_float *x, const MKL_INT incx, const iCI_complex_float *y, const MKL_INT incy)
    {
        iCI_complex_float Res{0.0, 0.0};
        cblas_cdotc_sub(n, x, incx, y, incy, &Res);
        return Res;
    }

    inline iCI_complex_double DOT(const MKL_INT n, const iCI_complex_double *x, const MKL_INT incx, const iCI_complex_double *y, const MKL_INT incy)
    {
        iCI_complex_double Res{0.0, 0.0};
        cblas_zdotc_sub(n, x, incx, y, incy, &Res);
        return Res;
    }

    /* Inner Product, with one sparse vec */

    inline double DOTI(const MKL_INT N, const double *X, const MKL_INT *indx, const double *Y) { return cblas_ddoti(N, X, indx, Y); }

    inline iCI_complex_double DOTI(const MKL_INT N, const double *X, const MKL_INT *indx, const iCI_complex_double *Y)
    {
        iCI_complex_double res{0.0, 0.0};
        for (MKL_INT i = 0; i < N; ++i)
        {
            res += X[i] * Y[indx[i]];
        }
        return res;
    }

    /* Norm of vector */

    inline float NORM2_VEC(const MKL_INT n, const float *x, const MKL_INT incx) { return cblas_snrm2(n, x, incx); }

    inline double NORM2_VEC(const MKL_INT n, const double *x, const MKL_INT incx) { return cblas_dnrm2(n, x, incx); }

    inline float NORM2_VEC(const MKL_INT n, const iCI_complex_float *x, const MKL_INT incx) { return cblas_scnrm2(n, x, incx); }

    inline double NORM2_VEC(const MKL_INT n, const iCI_complex_double *x, const MKL_INT incx) { return cblas_dznrm2(n, x, incx); }

    /* ----- BLAS 2 ----- */

    /* Rank 1 update */

    inline void GER(const MKL_INT M, const MKL_INT N, const float alpha, const float *X, const MKL_INT incX, const float *Y, const MKL_INT incY, float *A, const MKL_INT lda)
    {
        cblas_sger(CblasRowMajor, M, N, alpha, X, incX, Y, incY, A, lda);
    }

    inline void GER(const MKL_INT M, const MKL_INT N, const double alpha, const double *X, const MKL_INT incX, const double *Y, const MKL_INT incY, double *A, const MKL_INT lda)
    {
        cblas_dger(CblasRowMajor, M, N, alpha, X, incX, Y, incY, A, lda);
    }

    inline void GER(const MKL_INT M, const MKL_INT N, const iCI_complex_float alpha, const iCI_complex_float *X, const MKL_INT incX, const iCI_complex_float *Y, const MKL_INT incY,
                    iCI_complex_float *A, const MKL_INT lda)
    {
        cblas_cgerc(CblasRowMajor, M, N, &alpha, X, incX, Y, incY, A, lda);
    }

    inline void GER(const MKL_INT M, const MKL_INT N, const iCI_complex_double alpha, const iCI_complex_double *X, const MKL_INT incX, const iCI_complex_double *Y, const MKL_INT incY,
                    iCI_complex_double *A, const MKL_INT lda)
    {
        cblas_zgerc(CblasRowMajor, M, N, &alpha, X, incX, Y, incY, A, lda);
    }

    /* CSR MV */

    inline void CSRMV(const char *transa, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb,
                      const MKL_INT *pntre, const double *x, const double *beta, double *y)
    {
        mkl_dcsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y);
    }

    inline void CSRMV(const char *transa, const MKL_INT *m, const MKL_INT *k, const iCI_complex_double *alpha, const char *matdescra, const iCI_complex_double *val, const MKL_INT *indx,
                      const MKL_INT *pntrb, const MKL_INT *pntre, const iCI_complex_double *x, const iCI_complex_double *beta, iCI_complex_double *y)
    {
        mkl_zcsrmv(transa, m, k, (const MKL_Complex16 *)alpha, matdescra, (const MKL_Complex16 *)val, indx, pntrb, pntre, (const MKL_Complex16 *)x, (const MKL_Complex16 *)beta,
                   (MKL_Complex16 *)y);
    }

    /// 极简版

    inline void CSRMV(const MKL_INT *m, const MKL_INT *k, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y)
    {
        static const char notrans = 'N';
        static const char matdescra[6] = {'G', 'L', 'N', 'C', 0, 1};
        static const double alpha = 1.0, beta = 0.0;

        mkl_dcsrmv(&notrans, m, k, &alpha, matdescra, val, indx, pntrb, pntre, x, &beta, y);
    }

    inline void CSRMV(const MKL_INT *m, const MKL_INT *k, const iCI_complex_double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const iCI_complex_double *x,
                      iCI_complex_double *y)
    {
        static const char notrans = 'N';
        static const char matdescra[6] = {'G', 'L', 'N', 'C', 0, 1};
        static const double alpha[2]{1.0, 0.0}, beta[2]{0.0, 0.0};

        mkl_zcsrmv(&notrans, m, k, (MKL_Complex16 *)&alpha, matdescra, (MKL_Complex16 *)val, indx, pntrb, pntre, (MKL_Complex16 *)x, (MKL_Complex16 *)&beta, (MKL_Complex16 *)y);
    }

    inline void CSRMV(const MKL_INT *m, const MKL_INT *k, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const iCI_complex_double *x, iCI_complex_double *y)
    {
        for (auto irow = 0; irow < *m; ++irow)
        {
            iCI_complex_double tmp{0.0, 0.0};
            for (auto icol = pntrb[irow]; icol < pntre[irow]; ++icol)
            {
                tmp += val[icol] * x[indx[icol]];
            }
            y[irow] = tmp;
        }
    }

    /// real ccf and complex integrals

    /* ----- BLAS 3 ----- */

    /* Matrix Product */

    inline void MatrixProduct(const int M, const int K, const int N, const float alpha, const float beta, const float *A, const float *B, float *C)
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
    }
    inline void MatrixProduct(const int M, const int K, const int N, const double alpha, const double beta, const double *A, const double *B, double *C)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
    }
    inline void MatrixProduct(const int M, const int K, const int N, const iCI_complex_float alpha, const iCI_complex_float beta, const iCI_complex_float *A, const iCI_complex_float *B,
                              iCI_complex_float *C)
    {
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha, A, K, B, N, &beta, C, N);
    }
    inline void MatrixProduct(const int M, const int K, const int N, const iCI_complex_double alpha, const iCI_complex_double beta, const iCI_complex_double *A, const iCI_complex_double *B,
                              iCI_complex_double *C)
    {
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha, A, K, B, N, &beta, C, N);
    }
    inline void MatrixProduct(const int M, const int K, const int N, const float *A, const float *B, float *C) { MatrixProduct(M, K, N, 1.0, 0.0, A, B, C); }
    inline void MatrixProduct(const int M, const int K, const int N, const double *A, const double *B, double *C) { MatrixProduct(M, K, N, 1.0, 0.0, A, B, C); }
    inline void MatrixProduct(const int M, const int K, const int N, const iCI_complex_float *A, const iCI_complex_float *B, iCI_complex_float *C)
    {
        static const iCI_complex_float alpha(1.0, 0.0), beta(0.0, 0.0);
        MatrixProduct(M, K, N, alpha, beta, A, B, C);
    }
    inline void MatrixProduct(const int M, const int K, const int N, const iCI_complex_double *A, const iCI_complex_double *B, iCI_complex_double *C)
    {
        static const iCI_complex_double alpha(1.0, 0.0), beta(0.0, 0.0);
        MatrixProduct(M, K, N, alpha, beta, A, B, C);
    }

    inline void GEMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N, const MKL_INT K, const float alpha, const float *A,
                     const MKL_INT lda, const float *B, const MKL_INT ldb, const float beta, float *C, const MKL_INT ldc)
    {
        cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    inline void GEMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N, const MKL_INT K, const double alpha, const double *A,
                     const MKL_INT lda, const double *B, const MKL_INT ldb, const double beta, double *C, const MKL_INT ldc)
    {
        cblas_dgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    inline void GEMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N, const MKL_INT K, const iCI_complex_float alpha,
                     const iCI_complex_float *A, const MKL_INT lda, const iCI_complex_float *B, const MKL_INT ldb, const iCI_complex_float beta, iCI_complex_float *C, const MKL_INT ldc)
    {
        cblas_cgemm(Layout, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    inline void GEMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N, const MKL_INT K, const iCI_complex_double alpha,
                     const iCI_complex_double *A, const MKL_INT lda, const iCI_complex_double *B, const MKL_INT ldb, const iCI_complex_double beta, iCI_complex_double *C, const MKL_INT ldc)
    {
        cblas_zgemm(Layout, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
}; // namespace MathUtil

/**
 * @brief Use the Metaprogramming technique to unroll the mathematical opeations
 * automatically. Only useful when the length of the arraries is not very large
 * and can be pre-determined. If not so, please call MKL subroutines.
 *
 * @tparam Scalar , one of float/double/iCI_Complex_float/iCI_Complex_double
 * @tparam N , the length of the arraies.
 * @par Function
 * <table>
 * <tr><th>Function              <th> Description
 * <tr><td>y_equal_x             <td> \f$ y=x \f$
 * <tr><td>y_equal_x_plus_y      <td> \f$ y+=x \f$
 * <tr><td>y_equal_ax_plus_y     <td> \f$ y+=ax \f$
 * <tr><td>y_equal_ax            <td> \f$ y=ax \f$
 * <tr><td>norm                  <td> return \f$ \| x\|  \f$
 * <tr><td>norm_square           <td> return \f$ \| x\|^2  \f$
 * </table>
 */
template <typename Scalar, int N>
class math_automatic_unroll
{
public:
    static void y_equal_x_plus_c(const Scalar *x, Scalar *y, const Scalar c)
    {
        *y = *x + c;
        math_automatic_unroll<Scalar, N - 1>::y_equal_x_plus_c(x + 1, y + 1, c);
    }
    static void y_equal_x(const Scalar *x, Scalar *y)
    {
        *y = *x;
        math_automatic_unroll<Scalar, N - 1>::y_equal_x(x + 1, y + 1);
    }
    static void y_equal_x_plus_y(const Scalar *x, Scalar *y)
    {
        *y += *x;
        math_automatic_unroll<Scalar, N - 1>::y_equal_x_plus_y(x + 1, y + 1);
    }
    static void y_equal_ax_plus_y(const Scalar a, const Scalar *x, Scalar *y)
    {
        *y += a * *x;
        math_automatic_unroll<Scalar, N - 1>::y_equal_ax_plus_y(a, x + 1, y + 1);
    }
    static void y_equal_ax(const Scalar a, const Scalar *x, Scalar *y)
    {
        *y = a * *x;
        math_automatic_unroll<Scalar, N - 1>::y_equal_ax(a, x + 1, y + 1);
    }
    static double norm(const MKL_INT _N, const Scalar *X, const MKL_INT stride) { return sqrt(norm_square(X, stride)); }
    static double norm_square(const Scalar *X, const int stride) { return MathUtil::NORM_SQUARE(*X) + math_automatic_unroll<Scalar, N - 1>::norm_square(X + stride, stride); }
    static Scalar ddot(const Scalar *X, const int incX, const Scalar *Y, const int incY)
    {
        return MathUtil::ComplexConjugate(*X) * *Y + math_automatic_unroll<Scalar, N - 1>::ddot(X + incX, incX, Y + incY, incY);
    }
    static Scalar ddot(const Scalar *X, const Scalar *Y) { return MathUtil::ComplexConjugate(*X) * *Y + math_automatic_unroll<Scalar, N - 1>::ddot(X + 1, Y + 1); }
    static void axpyi(const Scalar a, const Scalar *x, const MKL_INT *indx, Scalar *y)
    {
        y[*indx] += a * *x;
        math_automatic_unroll<Scalar, N - 1>::axpyi(a, x + 1, indx + 1, y);
    }
    static void abxpyi(const Scalar a, const double *b, const int8_t *op, const Scalar *x, const MKL_INT *indx, Scalar *y)
    {
        y[*indx] += a * *b * x[*op];
        math_automatic_unroll<Scalar, N - 1>::abxpyi(a, b + 1, op + 1, x, indx + 1, y);
    }
    static void conj(double *x) {}
    static void conj(iCI_complex_double *x)
    {
        x->imag(-x->imag());
        math_automatic_unroll<Scalar, N - 1>::conj(x + 1);
    }
};

template <typename Scalar>
class math_automatic_unroll<Scalar, 1>
{
public:
    static void y_equal_x_plus_c(const Scalar *x, Scalar *y, const Scalar c) { *y = *x + c; }
    static void y_equal_x(const Scalar *x, Scalar *y) { *y = *x; }
    static void y_equal_x_plus_y(const Scalar *x, Scalar *y) { *y += *x; }
    static void y_equal_ax_plus_y(const Scalar a, const Scalar *x, Scalar *y) { *y += a * *x; }
    static void y_equal_ax(const Scalar a, const Scalar *x, Scalar *y) { *y = a * *x; }
    static double norm(const MKL_INT _N, const Scalar *X, const MKL_INT stride) { return MathUtil::NORM(*X); }
    static double norm_square(const Scalar *X, const int stride) { return MathUtil::NORM_SQUARE(*X); }
    static Scalar ddot(const Scalar *X, const int incX, const Scalar *Y, const int incY) { return MathUtil::ComplexConjugate(*X) * *Y; }
    static Scalar ddot(const Scalar *X, const Scalar *Y) { return MathUtil::ComplexConjugate(*X) * *Y; }
    static void axpyi(const Scalar a, const Scalar *x, const MKL_INT *indx, Scalar *y) { y[*indx] += a * *x; }
    static void abxpyi(const Scalar a, const double *b, const int8_t *op, const Scalar *x, const MKL_INT *indx, Scalar *y) { y[*indx] += a * *b * x[*op]; }
    static void conj(double *x) {}
    static void conj(iCI_complex_double *x) { x->imag(-x->imag()); }
};

#endif