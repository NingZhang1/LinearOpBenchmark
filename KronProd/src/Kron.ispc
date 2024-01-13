/*
 * @Author: Ning Zhang 
 * @Date: 2023-08-11 16:23:59 
 * @Last Modified by: Ning Zhang
 * @Last Modified time: 2023-08-11 17:30:38
 */

export void KronProduct(
    uniform const int N,
    uniform const int M, 
    uniform const double A[],
    uniform const double B[],
    uniform double C[]
){
    for (uniform int i = 0; i < N; i++){
        uniform double *ptr = C + i*M;
        const uniform double CONST = A[i];
        foreach (j = 0 ... M){
            ptr[j] = CONST * B[j];
        }
    }
}

/// kron inner 

export void KronProduct_Inner(
    uniform const int Bunch,
    uniform const int N,
    uniform const int M, 
    uniform const double A[],
    uniform const double B[],
    uniform double C[]
){
    foreach (i = 0 ... Bunch){
        double *A1 = A + i*N;
        double *B1 = B + i*M;
        double *C1 = C + i*N*M; 
        for(int j = 0; j < N; j++){
            double *ptr = C1 + j*M;
            const double CONST = A1[j]; /// memory access may be very slow
            for(int k = 0; k < M; k++){
                ptr[k] = CONST * B1[k]; /// memory access may be very slow
            }
        }
    }
}

/// kron outer

export void KronProduct_ReOrder(
    uniform const int Bunch,
    uniform const int N,
    uniform const int M, 
    uniform const double A[],
    uniform const double B[],
    uniform double C[]
){
    for(uniform i=0;i<N;i++){
        for(uniform j=0;j<M;++j)
        foreach (k = 0 ... Bunch){
            C[(i*M+j)*Bunch+k] = A[i*Bunch+k] * B[j*Bunch+k];
        }
    }   
}