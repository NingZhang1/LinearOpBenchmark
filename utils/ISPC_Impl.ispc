
export void vector_add_constant (uniform double a[],
                                 const uniform double b,
                                 const uniform int size)
{
    foreach (i = 0 ... size)
        a[i] += b;
}

/// BLAS 1 

export void vector_daxpy (uniform double a[],
                          const uniform double alpha,
                          const uniform double b[],
                          const uniform int size)
{
    foreach (i = 0 ... size)
        a[i] += alpha * b[i];
}

export void vector_zaxpy (uniform double a[],
                          const uniform double alpha[],
                          const uniform double b[],
                          const uniform int size)
{
    foreach(n = 0...size)
    {
        a[2*n]   += alpha[0] * b[2*n] + alpha[1] * b[2*n+1];
        a[2*n+1] += -alpha[1] * b[2*n] + alpha[0] * b[2*n+1];
    }
}

export uniform double vector_dot (const uniform double a[],
                                  const uniform double b[],
                                  const uniform int size)
{
    varying double sum = 0.0;
    foreach (i = 0 ... size)
        sum += a[i] * b[i];
    return reduce_add(sum);
}

