
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

export void AccumulateEptStateMajor(
    const uniform double sigma[],
    const uniform double dg[],
    const uniform double eiCI[],
    const uniform int ncsf,
    const uniform int nstates,
    uniform double out[])
{
    for(uniform int istate=0;istate<nstates;++istate)
    { 
        varying double restmp=0.0;
        const double uniform eiCI_now = eiCI[istate]; 
        const double * uniform PTR_UNIFORM = sigma + istate*ncsf;
        foreach(icsf=0...ncsf)
        {
            restmp+=PTR_UNIFORM[icsf]*PTR_UNIFORM[icsf]/(eiCI_now-dg[icsf]);
        }
        out[istate]+=reduce_add(restmp);
    }
}

export void AccumulateEptCSFMajorKernel(
    const uniform double sigma[],
    const uniform double dg,
    const uniform double eiCI[],
    const uniform int nstates,
    uniform double out[])
{
    foreach(istate=0...nstates)
    { 
        out[istate] += sigma[istate]*sigma[istate]/(eiCI[istate]-dg);
    }
}