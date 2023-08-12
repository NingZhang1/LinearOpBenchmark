/*
 * @Author: Ning Zhang
 * @Date: 2023-08-11 16:23:56
 * @Last Modified by: Ning Zhang
 * @Last Modified time: 2023-08-11 17:28:09
 */

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

#include "Kron_Test.h"

const int TEST_ARRAY[]{8, 64, 256, 1024};

int main()
{
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            const int N = TEST_ARRAY[i];
            const int M = TEST_ARRAY[j];

            Test(N, M, KronProduct_MKL);
        }
    }
}
