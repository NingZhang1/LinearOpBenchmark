#include "../../utils/config.h"
#include "../../utils/random_generator.h"
#include "../../utils/MathUtilISPC.h"
#include "../../utils/math_util.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#define OneGOperation 1000000000ULL
#define FACTOR 4

// test with different size of problem, while with different stride !

void vector_zaxpy_loop(const iCI_complex_double *a, const iCI_complex_double alpha, iCI_complex_double *b, const int32_t size)
{
    for (int i = 0; i < size; ++i)
    {
        b[i] += a[i] * alpha + b[i];
    }
}

size_t _get_loop_time(size_t n)
{
    return OneGOperation / (2 * n * FACTOR);
}

double _get_GFLO(size_t n, size_t test_time)
{
    return (2 * double(n) * test_time) / OneGOperation;
}

int main(int argc, const char **argv)
{
    Eigen::initParallel();

    /// define the size of the problem

    uint64_t TASK[]{
        // 4, 8, 16, 32,
        64, 128, 256, 512,
        1 * KB, 2 * KB, 4 * KB, 8 * KB, 16 * KB, 32 * KB,
        64 * KB,
        128 * KB,
        256 * KB,
        512 * KB,
        1 * MB,
        2 * MB, 4 * MB,
        // 8 * MB, 16 * MB, 32 * MB
    };
    const int nTASK = sizeof(TASK) / sizeof(uint64_t);

    /* (2) plain loop */

    auto nThread = OMP_NUM_OF_THREADS;

    printf("nThread=%d\n", nThread);

    for (int i = 0; i < nTASK; ++i)
    {
        auto TestTime = _get_loop_time(TASK[i]) * nThread;
        // printf("TestTime=%d\n", TestTime);
        std::vector<iCI_complex_double> VecA = GenerateRandomComplexDoubleVector(TASK[i] * nThread);
        std::vector<iCI_complex_double> VecB = GenerateRandomComplexDoubleVector(TASK[i] * nThread);

        iCI_complex_double *headB = VecB.data();
        iCI_complex_double B = GenerateRandomNumber(0.0);
        size_t size = VecB.size();
        memset(headB, 0, sizeof(iCI_complex_double) * size);

        /////// RUN the Test

        auto time_begin_wall = std::chrono::system_clock::now();
        auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
#pragma omp parallel for num_threads(nThread) schedule(dynamic)
        for (int j = 0; j < TestTime; ++j)
        {
            auto id = OMP_THREAD_LABEL;
            vector_zaxpy_loop(VecA.data() + id * TASK[i], B, VecB.data() + id * TASK[i], TASK[i]);
        }
        auto time_end_wall = std::chrono::system_clock::now();
        auto time_end_cpu = boost::chrono::process_cpu_clock::now();
        double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
        double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

        printf("vector_zaxpy_loop with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], duration_wall,
               duration_cpu, _get_GFLO(TASK[i], TestTime) * 1000 / duration_wall);
    }

    /// EigenMap

    for (int i = 0; i < nTASK; ++i)
    {
        auto TestTime = _get_loop_time(TASK[i]) * nThread;
        std::vector<iCI_complex_double> VecA = GenerateRandomComplexDoubleVector(TASK[i] * nThread);
        std::vector<iCI_complex_double> VecB = GenerateRandomComplexDoubleVector(TASK[i] * nThread);

        iCI_complex_double *headB = VecB.data();
        iCI_complex_double C = GenerateRandomNumber(0.0);
        size_t size = VecB.size();
        memset(headB, 0, sizeof(iCI_complex_double) * size);

        /////// RUN the Test

        std::vector<Eigen::Map<Eigen::ArrayXcd>> A;
        std::vector<Eigen::Map<Eigen::ArrayXcd>> B;

        for (int k = 0; k < nThread; ++k)
        {
            A.push_back(Eigen::Map<Eigen::ArrayXcd>(VecA.data() + k * TASK[i], TASK[i]));
            B.push_back(Eigen::Map<Eigen::ArrayXcd>(VecB.data() + k * TASK[i], TASK[i]));
        }

        auto time_begin_wall = std::chrono::system_clock::now();
        auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
#pragma omp parallel for num_threads(nThread) schedule(dynamic)
        for (int j = 0; j < TestTime; ++j)
        {
            auto id = OMP_THREAD_LABEL;
            B[id] += C * A[id];
        }
        auto time_end_wall = std::chrono::system_clock::now();
        auto time_end_cpu = boost::chrono::process_cpu_clock::now();
        double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
        double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

        printf("vector_zaxpy_Eigen with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], duration_wall,
               duration_cpu, _get_GFLO(TASK[i], TestTime) * 1000 / duration_wall);
    }

    /// ISPC

    for (int i = 0; i < nTASK; ++i)
    {
        auto TestTime = _get_loop_time(TASK[i]) * nThread;
        std::vector<iCI_complex_double> VecA = GenerateRandomComplexDoubleVector(TASK[i] * nThread);
        std::vector<iCI_complex_double> VecB = GenerateRandomComplexDoubleVector(TASK[i] * nThread);

        iCI_complex_double *headB = VecB.data();
        iCI_complex_double C = GenerateRandomNumber(0.0);
        size_t size = VecB.size();
        memset(headB, 0, sizeof(iCI_complex_double) * size);

        /////// RUN the Test

        auto time_begin_wall = std::chrono::system_clock::now();
        auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
#pragma omp parallel for num_threads(nThread) schedule(dynamic)
        for (int j = 0; j < TestTime; ++j)
        {
            auto id = OMP_THREAD_LABEL;
            ispc::vector_zaxpy((double *)(headB + id * TASK[i]), (const double *)&C, (const double *)(VecA.data() + id * TASK[i]), TASK[i]);
        }
        auto time_end_wall = std::chrono::system_clock::now();
        auto time_end_cpu = boost::chrono::process_cpu_clock::now();
        double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
        double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

        printf("vector_zaxpy_ISPC with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], duration_wall,
               duration_cpu, _get_GFLO(TASK[i], TestTime) * 1000 / duration_wall);
    }

    /// MKL

    for (int i = 0; i < nTASK; ++i)
    {
        auto TestTime = _get_loop_time(TASK[i]) * nThread;
        std::vector<iCI_complex_double> VecA = GenerateRandomComplexDoubleVector(TASK[i] * nThread);
        std::vector<iCI_complex_double> VecB = GenerateRandomComplexDoubleVector(TASK[i] * nThread);

        iCI_complex_double *headB = VecB.data();
        iCI_complex_double C = GenerateRandomNumber(0.0);
        size_t size = VecB.size();
        memset(headB, 0, sizeof(iCI_complex_double) * size);

        /////// RUN the Test

        auto time_begin_wall = std::chrono::system_clock::now();
        auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
#pragma omp parallel for num_threads(nThread) schedule(dynamic)
        for (int j = 0; j < TestTime; ++j)
        {
            auto id = OMP_THREAD_LABEL;
            MathUtil::AXPY(TASK[i], C, VecA.data() + id * TASK[i], 1, headB + id * TASK[i], 1);
        }
        auto time_end_wall = std::chrono::system_clock::now();
        auto time_end_cpu = boost::chrono::process_cpu_clock::now();
        double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
        double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

        printf("vector_zaxpy_MKL with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], duration_wall,
               duration_cpu, _get_GFLO(TASK[i], TestTime) * 1000 / duration_wall);
    }
}