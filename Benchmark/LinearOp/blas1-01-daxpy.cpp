#include "../../utils/config.h"
#include "../../utils/random_generator.h"
#include "../../utils/MathUtilISPC.h"
#include "../../utils/math_util.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#define OneGOperation 1000000000ULL
#define FACTOR 4

// test with different size of problem, while with different stride !

void vector_daxpy_loop(const double *a, const double alpha, double *b, const int32_t size)
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
    /// define the size of the problem

    uint64_t TASK[]{4, 8, 16, 32, 64, 128, 256, 512,
                    1 * KB, 2 * KB, 4 * KB, 8 * KB, 16 * KB, 32 * KB,
                    64 * KB,
                    128 * KB,
                    256 * KB,
                    512 * KB,
                    1 * MB,
                    2 * MB, 4 * MB, 8 * MB, 16 * MB, 32 * MB};
    const int nTASK = sizeof(TASK) / sizeof(uint64_t);

    /* (2) plain loop */

    for (int i = 0; i < nTASK; ++i)
    {
        auto TestTime = _get_loop_time(TASK[i]);
        std::vector<double> VecA = GenerateRandomDoubleVector(TASK[i]);
        std::vector<double> VecB = GenerateRandomDoubleVector(TASK[i]);

        double *headB = VecB.data();
        double B = GenerateRandomNumber(0.0);
        size_t size = VecB.size();
        memset(headB, 0, sizeof(double) * size);

        /////// RUN the Test

        auto time_begin_wall = std::chrono::system_clock::now();
        auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
        for (int j = 0; j < TestTime; ++j)
        {
            vector_daxpy_loop(VecA.data(), B, VecB.data(), size);
        }
        auto time_end_wall = std::chrono::system_clock::now();
        auto time_end_cpu = boost::chrono::process_cpu_clock::now();
        double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
        double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

        printf("vector_daxpy_loop with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], duration_wall,
               duration_cpu, _get_GFLO(VecA.size(), TestTime) * 1000 / duration_wall);
    }

    /* (3) eigen with fix size */

    /// size 4

    auto TestTime = _get_loop_time(4);
    std::vector<double> VecA = GenerateRandomDoubleVector(4);
    std::vector<double> VecB = GenerateRandomDoubleVector(4);

    double *headB = VecB.data();
    double C = GenerateRandomNumber(0.0);
    size_t size = VecB.size();
    memset(headB, 0, sizeof(double) * size);

    Eigen::Map<Eigen::Array4d> A(VecA.data());
    Eigen::Map<Eigen::Array4d> B(headB);

    /////// RUN the Test

    auto time_begin_wall = std::chrono::system_clock::now();
    auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
    for (int j = 0; j < TestTime; ++j)
    {
        B += A * C;
    }
    auto time_end_wall = std::chrono::system_clock::now();
    auto time_end_cpu = boost::chrono::process_cpu_clock::now();
    double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
    double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

    printf("vector_daxpy_FixedSize with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", 4, duration_wall,
           duration_cpu, _get_GFLO(4, TestTime) * 1000 / duration_wall);

    /* ------------------------- UNROLL ------------------------- */

    std::vector<double> VecAA(4);
    double *headAA = VecAA.data();
    memset(headAA, 0, sizeof(double) * size);

    time_begin_wall = std::chrono::system_clock::now();
    time_begin_cpu = boost::chrono::process_cpu_clock::now();
    printf("TestTime=%d\n", TestTime);
    for (int j = 0; j < TestTime; ++j)
    {
        math_automatic_unroll<double, 4>::y_equal_ax_plus_y(C, VecA.data(), headAA);
    }
    printf("headAA[0]=%f\n", headAA[0]);
    printf("headAA[1]=%f\n", headAA[1]);
    printf("headAA[2]=%f\n", headAA[2]);
    printf("headAA[3]=%f\n", headAA[3]);
    time_end_wall = std::chrono::system_clock::now();
    time_end_cpu = boost::chrono::process_cpu_clock::now();
    duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
    duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

    printf("vector_daxpy_unroll with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", 4, duration_wall,
           duration_cpu, _get_GFLO(4, TestTime) * 1000 / duration_wall);

    /// EigenMap

    for (int i = 0; i < nTASK; ++i)
    {
        auto TestTime = _get_loop_time(TASK[i]);
        std::vector<double> VecA = GenerateRandomDoubleVector(TASK[i]);
        std::vector<double> VecB = GenerateRandomDoubleVector(TASK[i]);

        double *headB = VecB.data();
        double C = GenerateRandomNumber(0.0);
        size_t size = VecB.size();
        memset(headB, 0, sizeof(double) * size);

        /////// RUN the Test

        Eigen::Map<Eigen::ArrayXd> A(VecA.data(), size);
        Eigen::Map<Eigen::ArrayXd> B(headB, size);

        auto time_begin_wall = std::chrono::system_clock::now();
        auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
        for (int j = 0; j < TestTime; ++j)
        {
            B += C * A;
        }
        auto time_end_wall = std::chrono::system_clock::now();
        auto time_end_cpu = boost::chrono::process_cpu_clock::now();
        double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
        double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

        printf("vector_daxpy_Eigen with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], duration_wall,
               duration_cpu, _get_GFLO(TASK[i], TestTime) * 1000 / duration_wall);
    }

    /// ISPC

    for (int i = 0; i < nTASK; ++i)
    {
        auto TestTime = _get_loop_time(TASK[i]);
        std::vector<double> VecA = GenerateRandomDoubleVector(TASK[i]);
        std::vector<double> VecB = GenerateRandomDoubleVector(TASK[i]);

        double *headB = VecB.data();
        double C = GenerateRandomNumber(0.0);
        size_t size = VecB.size();
        memset(headB, 0, sizeof(double) * size);

        /////// RUN the Test

        auto time_begin_wall = std::chrono::system_clock::now();
        auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
        for (int j = 0; j < TestTime; ++j)
        {
            ispc::vector_daxpy(headB, C, VecA.data(), TASK[i]);
        }
        auto time_end_wall = std::chrono::system_clock::now();
        auto time_end_cpu = boost::chrono::process_cpu_clock::now();
        double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
        double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

        printf("vector_daxpy_ISPC with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], duration_wall,
               duration_cpu, _get_GFLO(TASK[i], TestTime) * 1000 / duration_wall);
    }

    /// MKL

    for (int i = 0; i < nTASK; ++i)
    {
        auto TestTime = _get_loop_time(TASK[i]);
        std::vector<double> VecA = GenerateRandomDoubleVector(TASK[i]);
        std::vector<double> VecB = GenerateRandomDoubleVector(TASK[i]);

        double *headB = VecB.data();
        double C = GenerateRandomNumber(0.0);
        size_t size = VecB.size();
        memset(headB, 0, sizeof(double) * size);

        /////// RUN the Test

        auto time_begin_wall = std::chrono::system_clock::now();
        auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
        for (int j = 0; j < TestTime; ++j)
        {
            MathUtil::AXPY(TASK[i], C, VecA.data(), 1, headB, 1);
        }
        auto time_end_wall = std::chrono::system_clock::now();
        auto time_end_cpu = boost::chrono::process_cpu_clock::now();
        double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
        double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

        printf("vector_daxpy_MKL with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], duration_wall,
               duration_cpu, _get_GFLO(TASK[i], TestTime) * 1000 / duration_wall);
    }
}