#include "../../utils/config.h"
#include "../../utils/random_generator.h"
#include "../../utils/MathUtilISPC.h"
#include "../../utils/math_util.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#define OneGOperation 1000000000ULL
#define FACTOR 4
#define FIX_SIZE 4

double vector_ddot_loop(double *A, double *B, int size)
{
    double Res = 0.0;
    for (int i = 0; i < size; ++i)
    {
        Res += A[i] * B[i];
    }
    return Res;
}

size_t _get_loop_time(size_t n)
{
    return OneGOperation / ( 2 * n * FACTOR);
}

double _get_GFLO(size_t n, size_t test_time)
{
    return (2 * double(n) * test_time) / OneGOperation;
}

int main(int argc, const char **argv)
{
    /// define the size of the problem

    // uint64_t TASK[]{4, 8, 16, 32, 64, 128, 256, 512,
    //                 1 * KB, 2 * KB, 4 * KB, 8 * KB, 16 * KB, 32 * KB,
    //                 64 * KB,
    //                 128 * KB,
    //                 256 * KB,
    //                 512 * KB,
    //                 1 * MB,
    //                 2 * MB, 4 * MB, 8 * MB, 16 * MB, 32 * MB};
    uint64_t TASK[]{4, 8, 16, 32,
                    64, 128, 256, 512,
                    1ULL << 10,
                    1ULL << 11,
                    1ULL << 12,
                    1ULL << 13,
                    1ULL << 14,
                    1ULL << 15,
                    1ULL << 16,
                    1ULL << 17,
                    1ULL << 18,
                    1ULL << 19,
                    1ULL << 20,
                    1ULL << 21,
                    1ULL << 22,
                    1ULL << 23,
                    1ULL << 24,
                    1ULL << 25};
    const int nTASK = sizeof(TASK) / sizeof(uint64_t);

    /* (2) plain loop */

    for (int i = 0; i < nTASK; ++i)
    {
        auto TestTime = _get_loop_time(TASK[i]);
        std::vector<double> VecA = GenerateRandomDoubleVector(TASK[i]);
        std::vector<double> VecB = GenerateRandomDoubleVector(TASK[i]);

        /////// RUN the Test

        auto time_begin_wall = std::chrono::system_clock::now();
        auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
        for (int j = 0; j < TestTime; ++j)
        {
            auto res = vector_ddot_loop(VecA.data(), VecB.data(), TASK[i]);
        }
        auto time_end_wall = std::chrono::system_clock::now();
        auto time_end_cpu = boost::chrono::process_cpu_clock::now();
        double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
        double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

        printf("vector_add_contant_loop with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], duration_wall,
               duration_cpu, _get_GFLO(TASK[i], TestTime) * 1000 / duration_wall);
    }

    /* (3) eigen with fix size */

    /// size 4

    auto TestTime = _get_loop_time(FIX_SIZE);
    std::vector<double> VecA = GenerateRandomDoubleVector(FIX_SIZE);
    std::vector<double> VecB = GenerateRandomDoubleVector(FIX_SIZE);

    Eigen::Map<Eigen::Vector4d> A(VecA.data());
    Eigen::Map<Eigen::Vector4d> B(VecB.data());

    /////// RUN the Test

    auto time_begin_wall = std::chrono::system_clock::now();
    auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
    printf("test_time = %d\n", TestTime);
    double res = 0.0;
    for (int j = 0; j < TestTime; ++j)
    {
        res += A.adjoint() * B;
    }
    printf("res = %f\n", res);
    auto time_end_wall = std::chrono::system_clock::now();
    auto time_end_cpu = boost::chrono::process_cpu_clock::now();
    double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
    double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

    printf("vector_ddot_Eigen_FixedSize with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", 4, duration_wall,
           duration_cpu, _get_GFLO(4, TestTime) * 1000 / duration_wall);

    /* ------------------------- UNROLL ------------------------- */

    std::vector<double> VecAA = GenerateRandomDoubleVector(FIX_SIZE);
    double *headAA = VecAA.data();

    time_begin_wall = std::chrono::system_clock::now();
    time_begin_cpu = boost::chrono::process_cpu_clock::now();
    printf("TestTime=%d\n", TestTime);
    res = 0.0;
    for (int j = 0; j < TestTime; ++j)
    {
        res += math_automatic_unroll<double, 4>::ddot(VecAA.data(), VecB.data());
    }
    printf("res = %f\n", res);
    time_end_wall = std::chrono::system_clock::now();
    time_end_cpu = boost::chrono::process_cpu_clock::now();
    duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
    duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

    printf("vector_ddot_unroll with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", 4, duration_wall,
           duration_cpu, _get_GFLO(4, TestTime) * 1000 / duration_wall);

    /// EigenMap

    double res_eigen = 0.0;

    for (int i = 0; i < nTASK; ++i)
    {
        auto TestTime = _get_loop_time(TASK[i]);
        std::vector<double> VecA = GenerateRandomDoubleVector(TASK[i]);
        std::vector<double> VecB = GenerateRandomDoubleVector(TASK[i]);

        /////// RUN the Test

        Eigen::Map<Eigen::VectorXd> A(VecA.data(), TASK[i]);
        Eigen::Map<Eigen::VectorXd> B(VecB.data(), TASK[i]);

        auto time_begin_wall = std::chrono::system_clock::now();
        auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
        double res = 0.0;
        for (int j = 0; j < TestTime; ++j)
        {
            res += A.adjoint() * B;
        }
        res_eigen += res;
        auto time_end_wall = std::chrono::system_clock::now();
        auto time_end_cpu = boost::chrono::process_cpu_clock::now();
        double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
        double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

        printf("vector_ddot_Eigen with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], duration_wall,
               duration_cpu, _get_GFLO(TASK[i], TestTime) * 1000 / duration_wall);
    }

    printf("res_eigen = %f\n", res_eigen);

    /// ISPC

    for (int i = 0; i < nTASK; ++i)
    {
        auto TestTime = _get_loop_time(TASK[i]);
        std::vector<double> VecA = GenerateRandomDoubleVector(TASK[i]);
        std::vector<double> VecB = GenerateRandomDoubleVector(TASK[i]);

        /////// RUN the Test

        auto time_begin_wall = std::chrono::system_clock::now();
        auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
        for (int j = 0; j < TestTime; ++j)
        {
            auto res = ispc::vector_dot((double *)VecA.data(), (double *)VecB.data(), TASK[i]);
        }
        auto time_end_wall = std::chrono::system_clock::now();
        auto time_end_cpu = boost::chrono::process_cpu_clock::now();
        double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
        double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

        printf("vector_ddot_ispc with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], duration_wall,
               duration_cpu, _get_GFLO(TASK[i], TestTime) * 1000 / duration_wall);
    }

    /// MKL

    for (int i = 0; i < nTASK; ++i)
    {
        auto TestTime = _get_loop_time(TASK[i]);
        std::vector<double> VecA = GenerateRandomDoubleVector(TASK[i]);
        std::vector<double> VecB = GenerateRandomDoubleVector(TASK[i]);

        /////// RUN the Test

        auto time_begin_wall = std::chrono::system_clock::now();
        auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
        for (int j = 0; j < TestTime; ++j)
        {
            auto res = MathUtil::DOT(TASK[i], VecA.data(), 1, VecB.data(), 1);
        }
        auto time_end_wall = std::chrono::system_clock::now();
        auto time_end_cpu = boost::chrono::process_cpu_clock::now();
        double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
        double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

        printf("vector_ddot_MKL with size %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], duration_wall,
               duration_cpu, _get_GFLO(TASK[i], TestTime) * 1000 / duration_wall);
    }
}
