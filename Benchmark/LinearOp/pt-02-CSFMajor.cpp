#include "../../utils/config.h"
#include "../../utils/random_generator.h"
#include "../../utils/MathUtilISPC.h"
#include "../../utils/math_util.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#define OneGOperation 1000000000ULL
#define FACTOR 4
#define FIX_SIZE 4

void Accumulate_Ept(const double *Sigma, const double *Dg, const double *eCI, double *ept, int ncsf, int nstate)
{
    const double *ptr = Sigma;
    for (int icsf = 0; icsf < ncsf; ++icsf)
    {
        const double dg = Dg[icsf];
        for (int istate = 0; istate < nstate; ++istate, ptr++)
        {

            ept[istate] += *ptr * *ptr / (eCI[istate] - dg);
        }
    }
}

void Accumulate_Ept_ISDF(const double *Sigma, const double *Dg, const double *eCI, double *ept, int ncsf, int nstate)
{
    const double *ptr = Sigma;
    for (int icsf = 0; icsf < ncsf; ++icsf, ptr += nstate)
    {
        const double dg = Dg[icsf];
        // for (int istate = 0; istate < nstate; ++istate, ptr++)
        // {
        //     ept[istate] += *ptr * *ptr / (eCI[istate] - dg);
        // }
        ispc::AccumulateEptCSFMajorKernel(ptr, dg, eCI, nstate, ept);
    }
}

size_t _get_loop_time(int ncsf, int nstate)
{
    return OneGOperation / (ncsf * nstate * 3 * FACTOR);
}

double _get_GFLO(int ncsf, int nstate, size_t test_time)
{
    return (ncsf * nstate * 3 * test_time) / OneGOperation;
}

int main(int argc, const char **argv)
{
    /// define the size of the problem

    uint64_t TASK[]{1,
                    4,
                    32,
                    128,
                    512,
                    1ULL << 11,
                    1ULL << 13,
                    1ULL << 15,
                    1ULL << 17,
                    1ULL << 19,
                    1ULL << 20}; // the size of ocfg
    uint64_t NSTATE[]{1, 4, 8, 16, 32};
    const int nTASK = sizeof(TASK) / sizeof(uint64_t);

    const size_t LENGTH_VEC = 1ULL << 25;
    std::vector<double> VecSigma = GenerateRandomDoubleVector(LENGTH_VEC);
    std::vector<double> VecDg = GenerateRandomDoubleVector(LENGTH_VEC);
    std::vector<double> ene = GenerateRandomDoubleVector(32);
    std::vector<double> ept = GenerateRandomDoubleVector(32);
    const int NLOOP = 8;

    /* (2) plain loop */

    for (int i = 0; i < nTASK; ++i)
    {
        for (int i_nstate = 0; i_nstate < 5; ++i_nstate)
        {
            const int nstate = NSTATE[i_nstate];
            const int ncsf = TASK[i];

            auto nInnerLoop = LENGTH_VEC / (ncsf * nstate);

            /////// RUN the Test

            auto time_begin_wall = std::chrono::system_clock::now();
            auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
            for (int j = 0; j < NLOOP; ++j)
            {
                for (int k = 0; k < nInnerLoop; ++k)
                {
                    Accumulate_Ept(VecSigma.data() + k * ncsf * nstate, VecDg.data() + k * ncsf, ene.data(), ept.data(), ncsf, nstate);
                }
            }
            auto time_end_wall = std::chrono::system_clock::now();
            auto time_end_cpu = boost::chrono::process_cpu_clock::now();
            double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
            double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

            const double FLOP_TOT = double(ncsf * nstate * nInnerLoop * NLOOP * 3) / OneGOperation;
            // printf("FLOP_TOT = %f\n", FLOP_TOT);

            printf("Accumulate_Ept with ncsf %10d nstate %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], nstate, duration_wall,
                   duration_cpu, FLOP_TOT * 1000 / duration_wall);
        }
    }

    /// ISPC

    for (int i = 0; i < nTASK; ++i)
    {
        for (int i_nstate = 0; i_nstate < 5; ++i_nstate)
        {
            const int nstate = NSTATE[i_nstate];
            const int ncsf = TASK[i];

            auto nInnerLoop = LENGTH_VEC / (ncsf * nstate);

            /////// RUN the Test

            auto time_begin_wall = std::chrono::system_clock::now();
            auto time_begin_cpu = boost::chrono::process_cpu_clock::now();
            for (int j = 0; j < NLOOP; ++j)
            {
                for (int k = 0; k < nInnerLoop; ++k)
                {
                    Accumulate_Ept_ISDF(VecSigma.data() + k * ncsf * nstate, VecDg.data() + k * ncsf, ene.data(), ept.data(), ncsf, nstate);
                }
            }
            auto time_end_wall = std::chrono::system_clock::now();
            auto time_end_cpu = boost::chrono::process_cpu_clock::now();
            double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall);
            double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu);

            const double FLOP_TOT = double(ncsf * nstate * nInnerLoop * NLOOP * 3) / OneGOperation;
            // printf("FLOP_TOT = %f\n", FLOP_TOT);

            printf("Accumulate_Ept_ISPC with ncsf %10d nstate %10d wall time %12.3f ms cputime %12.3f FLOPS %12.3f \n", TASK[i], nstate, duration_wall,
                   duration_cpu, FLOP_TOT * 1000 / duration_wall);
        }
    }
}
