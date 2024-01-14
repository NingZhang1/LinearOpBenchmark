#include "../../utils/config.h"
#include "../../utils/random_generator.h"
#include "../../utils/MathUtilISPC.h"
#include "../../utils/math_util.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#define OneGOperation 1000000000ULL

int main(int argc, const char **argv)
{
    /// define the size of the problem

    uint64_t TASK[]{
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
        1ULL << 25,
        1ULL << 26,
        1ULL << 27,
        1ULL << 28};

    double DENSITY[]{
        1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2};

    const int nTASK = sizeof(TASK) / sizeof(uint64_t);
    const int nDENSITY = sizeof(DENSITY) / sizeof(double);


}