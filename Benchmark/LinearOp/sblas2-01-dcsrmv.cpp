#include "../../utils/config.h"
#include "../../utils/random_generator.h"
#include "../../utils/MathUtilISPC.h"
#include "../../utils/math_util.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#define OneGOperation 1000000000ULL
#define FACTOR 4

using SparseMat_t = Eigen::SparseMatrix<double, Eigen::RowMajor, MKL_INT>;
using triplet_t = Eigen::Triplet<double>;

SparseMat_t getSparseMat(int nRow, int nCol, double density)
{
    std::vector<triplet_t> triplets;

    // triplets.reserve(nRow * nCol * density);

    auto nnz = nRow * nCol * density;

    for (int k = 0; k < nnz; ++k)
    {
        triplets.push_back(triplet_t(rand() % nRow, rand() % nCol, GenerateRandomNumber<double>(0.0)));
    }

    SparseMat_t res(nRow, nCol);
    res.setFromTriplets(triplets.begin(), triplets.end());
    return res;
}

struct Statistic
{

    Statistic(
        int _nRow,
        int _nCol,
        int _nNnz,
        int _test_time,
        double _Eigen_time,
        double _MKL_time)
        : nRow(_nRow),
          nCol(_nCol),
          nNnz(_nNnz),
          test_time(_test_time),
          Eigen_time(_Eigen_time),
          MKL_time(_MKL_time)
    {
        flops = _nNnz * 2.0 * _test_time;
    }

    int nRow;
    int nCol;
    int nNnz;
    int test_time;
    double Eigen_time;
    double MKL_time;
    double flops;
};

int main(int argc, const char **argv)
{
    Eigen::setNbThreads(1); /// disable multi-threading

    /// define the size of the problem

    uint64_t TASK[]{
        1ULL << 14,
        1ULL << 15,
        1ULL << 16,
        1ULL << 17,
        1ULL << 18,
        1ULL << 19,
        1ULL << 20
    };

    double DENSITY[]{
        1e-4, 3e-4, 1e-3, 3e-3};

    const int nTASK = sizeof(TASK) / sizeof(uint64_t);
    const int nDENSITY = sizeof(DENSITY) / sizeof(double);

    ////

    std::vector<Statistic> Stat;

    for (int row_id = 0; row_id < nTASK; row_id++)
    {
        for (int col_id = 0; col_id < nTASK; col_id++)
        {
            for (int density_id = 0; density_id < nDENSITY; density_id++)
            {
                /// generate test data

                MKL_INT nRow = TASK[row_id];
                MKL_INT nCol = TASK[col_id];
                double density = DENSITY[density_id];

                SparseMat_t mat = getSparseMat(nRow, nCol, density);

                Eigen::VectorXd vec = Eigen::VectorXd::Random(nCol);
                Eigen::VectorXd res = Eigen::VectorXd::Zero(nRow);

                /// get the raw ptr to call mkl

                auto *RowIndx = mat.outerIndexPtr();
                auto *ColIndx = mat.innerIndexPtr();
                auto *Value = mat.valuePtr();
                auto nnz = mat.nonZeros();
                auto nFLOPS = nnz * 2.0;
                int testtime = (double)OneGOperation / (nFLOPS * FACTOR);

                if (testtime < 8)
                {
                    continue;
                }

                /// run the test

                auto time_begin_wall = std::chrono::system_clock::now();
                auto time_begin_cpu = boost::chrono::process_cpu_clock::now();

                for (int i = 0; i < testtime; ++i)
                {
                    MathUtil::CSRMV(&nRow, &nCol, Value, ColIndx, RowIndx, RowIndx + 1, vec.data(), res.data());
                }

                auto time_end_wall = std::chrono::system_clock::now();
                auto time_end_cpu = boost::chrono::process_cpu_clock::now();
                double duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall) / 1000;
                double duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu) / 1000;

                double duration_mkl = duration_wall;

                time_begin_wall = std::chrono::system_clock::now();
                time_begin_cpu = boost::chrono::process_cpu_clock::now();

                for (int i = 0; i < testtime; ++i)
                {
                    res += mat * vec;
                }

                time_end_wall = std::chrono::system_clock::now();
                time_end_cpu = boost::chrono::process_cpu_clock::now();
                duration_wall = get_duration_in_ms(time_begin_wall, time_end_wall) / 1000;
                duration_cpu = get_duration_in_ms(time_begin_cpu, time_end_cpu) / 1000;

                double duration_eigen = duration_wall;

                Stat.push_back(Statistic(nRow, nCol, nnz, testtime, duration_eigen, duration_mkl));

                std::cout << "nRow = " << nRow << " nCol = " << nCol << " nnz = " << nnz << " testtime = " << testtime << std::endl;
            }
        }
    }

    /// print the result

    for (auto &stat : Stat)
    {
        printf("nRow = %10d nCol = %10d nnz = %10d Eigen_time = %10.3f MKL_time = %10.3f Eigen_FLOPS = %10.3f MKL_FLOPS = %10.3f\n",
               stat.nRow, stat.nCol, stat.nNnz, stat.Eigen_time, stat.MKL_time, (stat.flops / stat.Eigen_time) / OneGOperation, (stat.flops / stat.MKL_time) / OneGOperation);
    }
}