#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "../../utils/random_generator.h"

using SparseMat_t = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using triplet_t = Eigen::Triplet<double>;

SparseMat_t getSparseMat(int nRow, int nCol, double density)
{
    std::vector<triplet_t> triplets;
    triplets.reserve(nRow * nCol * density);
    for (int i = 0; i < nRow; ++i)
    {
        for (int j = 0; j < nCol; ++j)
        {
            if (rand() / (double)(RAND_MAX) < density)
            {
                triplets.push_back(triplet_t(i, j, GenerateRandomNumber<double>(0.0)));
            }
        }
    }
    SparseMat_t res(nRow, nCol);
    res.setFromTriplets(triplets.begin(), triplets.end());
    return res;
}

#define NROW 16
#define NCOL 16

int main()
{
    srand(time(NULL));

    /// generate random sparse matrix and print

    SparseMat_t mat = getSparseMat(NROW, NCOL, 0.2);

    for (int k = 0; k < mat.outerSize(); ++k) /// outerSize() == # of rows
    {
        std::cout << "row " << k << std::endl;
        for (SparseMat_t::InnerIterator it(mat, k); it; ++it)
        {
            std::cout << it.row() << " " << it.col() << " " << it.value() << std::endl;
        }
    }

    /// generate random dense vec and print

    Eigen::VectorXd vec = Eigen::VectorXd::Random(NCOL);

    std::cout << vec << std::endl;

    /// do GEMV

    Eigen::VectorXd res = mat * vec;

    std::cout << res << std::endl;

    /// test map external buffer

    auto *RowIndx = mat.outerIndexPtr();
    auto *ColIndx = mat.innerIndexPtr();
    auto *Value = mat.valuePtr();

    for (int i = 0; i < NROW; ++i)
    {
        std::cout << "row " << i << std::endl;
        for (int j = RowIndx[i]; j < RowIndx[i + 1]; ++j)
        {
            std::cout << i << " " << ColIndx[j] << " " << Value[j] << std::endl;
        }
    }

    /// construct a map

    Eigen::Map<const SparseMat_t> SparseMatMap(
        NROW,
        NCOL,
        mat.nonZeros(),
        RowIndx,
        ColIndx,
        Value);

    /// do GEMV

    Eigen::VectorXd res2 = SparseMatMap * vec;

    std::cout << res2 << std::endl;
}