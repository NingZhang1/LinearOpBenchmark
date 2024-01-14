CXX := g++
OMP = -fopenmp
OPTLEVEL = -O3
# CXX := icpx
# OMP = -qopenmp
# ICPC_OPT_REPORT := -qopt-report -qopt-report-phase=vec 
EIGEN= /usr/local/include/Eigen
MKLROOT= /opt/intel/oneapi/mkl/latest
CXX_WARNING_OPTIONS := -Wall -Wextra -Wno-unused-variable -Wno-unused-function $(ICPC_OPT_REPORT) 
CBLASROOT := /home/nzhangcaltech/lapack-3.11.0/CBLAS
LAPACKROOT := /home/nzhangcaltech/lapack-3.11.0/LAPACKE
LIBROOT := /home/nzhangcaltech/lapack-3.11.0/

# CXX_STANDATD = c++11
# CXX_STANDATD = c++14
CXX_STANDATD = c++17

# with mkl 
CXXFLAGS := -std=$(CXX_STANDATD) $(OPTLEVEL)  $(OMP) $(CXX_WARNING_OPTIONS) -DMKL_ILP64 -I${MKLROOT}/include -D _MKL_ 
LDFALGS := -pthread $(OMP) -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lboost_chrono
# without mkl 
# CXXFLAGS := -std=$(CXX_STANDATD) $(OPTLEVEL)  $(OMP) $(CXX_WARNING_OPTIONS) -I${CBLASROOT}/include -I${LAPACKROOT}/include
# LDFALGS := -pthread $(OMP) -L${LIBROOT}  -llapacke -llapack -lcblas -lblas -lgfortran -lpthread -lm -ldl
# LDFALGS := -pthread $(OMP) -L${LIBROOT}  -llapacke -llapack -lcblas -lrefblas -lgfortran -lpthread -lm -ldl

LIB_DIR := ./lib 
# CXXFLAGS := $(CXXFLAGS) -I$(LIB_DIR) -I$(EIGEN) -I./ -g 
# CXXFLAGS := $(CXXFLAGS) -I$(LIB_DIR) -I$(EIGEN) -I./ -g -fPIC
DEBUGGING = -g
# DEBUGGING =
CXXFLAGS := $(CXXFLAGS) -I$(LIB_DIR) -I$(EIGEN) -I./ $(DEBUGGING) -fPIC

ISPC_O := ../../utils/MathUtilISPC.o

all: blas1daxpy blas1zaxpy blas1daxpc blas1ddot blas1zdot blas1ddot2

blas1daxpy: blas1-01-daxpy.o $(COMMONUTIL_O)
	$(CXX) -std=$(CXX_STANDATD) blas1-01-daxpy.o $(COMMONUTIL_O) $(ISPC_O) $(LDFLAGS) $(DEBUGGING) -o blas1-01-daxpy.exe $(LDFALGS)

blas1zaxpy: blas1-02-zaxpy.o $(COMMONUTIL_O)
	$(CXX) -std=$(CXX_STANDATD) blas1-02-zaxpy.o $(COMMONUTIL_O) $(ISPC_O) $(LDFLAGS) $(DEBUGGING) -o blas1-02-zaxpy.exe $(LDFALGS)

blas1daxpc: blas1-03-daxpc.o $(COMMONUTIL_O)
	$(CXX) -std=$(CXX_STANDATD) blas1-03-daxpc.o $(COMMONUTIL_O) $(ISPC_O) $(LDFLAGS) $(DEBUGGING) -o blas1-03-daxpc.exe $(LDFALGS)

blas1ddot: blas1-04-ddot.o $(COMMONUTIL_O)
	$(CXX) -std=$(CXX_STANDATD) blas1-04-ddot.o $(COMMONUTIL_O) $(ISPC_O) $(LDFLAGS) $(DEBUGGING) -o blas1-04-ddot.exe $(LDFALGS)

blas1ddot2: blas1-06-ddot2.o $(COMMONUTIL_O)
	$(CXX) -std=$(CXX_STANDATD) blas1-06-ddot2.o $(COMMONUTIL_O) $(ISPC_O) $(LDFLAGS) $(DEBUGGING) -o blas1-06-ddot2.exe $(LDFALGS)

blas1zdot: blas1-05-zdot.o $(COMMONUTIL_O)
	$(CXX) -std=$(CXX_STANDATD) blas1-05-zdot.o $(COMMONUTIL_O) $(ISPC_O) $(LDFLAGS) $(DEBUGGING) -o blas1-05-zdot.exe $(LDFALGS)


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm *.o
	rm *.exe