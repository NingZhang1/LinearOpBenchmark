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
LDFALGS := -pthread $(OMP) -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
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

EigenHelloWorld: 01-EigenHelloWorld.o $(COMMONUTIL_O)
	$(CXX) -std=$(CXX_STANDATD) 01-EigenHelloWorld.o $(COMMONUTIL_O) $(LDFLAGS) $(DEBUGGING) -o 01-EigenHelloWorld.exe $(LDFALGS)

Mat_and_Vec: 02-Mat_and_Vec.o $(COMMONUTIL_O)
	$(CXX) -std=$(CXX_STANDATD) 02-Mat_and_Vec.o $(COMMONUTIL_O) $(LDFLAGS) $(DEBUGGING) -o 02-Mat_and_Vec.exe $(LDFALGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm *.o
	rm *.exe