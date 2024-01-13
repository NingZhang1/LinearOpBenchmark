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

all: CoeffAccessor StorageOrders ArithmeticAddSub ArithmeticScalar ArithmeticConj ArithmeticMultiply ArithmeticReduce ArrayArithmetic ArrayArithmetic2 ArrayMatrixConversion BlockOpMatrix BlockOpArray BlockOpCorner Reduction Visitors Broadcasting Broadcasting2 MapClass MapClass2

CoeffAccessor: 01-CoeffAccessor.o 
	$(CXX) -std=$(CXX_STANDATD) 01-CoeffAccessor.o  $(LDFLAGS) $(DEBUGGING) -o 01-CoeffAccessor.exe $(LDFALGS)

StorageOrders: 02-StorageOrders.o 
	$(CXX) -std=$(CXX_STANDATD) 02-StorageOrders.o  $(LDFLAGS) $(DEBUGGING) -o 02-StorageOrders.exe $(LDFALGS)

ArithmeticAddSub: 03-01-Arithmetic-Add_Sub.o 
	$(CXX) -std=$(CXX_STANDATD) 03-01-Arithmetic-Add_Sub.o  $(LDFLAGS) $(DEBUGGING) -o 03-01-Arithmetic-Add_Sub.exe $(LDFALGS)

ArithmeticScalar: 03-02-Arithmetic-Scalar.o 
	$(CXX) -std=$(CXX_STANDATD) 03-02-Arithmetic-Scalar.o  $(LDFLAGS) $(DEBUGGING) -o 03-02-Arithmetic-Scalar.exe $(LDFALGS)

ArithmeticConj: 03-03-Arithmetic-Conj_T.o 
	$(CXX) -std=$(CXX_STANDATD) 03-03-Arithmetic-Conj_T.o  $(LDFLAGS) $(DEBUGGING) -o 03-03-Arithmetic-Conj_T.exe $(LDFALGS)

ArithmeticMultiply: 03-04-Arithmetic-Multi.o 
	$(CXX) -std=$(CXX_STANDATD) 03-04-Arithmetic-Multi.o  $(LDFLAGS) $(DEBUGGING) -o 03-04-Arithmetic-Multi.exe $(LDFALGS)

ArithmeticReduce: 03-05-Arithmetic-Reduce.o 
	$(CXX) -std=$(CXX_STANDATD) 03-05-Arithmetic-Reduce.o  $(LDFLAGS) $(DEBUGGING) -o 03-05-Arithmetic-Reduce.exe $(LDFALGS)

ArrayArithmetic: 04-01-Array-Arithmetic.o 
	$(CXX) -std=$(CXX_STANDATD) 04-01-Array-Arithmetic.o  $(LDFLAGS) $(DEBUGGING) -o 04-01-Array-Arithmetic.exe $(LDFALGS)

ArrayArithmetic2: 04-02-Array-Arithmetic2.o 
	$(CXX) -std=$(CXX_STANDATD) 04-02-Array-Arithmetic2.o  $(LDFLAGS) $(DEBUGGING) -o 04-02-Array-Arithmetic2.exe $(LDFALGS)

ArrayMatrixConversion: 04-03-Array-Matrix-Conversion.o
	$(CXX) -std=$(CXX_STANDATD) 04-03-Array-Matrix-Conversion.o  $(LDFLAGS) $(DEBUGGING) -o 04-03-Array-Matrix-Conversion.exe $(LDFALGS)

BlockOpMatrix: 05-01-BlockOp-Matrix.o
	$(CXX) -std=$(CXX_STANDATD) 05-01-BlockOp-Matrix.o  $(LDFLAGS) $(DEBUGGING) -o 05-01-BlockOp-Matrix.exe $(LDFALGS)

BlockOpArray: 05-02-BlockOp-Array.o
	$(CXX) -std=$(CXX_STANDATD) 05-02-BlockOp-Array.o  $(LDFLAGS) $(DEBUGGING) -o 05-02-BlockOp-Array.exe $(LDFALGS)

BlockOpCorner: 05-03-BlockOp-Corner.o
	$(CXX) -std=$(CXX_STANDATD) 05-03-BlockOp-Corner.o  $(LDFLAGS) $(DEBUGGING) -o 05-03-BlockOp-Corner.exe $(LDFALGS)

Reduction: 06-01-Reduction.o
	$(CXX) -std=$(CXX_STANDATD) 06-01-Reduction.o  $(LDFLAGS) $(DEBUGGING) -o 06-01-Reduction.exe $(LDFALGS)

Visitors: 06-02-Visitors.o
	$(CXX) -std=$(CXX_STANDATD) 06-02-Visitors.o  $(LDFLAGS) $(DEBUGGING) -o 06-02-Visitors.exe $(LDFALGS)

Broadcasting: 06-03-Broadcasting.o
	$(CXX) -std=$(CXX_STANDATD) 06-03-Broadcasting.o  $(LDFLAGS) $(DEBUGGING) -o 06-03-Broadcasting.exe $(LDFALGS)

Broadcasting2: 06-04-Broadcasting2.o
	$(CXX) -std=$(CXX_STANDATD) 06-04-Broadcasting2.o  $(LDFLAGS) $(DEBUGGING) -o 06-04-Broadcasting2.exe $(LDFALGS)

MapClass: 07-01-MapClass.o
	$(CXX) -std=$(CXX_STANDATD) 07-01-MapClass.o  $(LDFLAGS) $(DEBUGGING) -o 07-01-MapClass.exe $(LDFALGS)

MapClass2: 07-02-MapClass2.o
	$(CXX) -std=$(CXX_STANDATD) 07-02-MapClass2.o  $(LDFLAGS) $(DEBUGGING) -o 07-02-MapClass2.exe $(LDFALGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm *.o
	rm *.exe