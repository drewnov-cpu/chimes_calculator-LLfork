CC_LOC = $(realpath .)

CXX=nvcc -O3 -arch=sm_70 -rdc=true --compiler-options -fPIC

CHIMESFF_SRC=$(CC_LOC)/../../src/chimesFF.cu $(CC_LOC)/../../src/gpu_compute.cu
CHIMESFF_HDR=$(CC_LOC)/../../src/chimesFF.h $(CC_LOC)/../../src/gpu_compute.h

chimesFF.o : $(CHIMESFF_SRC)
	$(CXX) -c $(CHIMESFF_SRC)

CC_WRAPPER_SRC=$(CC_LOC)/../../api/chimescalc_C.cpp
CC_WRAPPER_HDR=$(CC_LOC)/../../api/chimescalc_C.h

chimescalc_C.o : $(CC_WRAPPER_SRC) $(CC_WRAPPER_HDR) $(CHIMESFF_SRC)  $(CHIMESFF_HDR)
	$(CXX) -c $(CC_WRAPPER_SRC) $(CHIMESFF_SRC) -I $(CC_LOC)/../../api/ -I $(CC_LOC)/../../src/  
	
chimescalc_C.so:
	$(CXX) -shared -o libchimescalc-direct_dl.so chimescalc_C.o chimesFF.o

	
clean:
	rm -f *.o

clean-all:

	make clean
	rm -f *.so

all:
	make chimesFF.o
	make chimescalc_C.o
	make chimescalc_C.so
	make clean
