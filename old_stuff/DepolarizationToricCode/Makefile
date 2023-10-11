CUDA_HOME=/usr/local/cuda
CUDACC=$(CUDA_HOME)/bin/nvcc
NVCC_FLAGS=-std=c++11 -O3 -lineinfo -arch=sm_70 -Xptxas=-v -I../external/cub
LD=$(CUDACC)
LDFLAGS=-lcurand

all: DepolarizationToricCode

DepolarizationToricCode: DepolarizationToricCode.cu
	$(CUDACC) $(NVCC_FLAGS) DepolarizationToricCode.cu -o DepolarizationToricCode $(LDFLAGS)

clean:
	rm -f DepolarizationToricCode
