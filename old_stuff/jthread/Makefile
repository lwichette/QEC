CUDA_HOME = /usr/local/cuda
CUDACC= $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS=-std=c++20 -O3 -lineinfo -arch=sm_70 -Xptxas=-v -I../external/cub -I header
LD=$(CUDACC)
LDFLAGS=-lcurand

all: output

output: parallelization.cu
	$(CUDACC) $(NVCC_FLAGS) header/utils.cu parallelization.cu -o test $(LDFLAGS)

clean:
	rm -f test
