CUDA_HOME = /usr/local/cuda
CUDACC= $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS=-std=c++17 -O3 -lineinfo -arch=sm_70 -Xptxas=-v -I../external/cub -I ../header
LD=$(CUDACC)
LDFLAGS=-lcurand

all: output

output: open_boundary.cu
	$(CUDACC) $(NVCC_FLAGS) ../header/utils_big.cu open_boundary.cu -o open_boundary $(LDFLAGS)

clean:
	rm -f open_boundary
