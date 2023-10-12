CUDA_HOME = /usr/local/cuda
CUDACC= $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS=-std=c++17 -O3 -lineinfo -arch=sm_70 -Xptxas=-v -I../external/cub -I ../header
LD=$(CUDACC)
LDFLAGS=-lcurand

all: output

output: nishimori.cu
	$(CUDACC) $(NVCC_FLAGS) linearized_big.cu -o test $(LDFLAGS)

clean:
	rm -f test