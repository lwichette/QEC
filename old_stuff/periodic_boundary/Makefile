CUDA_HOME = /usr/local/cuda
CUDACC= $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS=-std=c++17 -O3 -lineinfo -arch=sm_70 -Xptxas=-v -I ../external/cub -I ../header -I /usr/share/doc
LD=$(CUDACC)
LDFLAGS=-lcurand
LBOOSTFLAGS=-lboost_program_options

all: output

output: lattices.cu
	$(CUDACC) $(NVCC_FLAGS) ../header/utils_big.cu lattices.cu -o lattices $(LDFLAGS) $(LBOOSTFLAGS)

clean:
	rm -f lattices
