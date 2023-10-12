CUDA_HOME=/usr/local/cuda
CUDACC=$(CUDA_HOME)/bin/nvcc
NVCC_FLAGS=-std=c++17 -O3 -lineinfo -arch=sm_70 -Xptxas=-v -I../external/cub -I ../header
LD=$(CUDACC)
LDFLAGS=-lcurand

all: output

output: single_run.cu 
	$(CUDACC) $(NVCC_FLAGS) ../header/utils.cu single_run.cu -o single_run $(LDFLAGS)

clean:
	rm -f single_run

