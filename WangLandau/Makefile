CUDA_HOME = /usr/local/cuda
CUDACC= $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS=-std=c++17 -O3 -lineinfo -arch=sm_70 -Xptxas=-v  -I ./header -I /usr/share/doc
LD=$(CUDACC)
LDFLAGS=-lcurand
LBOOSTFLAGS=-lboost_program_options

all: output

output: main.cu
	$(CUDACC) $(NVCC_FLAGS) main.cu -o wl $(LDFLAGS) $(LBOOSTFLAGS)

clean:
	rm -f wl
