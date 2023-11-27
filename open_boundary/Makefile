CUDA_HOME = /usr/local/cuda
CUDACC= $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS=-std=c++17 -O3 -lineinfo -arch=sm_70 -Xptxas=-v  -I ../header -I /usr/share/doc
LD=$(CUDACC)
LDFLAGS=-lcurand
LBOOSTFLAGS=-lboost_program_options
VALGRIND = valgrind
VALGRIND_FLAGS = --tool=memcheck --leak-check=full --track-origins=yes --error-limit=no

all: output

output: open_boundary.cu
	$(CUDACC) $(NVCC_FLAGS) ../header/utils.cu open_boundary.cu -o open_boundary $(LDFLAGS) $(LBOOSTFLAGS)

run: output
	$(VALGRIND) $(VALGRIND_FLAGS) ./open_boundary

clean:
	rm -f open_boundary
