CUDA_HOME = /usr/local/cuda
CUDACC = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS = -std=c++17 -O3 -lineinfo -arch=sm_70 -Xptxas=-v -I./header -I/usr/share/doc
LD = $(CUDACC)
CUDACFLAGS = -c -G -g
LDFLAGS = -lcurand
LBOOSTFLAGS = -lboost_program_options

all: output

output: open_boundary.o utils.o
	$(LD) $^ -o dev_open_boundary $(LDFLAGS) $(LBOOSTFLAGS)

open_boundary.o: open_boundary.cu
	$(CUDACC) $(CUDACFLAGS) $(NVCC_FLAGS) -c $< -o $@

utils.o: header/utils.cu
	$(CUDACC) $(CUDACFLAGS) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f dev_open_boundary *.o
