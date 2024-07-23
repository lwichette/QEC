CUDA_HOME = /usr/local/cuda
CUDACC= $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS=-std=c++17 -O3 -lineinfo -arch=sm_70 -Xptxas=-v  -I ./header -I /usr/share/doc
LD=$(CUDACC)
LDFLAGS=-lcurand
LBOOSTFLAGS=-lboost_program_options

all: wl prerun

wl: main_WL.cu
	$(CUDACC) $(NVCC_FLAGS) main_WL.cu -o wl $(LDFLAGS) $(LBOOSTFLAGS)

prerun: main_prerun.cu
	$(CUDACC) $(NVCC_FLAGS) main_prerun.cu -o prerun $(LDFLAGS) $(LBOOSTFLAGS)

clean:
	rm -f wl prerun
