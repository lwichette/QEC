CUDA_HOME=/usr/local/cuda
CUDACC=$(CUDA_HOME)/bin/nvcc
NVCC_FLAGS=-std=c++17 -O3 -lineinfo -arch=sm_70 -Xptxas=-v -I../external/cub -I /home/dfki.uni-bremen.de/mbeuerle/User/mbeuerle/Code/qec/linearization/header
LD=$(CUDACC)
LDFLAGS=-lcurand

all: output

output: single_run.cu 
	$(CUDACC) $(NVCC_FLAGS) utils.cu single_run.cu -o single_run $(LDFLAGS)

clean:
	rm -f single_run

