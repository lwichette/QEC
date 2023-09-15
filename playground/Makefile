#CUDA_HOME=/usr/local/cuda
#CUDACC=$(CUDA_HOME)/bin/nvcc

#NVCC_FLAGS=-std=c++11	-O3	-lineinfo	-arch=sm_70	-Xptxas=-v	-I../external/cub

#LD=$(CUDACC)

#all:	test

#test:	test.cu
#	$(CUDACC)	$(NVCC_FLAGS)	test.cu	-o	test
#clean:
#	rm	-f	test


CUDA_HOME=/usr/local/cuda
CUDACC=$(CUDA_HOME)/bin/nvcc
NVCC_FLAGS=-std=c++11 -O3 -lineinfo -arch=sm_70 -Xptxas=-v -I../external/cub
LD=$(CUDACC)
LDFLAGS=-lcurand

all: output

output: test2.cu
	$(CUDACC) $(NVCC_FLAGS) test2.cu -o test2 $(LDFLAGS)

clean:
	rm -f test2

