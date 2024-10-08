CUDA_HOME = /usr/local/cuda
CUDACC= $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS=-std=c++17 -O3 -lineinfo -arch=sm_70 -Xptxas=-v  -I ./header -I /usr/share/doc
LD=$(CUDACC)
LDFLAGS=-lcurand
LBOOSTFLAGS=-lboost_program_options

ID ?= -10

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
all: wl prerun prerun_eight_vertex
=======
all: wl prerun prerun_eight_vertex main_eight_vertex
>>>>>>> 5d20b7902a58f1ee1d3108761e1d3237e3049c0c
=======
all: wl prerun prerun_eight_vertex main_eight_vertex
>>>>>>> a885bb030b31175c0fbcbee1361f09a4c13701fc
=======
all: wl prerun prerun_eight_vertex main_eight_vertex
>>>>>>> a885bb030b31175c0fbcbee1361f09a4c13701fc
=======
all: wl prerun prerun_eight_vertex main_eight_vertex
>>>>>>> a885bb030b31175c0fbcbee1361f09a4c13701fc

wl: main_WL.cu
	$(CUDACC) $(NVCC_FLAGS) ./header/wlutils.cu main_WL.cu -o wl_$(ID) $(LDFLAGS) $(LBOOSTFLAGS)

prerun: main_prerun.cu
	$(CUDACC) $(NVCC_FLAGS) ./header/wlutils.cu main_prerun.cu -o prerun_$(ID) $(LDFLAGS) $(LBOOSTFLAGS)

prerun_eight_vertex: prerun_eight_vertex.cu
	$(CUDACC) $(NVCC_FLAGS) ./header/wlutils.cu prerun_eight_vertex.cu -o prerun_eight_vertex_$(ID) $(LDFLAGS) $(LBOOSTFLAGS)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
main_eight_vertex: main_eight_vertex.cu
	$(CUDACC) $(NVCC_FLAGS) ./header/wlutils.cu main_eight_vertex.cu -o main_eight_vertex_$(ID) $(LDFLAGS) $(LBOOSTFLAGS)

>>>>>>> 5d20b7902a58f1ee1d3108761e1d3237e3049c0c
=======
main_eight_vertex: main_eight_vertex.cu
	$(CUDACC) $(NVCC_FLAGS) ./header/wlutils.cu main_eight_vertex.cu -o main_eight_vertex_$(ID) $(LDFLAGS) $(LBOOSTFLAGS)

>>>>>>> a885bb030b31175c0fbcbee1361f09a4c13701fc
=======
main_eight_vertex: main_eight_vertex.cu
	$(CUDACC) $(NVCC_FLAGS) ./header/wlutils.cu main_eight_vertex.cu -o main_eight_vertex_$(ID) $(LDFLAGS) $(LBOOSTFLAGS)

>>>>>>> a885bb030b31175c0fbcbee1361f09a4c13701fc
=======
main_eight_vertex: main_eight_vertex.cu
	$(CUDACC) $(NVCC_FLAGS) ./header/wlutils.cu main_eight_vertex.cu -o main_eight_vertex_$(ID) $(LDFLAGS) $(LBOOSTFLAGS)

>>>>>>> a885bb030b31175c0fbcbee1361f09a4c13701fc

clean:
	rm -f wl_$(ID) prerun_$(ID) prerun_eight_vertex_$(ID)
