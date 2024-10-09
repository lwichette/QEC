CUDA_HOME = /usr/local/cuda
CUDACC= $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS=-std=c++17 -O3 -lineinfo -arch=sm_70 -Xptxas=-v  -I ./header -I /usr/share/doc
LD=$(CUDACC)
LDFLAGS=-lcurand
LBOOSTFLAGS=-lboost_program_options

ID ?= -10

all: wl prerun prerun_eight_vertex main_eight_vertex

wl: main_WL.cu
	$(CUDACC) $(NVCC_FLAGS) ./header/wlutils.cu main_WL.cu -o wl_$(ID) $(LDFLAGS) $(LBOOSTFLAGS)

prerun: main_prerun.cu
	$(CUDACC) $(NVCC_FLAGS) ./header/wlutils.cu main_prerun.cu -o prerun_$(ID) $(LDFLAGS) $(LBOOSTFLAGS)

prerun_eight_vertex: prerun_eight_vertex.cu
	$(CUDACC) $(NVCC_FLAGS) ./header/wlutils.cu prerun_eight_vertex.cu -o prerun_eight_vertex_$(ID) $(LDFLAGS) $(LBOOSTFLAGS)

main_eight_vertex: main_eight_vertex.cu
	$(CUDACC) $(NVCC_FLAGS) ./header/wlutils.cu main_eight_vertex.cu -o main_eight_vertex_$(ID) $(LDFLAGS) $(LBOOSTFLAGS)


clean_wl:
	rm -f wl_$(ID)

clean_prerun:
	rm -f prerun_$(ID)

clean_prerun_eight_vertex:
	rm -f prerun_eight_vertex_$(ID)

clean_main_eight_vertex:
	rm -f main_eight_vertex_$(ID)

# General clean target
clean: clean_wl clean_prerun clean_prerun_eight_vertex clean_main_eight_vertex