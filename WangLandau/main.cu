#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <vector>
#include <thread>  

#include "./header/cudamacro.h"

using namespace std;

const unsigned int THREADS = 128;

void write(signed char* array, std::string filename, const long nx, const long ny, const int num_lattices, bool lattice){
    printf("Writing to %s ...\n", filename.c_str());

    int nx_w = (lattice) ? nx : 2*nx;

    std::vector<signed char> array_host(nx_w*ny*num_lattices);

    CHECK_CUDA(cudaMemcpy(array_host.data(), array, nx_w*ny*num_lattices*sizeof(*array), cudaMemcpyDeviceToHost));

    int offset;

    for (int l=0; l < num_lattices; l++){

        offset = l*nx_w*ny;

        std::ofstream f;
        f.open(filename + std::to_string(l) + std::string(".txt"));
        if (f.is_open()) {
            for (int i = 0; i < nx_w; i++) {
                for (int j = 0; j < ny; j++) {
                    f << (int)array_host[offset + i * ny + j] << " ";
                }
                f << std::endl;
            }
        }
        f.close();
    }
}

__global__ void init_lattice(signed char* lattice, const int nx, const int ny, const int num_lattices, const int seed){
    
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= nx*ny*num_lattices) return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st); // offset??

    float randval = curand_uniform(&st);

    if (randval < 0.5f){
        lattice[tid] = -1;
    }
}

__global__ void init_interactions(signed char* interactions, const int nx, const int ny, const int num_lattices, const int seed, const double p){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;
    
    int it = 0;

    while (tid < nx*ny*num_lattices*2){
        
        curandStatePhilox4_32_10_t st;
    	curand_init(seed, tid, it, &st);

        if (curand_uniform(&st) < p){
            interactions[tid] = -1;
        }
        
        it += 1;
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void calc_energy(signed char* lattice, signed char* interactions, int* d_energy, const int nx, const int ny, const int num_lattices){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    while (tid < num_lattices){

        int offset_l = tid*nx*ny;
        int offset_i = 2*offset_l;

        int energy = 0; 

        for (int l = 0; l < nx*ny; l++){
            
            int i = l/ny;
            int j = l%ny;

            int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
            int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;  

            // Only two entries correspond to Hamiltonian
            energy += lattice[offset_l + i*ny +j]*(lattice[offset_l + inn*ny + j]*interactions[offset_i + nx*ny + inn*ny + j] + lattice[offset_l + i*ny + jnn]*interactions[offset_i + i*ny + jnn]);
        }

        d_energy[tid] = energy;

        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char **argv){
    
    const int L = 12;
    
    const int num_walker = 16; //number of threads on the GPU
    const int threads_walker = 16;
    const int blocks_walker = num_walker/threads_walker;
    
    const int seed = 42;
    const float prob = 0.5;

    signed char *d_lattice;
    CHECK_CUDA(cudaMalloc(&d_lattice, num_walker * L * L * sizeof(*d_lattice)));
    CHECK_CUDA(cudaMemset(d_lattice, 1, num_walker * L * L * sizeof(*d_lattice)));
    
    signed char* d_interactions;
    CHECK_CUDA(cudaMalloc(&d_interactions, num_walker * L * L * 2 * sizeof(*d_interactions)));
    CHECK_CUDA(cudaMemset(d_interactions, 1, num_walker * L * L * 2 * sizeof(*d_interactions)));

    int* d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, num_walker * sizeof(*d_energy)));

    const int blocks_init = (L*L*2*num_walker + THREADS - 1)/THREADS;
    init_lattice<<<blocks_init, THREADS>>>(d_lattice, L, L, num_walker, seed);
    init_interactions<<<blocks_init, THREADS>>>(d_interactions, L, L, num_walker, seed + 1, prob); //use different seed

    write(d_lattice, "lattices_", L, L, num_walker, true);
    write(d_interactions, "interactions_", L, L, num_walker, false);

    calc_energy<<<blocks_walker, threads_walker>>>(d_lattice, d_interactions, d_energy, L, L, num_walker);

    std::vector<int> h_energy(num_walker);

    CHECK_CUDA(cudaMemcpy(h_energy.data(), d_energy, num_walker*sizeof(*d_energy), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_walker; i++){
        std::cout << "Lattice " << i << " with energy " << h_energy[i] << endl;
    }

    return 0;
}