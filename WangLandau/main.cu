#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <vector>
#include <thread>  
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <tuple>
#include <cmath>
#include <algorithm>

#include "./header/cudamacro.h"

using namespace std;

const unsigned int THREADS = 128;

// Might be possible to do on GPU 
std::tuple<std::vector<int>, std::vector<int>, int, int> generate_intervals(const int E_min, const int E_max, const int num_intervals, const int num_walker){
    
    std::vector<int> h_start(num_intervals);
    std::vector<int> h_end(num_intervals);
    std::vector<int> h_len(num_intervals);

    const int E_range = E_max - E_min;
    const int len_interval = E_range / (1.0f + 0.25*(num_intervals - 1)); //Interval length
    const int step_size = 0.25 * len_interval;

    int start_interval = E_min;

    int needed_space = 0;

    for (int i = 0; i < num_intervals; i++){
        
        h_start[i] = start_interval;

        if (i < num_intervals - 1){
            h_end[i] = start_interval + len_interval;
            needed_space += num_walker * len_interval;
        }
        else{
            h_end[i] = E_max;
            needed_space += num_walker * (E_max - h_start[i]);
        }

        start_interval += step_size;
    } 

    return std::make_tuple(h_start, h_end, needed_space, len_interval);
}

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

__global__ void init_lattice(signed char* lattice, const int nx, const int ny, const int num_lattices, const int seed, const float prob){
    
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= nx*ny*num_lattices) return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st); // offset??

    float randval = curand_uniform(&st);
    signed char val = (randval < prob) ? -1 : 1;

    lattice[tid] = val;
}

__global__ void init_interactions(signed char* interactions, const int nx, const int ny, const int num_lattices, const int seed, const double prob){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;
    
    int it = 0;

    while (tid < nx*ny*2){
        
        curandStatePhilox4_32_10_t st;
    	curand_init(seed, tid, it, &st);

        float randval = curand_uniform(&st);
        signed char val = (randval < prob) ? -1 : 1;
        
        interactions[tid] = val;

        it += 1;
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void calc_energy(signed char* lattice, signed char* interactions, int* d_energy, const int nx, const int ny, const int num_lattices){

    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    int offset_l = tid*nx*ny;

    int energy = 0; 

    for (int l = 0; l < nx*ny; l++){
        
        int i = l/ny;
        int j = l%ny;

        int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
        int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;  

        energy += lattice[offset_l + i*ny +j]*(lattice[offset_l + inn*ny + j]*interactions[nx*ny + inn*ny + j] + lattice[offset_l + i*ny + jnn]*interactions[i*ny + jnn]);
    }

    d_energy[tid] = energy;

    tid += blockDim.x * gridDim.x;
}

__global__ void check_energy_ranges(int *d_energy, int *d_start, int *d_end){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;
        
    int check = 1;
    
    if (d_energy[tid] > d_end[blockIdx.x] || d_energy[tid] < d_start[blockIdx.x]){
        check = 0;
    }

    assert(check);
}

__device__ void fisher_yates(int *d_shuffle, int seed, int it){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;
    
    int offset = blockDim.x*blockIdx.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, it, &st);
    
    for (int i = blockDim.x - 1; i > 0; i--){
        double randval = curand_uniform(&st);
        randval *= (i + 0.999999);
        int random_index = (int)truncf(randval);

        int temp = d_shuffle[offset + i];
        d_shuffle[offset + i] = d_shuffle[offset + random_index];
        d_shuffle[offset + random_index] = temp;
    }
}

__global__ void replica_exchange(
    int *d_offsets, int *d_energy, int *d_start, int *d_end, int *d_indices, 
    float* d_G, bool odd, int seed, int it_seed
    ){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;
    
    if ((blockIdx.x + 1) == (gridDim.x -1)){
        return;
    }
    
    long long cid = static_cast<long long>(blockDim.x)*blockIdx.x;

    if (threadIdx.x == 0){
        fisher_yates(d_indices, seed, it_seed);
    }

    if (odd){
        if (blockIdx.x % 2 != 0) return;
    }
    else{
        if (blockIdx.x % 2 != 1) return;
    }
    
    cid += d_indices[tid];

    double prob = min(1.0, d_G[d_energy[tid]]/d_G[d_energy[tid]]*d_G[d_energy[cid]]/d_G[d_energy[cid]]);
    
    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, it_seed, &st);

    if (curand_uniform(&st) < prob){
        int temp_off = d_offsets[tid];
        int temp_energy = d_energy[tid];

        d_offsets[tid] = d_offsets[cid];
        d_energy[tid] = d_energy[cid];

        d_offsets[cid] = temp_off;
        d_energy[cid] = temp_energy;
    }
}

__global__ void check_histogram(int *d_H, int *d_offset, int *d_end, int* d_start, int* d_cond, int nx, int ny, double alpha){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    int min = d_H[d_offset[tid]];

    double average = 0;

    for (int i = 0; i < (d_end[blockIdx.x] - d_start[blockIdx.x]); i++){
        if (d_H[d_offset[tid] + i] < min){
            min = d_H[d_offset[tid] + i];
        }
        average += d_H[d_offset[tid] + i];
    }

    if (min > alpha*average){
        d_cond[tid] = 1;
    }
}

__global__ void wang_landau(
    signed char *d_lattice, signed char *d_interactions, int *d_cond, int *d_energy, 
    int *d_start, int *d_end, int *d_H, float *d_G, int *d_offset, const int num_iterations, 
    const int nx, const int ny, const int seed, double factor, int it_seed
    ){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;
    
    if (d_cond[tid] != 0) return;

    int blockId = blockIdx.x;
    int offset_l = tid * nx * ny;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, it_seed, &st);
    
    for (int it = 0; it < num_iterations; it++){
        
        // Generate random int --> is that actually uniformly?
        double randval = curand_uniform(&st);
        randval *= (nx*ny - 1 + 0.999999);
        int random_index = (int)truncf(randval);

        int i = random_index/ny;
        int j = random_index % ny;

        // Set up periodic boundary conditions
        int ipp = (i + 1 < nx) ? i + 1 : 0;
        int inn = (i - 1 >= 0) ? i - 1: nx - 1;
        int jpp = (j + 1 < ny) ? j + 1 : 0;
        int jnn = (j - 1 >= 0) ? j - 1: ny - 1; 

        // Nochmal checken
        signed char energy_diff = -1 * d_lattice[offset_l + i*ny +j]*(d_lattice[offset_l + inn*ny + j]*d_interactions[nx*ny + inn*ny + j] + d_lattice[offset_l + i*ny + jnn]*d_interactions[i*ny + jnn]
                                                                    + d_lattice[offset_l + ipp*ny + j]*d_interactions[nx*ny + i*ny + j] + d_lattice[offset_l + i*ny + jpp]*d_interactions[i*ny + j]);
        
        int d_new_energy = d_energy[tid] + energy_diff; 
        
        if (d_new_energy > d_end[blockId] || d_new_energy < d_start[blockId]){
            continue;
        }
        else{
            int index_old = d_offset[tid] + d_energy[tid] - d_start[blockId];
            int index_new = d_offset[tid] + d_new_energy - d_start[blockId];

            float prob = min(1.0f, d_G[index_old]/d_G[index_new]);

            if(curand_uniform(&st) < prob){
                d_lattice[offset_l + i*ny +j] *= -1;

                d_H[index_new] += 1;
                d_G[index_new] += log(factor);

                d_energy[tid] = d_new_energy;
            }
            else{
                d_H[index_old] += 1;
                d_G[index_old] += log(factor);
            }
        }
    }
}

void read(std::vector<signed char>& lattice, std::string filename){
    
    std::ifstream inputFile(filename);
    
    if (!inputFile) {
        std::cerr << "Unable to open file " << filename << std::endl;
        return; // Return with error code
    }

    int spin = 0;

    while (inputFile >> spin){
        lattice.push_back(static_cast<signed char>(spin));
    }
}

__global__ void init_offsets(int *d_offset, int *d_start, int *d_end){

    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    // Length of interval equal except last one --> length of array is given by num_threads_per_block * (length of interval + length of last interval)  
    if (blockIdx.x == gridDim.x - 1){ 
        d_offset[tid] = (gridDim.x - 1)*blockDim.x*(d_end[0] - d_start[0]) + threadIdx.x*(d_end[gridDim.x - 1] - d_start[gridDim.x - 1]);
    }
    else{
        d_offset[tid] = tid * (d_end[0] - d_start[0]);    
    }
}

__global__ void init_indices(int *d_indices){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    d_indices[tid] = threadIdx.x;
}

int main(int argc, char **argv){
    
    const int seed = 43;

    const int num_iterations = 10;
    
    const double alpha = 0.2; // condition for histogram
    const double beta = 0.4; //end condition for factor
    
    double factor = std::exp(0);

    // lattice
    const int L = 12;

    const float prob_interactions = 0; // prob of error
    const float prob_spins = 0.5; // prob of down spin

    const int num_intervals = 3; // number of intervals
    const int threads_walker = 4; // walkers per interval
    const int num_walker = num_intervals*threads_walker; //number of all walkers

    // intervals
    const int E_min = -2*L*L;
    const int E_max = -E_min;
    
    // Struct?
    auto result = generate_intervals(E_min, E_max, num_intervals, num_walker);
    std::vector<int> h_start = std::get<0>(result);
    std::vector<int> h_end = std::get<1>(result);
    int len_histogram = std::get<2>(result);
    int len_interval = std::get<3>(result); // not sure if needed
    
    for (int i = 0; i < num_intervals; i++){
        std::cout << h_start[i] << " " << h_end[i] << endl; 
    }

    int *d_start, *d_end;
    CHECK_CUDA(cudaMalloc(&d_start, num_intervals*sizeof(*d_start)));
    CHECK_CUDA(cudaMalloc(&d_end, num_intervals*sizeof(*d_end)));

    CHECK_CUDA(cudaMemcpy(d_start, h_start.data(), num_intervals*sizeof(*d_start), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_end, h_end.data(), num_intervals*sizeof(*d_start), cudaMemcpyHostToDevice));
    
    // Histogramm and G array
    int *d_H; 
    CHECK_CUDA(cudaMalloc(&d_H, len_histogram * sizeof(*d_H)));
    CHECK_CUDA(cudaMemset(d_H, 0, len_histogram*sizeof(*d_H)));
    
    float *d_G;
    CHECK_CUDA(cudaMalloc(&d_G, len_histogram * sizeof(*d_G)));
    CHECK_CUDA(cudaMemset(d_G, 0, len_histogram*sizeof(*d_G)));

    int *d_offset;
    CHECK_CUDA(cudaMalloc(&d_offset, num_walker*sizeof(*d_offset)));

    int *d_cond;
    CHECK_CUDA(cudaMalloc(&d_cond, num_walker*sizeof(*d_cond)));
    CHECK_CUDA(cudaMemset(d_cond, 0, num_walker*sizeof(*d_cond)));

    int *d_indices;
    CHECK_CUDA(cudaMalloc(&d_indices, num_walker*sizeof(*d_indices)));

    // lattice & interactions and energies
    signed char *d_lattice;
    CHECK_CUDA(cudaMalloc(&d_lattice, num_walker * L * L * sizeof(*d_lattice)));
    
    signed char* d_interactions;
    CHECK_CUDA(cudaMalloc(&d_interactions, L * L * 2 * sizeof(*d_interactions)));

    int* d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, num_walker * sizeof(*d_energy)));
    
    const int blocks_init = (L*L*num_walker + THREADS - 1)/THREADS;

    init_lattice<<<blocks_init, THREADS>>>(d_lattice, L, L, num_walker, seed, prob_spins);
    
    init_interactions<<<blocks_init, THREADS>>>(d_interactions, L, L, num_walker, seed + 1, prob_interactions); //use different seed

    calc_energy<<<num_intervals, threads_walker>>>(d_lattice, d_interactions, d_energy, L, L, num_walker);    
    
    check_energy_ranges<<<num_intervals, threads_walker>>>(d_energy, d_start, d_end);

    init_offsets<<<num_intervals, threads_walker>>>(d_offset, d_start, d_end);
    
    init_indices<<<num_intervals, threads_walker>>>(d_indices);

    int it = 0;
    int min_value = 0;

    //while (factor > std::exp(beta)){
        // seeds anpassen
    while(min_value == 0){
        
        wang_landau<<<num_intervals, threads_walker>>>(d_lattice, d_interactions, d_cond, d_energy, d_start, d_end, d_H, d_G, d_offset, num_iterations, L, L, seed + 2, factor, it);
        
        check_histogram<<<num_intervals, threads_walker>>>(d_H, d_offset, d_end, d_start, d_cond, L, L, alpha);

        thrust::device_ptr<int> dev_ptr(d_cond);
        thrust::device_ptr<int> min_ptr = thrust::min_element(dev_ptr, dev_ptr + num_walker);

        min_value = *min_ptr;

        replica_exchange<<<num_intervals, threads_walker>>>(d_offset, d_energy, d_start, d_end, d_indices, d_G, true, seed, it);
        replica_exchange<<<num_intervals, threads_walker>>>(d_offset, d_energy, d_start, d_end, d_indices, d_G, false, seed, it);

        it += 1;
    }

    cudaMemset(d_H, 0, len_histogram*sizeof(*d_H));

    factor = sqrt(factor);
    
    cudaDeviceSynchronize();

    return 0;
}