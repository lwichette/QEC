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
//#include <math.h>     
#include <algorithm>
#include <unistd.h> // For Sleep

#include "./header/cudamacro.h"

using namespace std;

const unsigned int THREADS = 128;

// Might be possible to do on GPU 
std::tuple<std::vector<int>, std::vector<int>, long long, int> generate_intervals(const int E_min, const int E_max, const int num_intervals, const int num_walker){
    
    std::vector<int> h_start(num_intervals);
    std::vector<int> h_end(num_intervals);

    const int E_range = E_max - E_min + 1;
    const int len_interval = E_range / (1.0f + 0.25*(num_intervals - 1)); //Interval length
    const int step_size = 0.25 * len_interval;

    int start_interval = E_min;

    long long needed_space = 0;

    for (int i = 0; i < num_intervals; i++){
        
        h_start[i] = start_interval;

        if (i < num_intervals - 1){
            h_end[i] = start_interval + len_interval;
            needed_space += num_walker * len_interval;
        }
        else{
            h_end[i] = E_max;
            needed_space += num_walker * (E_max - h_start[i] + 1);
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
    
    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    while (tid < nx*ny*2){

        float randval = curand_uniform(&st);
        signed char val = (randval < prob) ? -1 : 1;
        
        interactions[tid] = val;

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void calc_energy(signed char* lattice, signed char* interactions, int* d_energy, int *d_offset_lattice, const int nx, const int ny, const int num_lattices){

    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    int energy = 0; 

    for (int l = 0; l < nx*ny; l++){
        
        int i = l/ny;
        int j = l%ny;

        int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
        int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;  

        energy += lattice[d_offset_lattice[tid] + i*ny +j]*(lattice[d_offset_lattice[tid] + inn*ny + j]*interactions[nx*ny + inn*ny + j] + lattice[d_offset_lattice[tid] + i*ny + jnn]*interactions[i*ny + jnn]);
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

__device__ void fisher_yates(int *d_shuffle, int seed, int *d_iter){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;
    
    int offset = blockDim.x*blockIdx.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_iter[tid], &st);
    
    for (int i = blockDim.x - 1; i > 0; i--){
        double randval = curand_uniform(&st);
        randval *= (i + 0.999999);
        int random_index = (int)truncf(randval);
        d_iter[tid] += 1;

        int temp = d_shuffle[offset + i];
        d_shuffle[offset + i] = d_shuffle[offset + random_index];
        d_shuffle[offset + random_index] = temp;
    }
}

__global__ void replica_exchange(
    int *d_offset_lattice, int *d_energy, int *d_start, int *d_end, int *d_indices, 
    float* d_logG, bool even, int seed, int *d_iter
    ){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;
    
    // Check if last block
    if (blockIdx.x == (gridDim.x -1)){
        return;
    }
    
    //change index
    long long cid = static_cast<long long>(blockDim.x)*(blockIdx.x + 1);

    if (threadIdx.x == 0){
        fisher_yates(d_indices, seed, d_iter);
    }

    // Synchronize

    if (even){
        if (blockIdx.x % 2 != 0) return;
    }
    else{
        if (blockIdx.x % 2 != 1) return;
    }
    
    cid += d_indices[tid];

    //Check energy ranges
    if (d_energy[tid] > d_end[blockIdx.x+1] || d_energy[tid] < d_start[blockIdx.x + 1]) return;
    if (d_energy[cid] > d_end[blockIdx.x] || d_energy[tid] < d_start[blockIdx.x]) return;

    double prob = min(1.0, d_logG[d_energy[tid]]/d_logG[d_energy[tid]]*d_logG[d_energy[cid]]/d_logG[d_energy[cid]]);
    
    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_iter[tid], &st);

    if (curand_uniform(&st) < prob){
        
        int temp_off = d_offset_lattice[tid];
        int temp_energy = d_energy[tid];

        d_offset_lattice[tid] = d_offset_lattice[cid];
        d_energy[tid] = d_energy[cid];

        d_offset_lattice[cid] = temp_off;
        d_energy[cid] = temp_energy;
        
        d_iter[tid] += 1;
    }
}

__global__ void check_histogram(int *d_H, int *d_offset_histogramm, int *d_end, int* d_start, int* d_cond, int nx, int ny, double alpha){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    int min = d_H[d_offset_histogramm[tid]];

    double average = 0;

    for (int i = 0; i < (d_end[blockIdx.x] - d_start[blockIdx.x] + 1); i++){
        if (d_H[d_offset_histogramm[tid] + i] < min){
            min = d_H[d_offset_histogramm[tid] + i];
        }
        average += d_H[d_offset_histogramm[tid] + i];
    }

    average = average/(d_end[blockIdx.x] - d_start[blockIdx.x] + 1);

    if (min >= alpha*average){
        d_cond[tid] = 1;
    }
}

__global__ void wang_landau(
    signed char *d_lattice, signed char *d_interactions, int *d_energy, 
    int *d_start, int *d_end, int *d_H, float *d_logG, int *d_offset_histogramm, int *d_offset_lattice, const int num_iterations, 
    const int nx, const int ny, const int seed, double factor, int* d_iter
    ){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    int blockId = blockIdx.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_iter[tid], &st);
    
    for (int it = 0; it < num_iterations; it++){
        
        // Generate random int --> is that actually uniformly?
        double randval = curand_uniform(&st);
        randval *= (nx*ny - 1 + 0.999999);
        int random_index = (int)truncf(randval);

        d_iter[tid] += 1;

        int i = random_index/ny;
        int j = random_index % ny;

        // Set up periodic boundary conditions
        int ipp = (i + 1 < nx) ? i + 1 : 0;
        int inn = (i - 1 >= 0) ? i - 1: nx - 1;
        int jpp = (j + 1 < ny) ? j + 1 : 0;
        int jnn = (j - 1 >= 0) ? j - 1: ny - 1; 

        // Nochmal checken
        signed char energy_diff = -2 * d_lattice[d_offset_lattice[tid] + i*ny +j]*(d_lattice[d_offset_lattice[tid] + inn*ny + j]*d_interactions[nx*ny + inn*ny + j] + d_lattice[d_offset_lattice[tid] + i*ny + jnn]*d_interactions[i*ny + jnn]
                                                                    + d_lattice[d_offset_lattice[tid] + ipp*ny + j]*d_interactions[nx*ny + i*ny + j] + d_lattice[d_offset_lattice[tid] + i*ny + jpp]*d_interactions[i*ny + j]);

        int d_new_energy = d_energy[tid] + energy_diff; 
        
        int index_old = d_offset_histogramm[tid] + d_energy[tid] - d_start[blockId];
        
        if (d_new_energy > d_end[blockId] || d_new_energy < d_start[blockId]){
            d_H[index_old] += 1;
            d_logG[index_old] += log(factor);
        }
        else{
            
            int index_new = d_offset_histogramm[tid] + d_new_energy - d_start[blockId];

            float prob = exp(d_logG[index_old] - d_logG[index_new]);

            if(curand_uniform(&st) < prob){
                d_lattice[d_offset_lattice[tid] + i*ny +j] *= -1;

                d_H[index_new] += 1;
                d_logG[index_new] += log(factor);

                d_energy[tid] = d_new_energy;

                d_iter[tid] += 1;
            }
            else{
                d_H[index_old] += 1;
                d_logG[index_old] += log(factor);
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

__global__ void init_offsets_histogramm(int *d_offset_histogramm, int *d_start, int *d_end){

    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    // Length of interval equal except last one --> length of array is given by num_threads_per_block * (length of interval + length of last interval)  
    if (blockIdx.x == gridDim.x - 1){ 
        d_offset_histogramm[tid] = (gridDim.x - 1)*blockDim.x*(d_end[0] - d_start[0] + 1) + threadIdx.x*(d_end[gridDim.x - 1] - d_start[gridDim.x - 1] + 1);
    }
    else{
        d_offset_histogramm[tid] = tid * (d_end[0] - d_start[0] + 1);    
    }
}

__global__ void init_offsets_lattice(int *d_offset_lattice, int nx, int ny){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    d_offset_lattice[tid] = tid*nx*ny;
}

__global__ void init_indices(int *d_indices){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    d_indices[tid] = threadIdx.x;
}

int main(int argc, char **argv){

    const int pre_run_duration = 1000;
    
    const int seed = 42;

    const int num_iterations = 10000;
    
    const double alpha = 0.2; // condition for histogram
    const double beta = 0.4; //end condition for factor
    
    double factor = std::exp(1);

    // lattice
    const int L = 4;

    const float prob_interactions = 0; // prob of error
    const float prob_spins = 0; // prob of down spin

    const int num_intervals = 1; // number of intervals
    const int threads_walker = 16; // walkers per interval
    const int num_walker = num_intervals*threads_walker; //number of all walkers

    // intervals
    const int E_min = -2*L*L;
    const int E_max = -E_min;
    
    // Struct?
    auto result = generate_intervals(E_min, E_max, num_intervals, num_walker);
    std::vector<int> h_start = std::get<0>(result);
    std::vector<int> h_end = std::get<1>(result);
    long long len_histogram = std::get<2>(result);
    long long len_interval = std::get<3>(result); // not sure if needed

    int *d_start, *d_end;
    CHECK_CUDA(cudaMalloc(&d_start, num_intervals*sizeof(*d_start)));
    CHECK_CUDA(cudaMalloc(&d_end, num_intervals*sizeof(*d_end)));

    CHECK_CUDA(cudaMemcpy(d_start, h_start.data(), num_intervals*sizeof(*d_start), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_end, h_end.data(), num_intervals*sizeof(*d_start), cudaMemcpyHostToDevice));

    // Histogramm and G array
    int *d_H; 
    CHECK_CUDA(cudaMalloc(&d_H, len_histogram * sizeof(*d_H)));
    CHECK_CUDA(cudaMemset(d_H, 0, len_histogram*sizeof(*d_H)));

    float *d_logG;
    CHECK_CUDA(cudaMalloc(&d_logG, len_histogram * sizeof(*d_logG)));
    CHECK_CUDA(cudaMemset(d_logG, 0, len_histogram*sizeof(*d_logG)));

    int *d_offset_histogramm;
    CHECK_CUDA(cudaMalloc(&d_offset_histogramm, num_walker*sizeof(*d_offset_histogramm)));

    int *d_offset_lattice;
    CHECK_CUDA(cudaMalloc(&d_offset_lattice, num_walker*sizeof(*d_offset_lattice)));

    int *d_cond;
    CHECK_CUDA(cudaMalloc(&d_cond, num_walker*sizeof(*d_cond)));
    CHECK_CUDA(cudaMemset(d_cond, 0, num_walker*sizeof(*d_cond)));

    int *d_indices;
    CHECK_CUDA(cudaMalloc(&d_indices, num_walker*sizeof(*d_indices)));

    int *d_iter;
    CHECK_CUDA(cudaMalloc(&d_iter, num_walker*sizeof(*d_iter)));
    CHECK_CUDA(cudaMemset(d_iter, 0, num_walker*sizeof(*d_iter)));

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
    
    init_offsets_lattice<<<num_intervals, threads_walker>>>(d_offset_lattice, L, L);

    calc_energy<<<num_intervals, threads_walker>>>(d_lattice, d_interactions, d_energy, d_offset_lattice, L, L, num_walker);    

    check_energy_ranges<<<num_intervals, threads_walker>>>(d_energy, d_start, d_end);

    init_offsets_histogramm<<<num_intervals, threads_walker>>>(d_offset_histogramm, d_start, d_end);
    
    init_indices<<<num_intervals, threads_walker>>>(d_indices);

    for (int i=0; i < pre_run_duration; i++){

        wang_landau<<<num_intervals, threads_walker>>>(d_lattice, d_interactions, d_energy, d_start, d_end, d_H, d_logG, d_offset_histogramm, d_offset_lattice, num_iterations, L, L, seed + 2, factor, d_iter);

        replica_exchange<<<num_intervals, threads_walker>>>(d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG, true, seed + 2, d_iter);

        replica_exchange<<<num_intervals, threads_walker>>>(d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG, false, seed + 2, d_iter);
    }

    std::vector<int> h_histogram(len_histogram);
        
    CHECK_CUDA(cudaMemcpy(h_histogram.data(), d_H, len_histogram*sizeof(*d_H), cudaMemcpyDeviceToHost));

    std::vector<int> energies_histogram;

    for (int i=0; i<num_intervals; i++){

        int start_energy = h_start[i];
        int end_energy = h_end[i];
        int len_int = h_end[i] - h_start[i] + 1;

        for (int j=0; j<threads_walker; j++){
            for (int k=0; k < len_int; k++){
                energies_histogram.push_back(h_start[i] + k);
            }    
        }
    }

    std::ofstream f;
    f.open("histogramm.txt");

    if (f.is_open()) {
        for (int i=0; i<len_histogram; i++){
            f << (int)energies_histogram[i] << " : " << (int)h_histogram[i];
            f << std::endl;
        }
    }
    f.close();
    
    return 0;
}