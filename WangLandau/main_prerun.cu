#include <getopt.h>
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
#include <math.h>     
#include <algorithm>
#include <filesystem>

#include "./header/cudamacro.h"

using namespace std;


void write(
    signed char* array, std::string filename, const long nx, const long ny, 
    const int num_lattices, bool lattice, const std::vector<int>& energies = std::vector<int>()
    ){
    printf("Writing to %s ...\n", filename.c_str());

    int nx_w = (lattice) ? nx : 2*nx;

    std::vector<signed char> array_host(nx_w*ny*num_lattices);

    CHECK_CUDA(cudaMemcpy(array_host.data(), array, nx_w*ny*num_lattices*sizeof(*array), cudaMemcpyDeviceToHost));

    int offset;

    if (num_lattices == 1){
        offset = 0;
    
        std::ofstream f;
        f.open(filename + std::string(".txt"));
        
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
    else{
        for (int l=0; l < num_lattices; l++){

            offset = l*nx_w*ny;

            std::ofstream f;
            if (energies.empty()){
                f.open(filename + "_" + std::to_string(l) + std::string(".txt"));
            }
            else{
                f.open(filename + "_" + std::to_string(energies[l]) + std::string(".txt"));
            }
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
}

__global__ void init_lattice(signed char* lattice, const int nx, const int ny, const int num_lattices, const int seed){
    
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st); 

    __shared__ float prob;
    
    if (threadIdx.x == 0){
        prob = curand_uniform(&st);
    }
    
    __syncthreads();
    
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

__global__ void calc_energy(signed char* lattice, signed char* interactions, int* d_energy, const int nx, const int ny, const int num_lattices){

    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    int energy = 0; 

    int offset_lattice = tid*nx*ny;

    for (int l = 0; l < nx*ny; l++){
        
        int i = l/ny;
        int j = l%ny;

        int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
        int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;  

        energy += lattice[offset_lattice + i*ny +j]*(lattice[offset_lattice + inn*ny + j]*interactions[nx*ny + inn*ny + j] + lattice[offset_lattice + i*ny + jnn]*interactions[i*ny + jnn]);
    }

    d_energy[tid] = energy;

    tid += blockDim.x * gridDim.x;
}

__device__ void store_lattice(
    signed char *d_lattice, int *d_energy, int* d_found_interval, signed char* d_store_lattice,
    const int E_min, const int nx, const int ny, const int tid, const int len_interval
    ){
    
    int interval_index = (d_energy[tid] - E_min)/(len_interval);

    if (atomicCAS(&d_found_interval[interval_index], 0, 1) != 0) return;

    for (int i=0; i < nx; i++){
        for (int j=0; j < ny; j++){
            d_store_lattice[interval_index*nx*ny + i*ny + j] = d_lattice[tid*nx*ny + i*ny +j];
        }
    }

    return;
}

__global__ void wang_landau_pre_run(
    signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_H, int* d_iter, int *d_found_interval,
    signed char *d_store_lattice, const int E_min, const int E_max, const int num_iterations, const int nx, const int ny, 
    const int seed, const int len_interval, const int found_interval
    ){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    const int offset_lattice = tid*nx*ny;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_iter[tid], &st);
    
    for (int it = 0; it < num_iterations; it++){

        // Generate random int --> is that actually uniformly?
        double randval = curand_uniform(&st);
        randval *= (nx*ny - 1 + 0.999999);
        int random_index = (int)trunc(randval);

        d_iter[tid] += 1;

        int i = random_index/ny;
        int j = random_index % ny;

        // Set up periodic boundary conditions
        int ipp = (i + 1 < nx) ? i + 1 : 0;
        int inn = (i - 1 >= 0) ? i - 1: nx - 1;
        int jpp = (j + 1 < ny) ? j + 1 : 0;
        int jnn = (j - 1 >= 0) ? j - 1: ny - 1; 

        // Nochmal checken
        signed char energy_diff = -2 * d_lattice[offset_lattice + i*ny +j]*(d_lattice[offset_lattice + inn*ny + j]*d_interactions[nx*ny + inn*ny + j] + d_lattice[offset_lattice + i*ny + jnn]*d_interactions[i*ny + jnn]
                                                                    + d_lattice[offset_lattice + ipp*ny + j]*d_interactions[nx*ny + i*ny + j] + d_lattice[offset_lattice + i*ny + jpp]*d_interactions[i*ny + j]);

        int d_new_energy = d_energy[tid] + energy_diff; 

        int index_old = d_energy[tid] - E_min;
        
        if (d_new_energy > E_max || d_new_energy < E_min){
            printf("Iterator %d \n", it);
            printf("Thread Id %lld \n", tid);
            printf("Randval %f \n", randval);
            printf("Energy out of range %d \n", d_new_energy);
            printf("Old energy %d \n", d_energy[tid]);
            assert(0);
            return;
        }
        else{
            
            int index_new = d_new_energy - E_min;

            float prob = expf(d_H[index_old] - d_H[index_new]);

            if(curand_uniform(&st) < prob){
                d_lattice[offset_lattice + i*ny +j] *= -1;
                d_energy[tid] = d_new_energy;
                d_iter[tid] += 1;

                atomicAdd(&d_H[index_new], 1);

                if (found_interval == 0){
                    store_lattice(d_lattice, d_energy, d_found_interval, d_store_lattice, E_min, nx, ny, tid, len_interval);
                }
            }
            else{
                atomicAdd(&d_H[index_old], 1);
            }
        }
    }
}

void create_directory(std::string path){
    if (!std::filesystem::exists(path)) {
        // Create directory
        if (std::filesystem::create_directories(path)) {
            std::cout << "Successfully created directory: " << path << std::endl; 
        } else {
            std::cerr << "Failed to create directory: " << path << std::endl;
        }
    } else {
        std::cout << "Directory already exists: " << path << std::endl;
    }
}

void write_histograms(int *d_H, std::string path_histograms, int len_histogram, int seed, int E_min){
    
    printf("Writing to %s ...\n", path_histograms.c_str());

    std::vector<int> h_histogram(len_histogram);

    CHECK_CUDA(cudaMemcpy(h_histogram.data(), d_H, len_histogram*sizeof(*d_H), cudaMemcpyDeviceToHost));

    std::ofstream f;
    f.open(std::string(path_histograms + "/histogram.txt"));
    
    if (f.is_open()) {
        for (int i=0; i < len_histogram; i++){     
            int energy = E_min + i;
            if (h_histogram[i] > 0){
                f << energy << " " << 1 << std::endl;   
            }
            else{
                f << energy << " " << 0 << std::endl;
            }
        }
    }
}

typedef struct
{
    std::vector<int> h_start;
    std::vector<int> h_end;
    long long len_histogram_over_all_walkers;
    int len_interval;
} IntervalResult;



IntervalResult generate_intervals(const int E_min, const int E_max, int num_intervals, int num_walker, float overlap_decimal)
{
    IntervalResult interval_result;

    std::vector<int> h_start(num_intervals);
    std::vector<int> h_end(num_intervals);

    const int E_range = E_max - E_min + 1;
    const int len_interval = E_range / (1.0f + overlap_decimal * (num_intervals - 1)); // Len_interval computation stems from condition: len_interval + overlap * len_interval * (num_intervals - 1) = E_range
    const int step_size = overlap_decimal * len_interval;

    int start_interval = E_min;

    long long len_histogram_over_all_walkers = 0;

    for (int i = 0; i < num_intervals; i++)
    {

        h_start[i] = start_interval;

        if (i < num_intervals - 1)
        {
            h_end[i] = start_interval + len_interval - 1;
            len_histogram_over_all_walkers += num_walker * len_interval;
        }
        else
        {
            h_end[i] = E_max;
            len_histogram_over_all_walkers += num_walker * (E_max - h_start[i] + 1);
        }

        start_interval += step_size;
    }
    interval_result.h_start = h_start;
    interval_result.h_end = h_end;
    interval_result.len_histogram_over_all_walkers = len_histogram_over_all_walkers;
    interval_result.len_interval = len_interval;

    return interval_result;
}

int main(int argc, char **argv){
    
    int X, Y;
    
    float prob_interactions;

    int num_wl_loops, num_iterations, num_walker;

    int seed;

    int num_intervals;
    
    int och;
    while (1) {
        int option_index = 0;
        static struct option long_options[] = {
            {"X", required_argument, 0, 'x'},
            {"Y", required_argument, 0, 'y'},
            {"prob", required_argument, 0, 'p'},
            {"nit", required_argument, 0, 'n'},
            {"nl", required_argument, 0, 'l'},
            {"nw", required_argument, 0, 'w'},
            {"seed", required_argument, 0, 's'},
            {"num_intervals", required_argument, 0, 'i'},
            {0, 0, 0, 0}
        };

        och = getopt_long(argc, argv, "x:y:p:n:l:w:s:i:", long_options, &option_index);
        
        if (och == -1)
            break;
        switch (och) {
			case 0:// handles long opts with non-NULL flag field
				break;
            case 'x':
				X = atoi(optarg);
                break;
            case 'y':
                Y = atoi(optarg);
                break;
            case 'p':
                prob_interactions = atof(optarg);
                break;
			case 'n':
                num_iterations = atoi(optarg);
                break;
			case 'l':
                num_wl_loops = atoi(optarg);
                break;
			case 'w':
                num_walker = atoi(optarg);
                break;
			case 's':
				seed = atoi(optarg);
                break;
            case 'i':
			    num_intervals = atoi(optarg);
			    break;
			case '?':
				exit(EXIT_FAILURE);

			default:
				fprintf(stderr, "unknown option: %c\n", och);
				exit(EXIT_FAILURE);
        }
    }

    double factor = std::exp(1);

    const int E_min = -2*X*Y;
    const int E_max = -E_min;

    IntervalResult interval_result = generate_intervals(E_min, E_max, num_intervals, 1, 1.0f);

    for (int i=0; i< num_intervals; i++){
        std::cout << interval_result.h_start[i] << " " << interval_result.h_end[i] << std::endl;
    }

    long long len_histogram = E_max - E_min + 1;
    
    int *d_H; 
    CHECK_CUDA(cudaMalloc(&d_H, len_histogram * sizeof(*d_H)));
    CHECK_CUDA(cudaMemset(d_H, 0, len_histogram*sizeof(*d_H)));

    int *d_iter;
    CHECK_CUDA(cudaMalloc(&d_iter, num_walker*sizeof(*d_iter)));
    CHECK_CUDA(cudaMemset(d_iter, 0, num_walker*sizeof(*d_iter)));

    // lattice & interactions and energies
    signed char *d_lattice;
    CHECK_CUDA(cudaMalloc(&d_lattice, num_walker * X * Y * sizeof(*d_lattice)));
    
    signed char* d_interactions;
    CHECK_CUDA(cudaMalloc(&d_interactions, X * Y * 2 * sizeof(*d_interactions)));

    int* d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, num_walker * sizeof(*d_energy)));

    signed char* d_store_lattice;
    CHECK_CUDA(cudaMalloc(&d_store_lattice, num_intervals*X*Y*sizeof(*d_store_lattice)));

    int *d_found_interval;
    CHECK_CUDA(cudaMalloc(&d_found_interval, num_intervals*sizeof(*d_found_interval)));
    CHECK_CUDA(cudaMemset(d_found_interval, 0, num_intervals*sizeof(*d_found_interval)));

    int *d_interval_energies;
    CHECK_CUDA(cudaMalloc(&d_interval_energies, num_intervals*sizeof(*d_interval_energies)));
    CHECK_CUDA(cudaMemset(d_interval_energies, 0, num_intervals*sizeof(*d_interval_energies)));

    init_lattice<<<num_walker, X*Y>>>(d_lattice, X, Y, num_walker, seed);

    init_interactions<<<1, 2*X*Y>>>(d_interactions, X, Y, num_walker, seed + 1, prob_interactions); //use different seed

    calc_energy<<<1, num_walker>>>(d_lattice, d_interactions, d_energy, X, Y, num_walker);    

    int found_interval = 0;

    for (int i=0; i < num_wl_loops; i++){
        if (i % 100 == 0) printf("Num wl loop: %d \n", i);
        
        wang_landau_pre_run<<<1, num_walker>>>(d_lattice, d_interactions, d_energy, d_H, d_iter, d_found_interval, d_store_lattice, E_min, E_max, num_iterations, X, Y, seed + 2, interval_result.len_interval, found_interval);
        cudaDeviceSynchronize();
        
        if (found_interval == 0){
            thrust::device_ptr<int> d_found_interval_ptr(d_found_interval);
            thrust::device_ptr<int> min_found_interval_ptr = thrust::min_element(d_found_interval_ptr, d_found_interval_ptr + num_intervals);
            found_interval = *min_found_interval_ptr;
        }
    }

    calc_energy<<<1,num_intervals>>>(d_store_lattice, d_interactions, d_interval_energies, X, Y, num_intervals);
    
    std::vector<int> h_interval_energies(num_intervals);
    CHECK_CUDA(cudaMemcpy(h_interval_energies.data(), d_interval_energies, num_intervals*sizeof(*d_interval_energies), cudaMemcpyDeviceToHost));

    std::string path = "init/prob_" + std::to_string(prob_interactions) + "/X_" + std::to_string(X) + "_Y_" + std::to_string(Y) + "/seed_" + std::to_string(seed);

    create_directory(path + "/interactions");
    create_directory(path + "/lattice");
    create_directory(path + "/histogram");

    write(d_interactions, path + "/interactions/interactions", X, Y, 1, false);
    write(d_store_lattice, path + "/lattice/lattice", X, Y, num_intervals, true, h_interval_energies);
    write_histograms(d_H, path + "/histogram/", len_histogram, seed, E_min);

    return 0;
}