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
            f.open(filename + "_" + std::to_string(l) + std::string(".txt"));
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
        int random_index = (int)trunc(randval);
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

__global__ void wang_landau_pre_run(
    signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_H, int* d_iter,
    const int E_min, const int E_max, const int num_iterations, const int nx, const int ny, const int seed
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

        if (random_index == 144){
            printf("ERROOOOOORRRR");
        }

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
            printf("Thread Id %d \n", tid);
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
            }
            else{
                atomicAdd(&d_H[index_old], 1);
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

void write_histograms(int *d_H, std::string path_histograms, int len_histogram, int seed, unsigned long long num_iterations, int E_min){
    
    std::vector<int> h_histogram(len_histogram);

    CHECK_CUDA(cudaMemcpy(h_histogram.data(), d_H, len_histogram*sizeof(*d_H), cudaMemcpyDeviceToHost));

    std::ofstream f;
    f.open(std::string(path_histograms + "/histogram_seed_" + std::to_string(seed) + "_ni_" + std::to_string(num_iterations) + ".txt"));
    
    if (f.is_open()) {
        for (int i=0; i < len_histogram; i++){     
            int energy = E_min + i;
            bool condition = false;

            if (energy < 0){
                condition = (h_histogram[i] > 0 || h_histogram[i + 2 * abs(energy)] > 0);
            }
            else if (energy == 0){
                condition = (h_histogram[i] > 0);
            }
            else{
                condition = (h_histogram[i] > 0 || h_histogram[i - 2 * abs(energy)] > 0);
            }

            f << energy << " " << (condition ? 1 : 0) << std::endl;   
        }
    }
}

int main(int argc, char **argv){

    int X, Y;
    
    float prob_interactions;

    int num_wl_loops, num_iterations, num_walker;

    int seed;

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
            {0, 0, 0, 0}
        };

        och = getopt_long(argc, argv, "l:p:i:n:w:s", long_options, &option_index);
        
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

    init_lattice<<<num_walker, X*Y>>>(d_lattice, X, Y, num_walker, seed);

    init_interactions<<<1, 2*X*Y>>>(d_interactions, X, Y, num_walker, seed + 1, prob_interactions); //use different seed

    calc_energy<<<1, num_walker>>>(d_lattice, d_interactions, d_energy, X, Y, num_walker);    

    for (int i=0; i < num_wl_loops; i++){
        if (i % 100 == 0){
            printf("%d \n", i);
        }
        wang_landau_pre_run<<<1, num_walker>>>(d_lattice, d_interactions, d_energy, d_H, d_iter, E_min, E_max, num_iterations, X, Y, seed + 2);
    }

    std::string path_interactions = "interactions/prob_" + std::to_string(prob_interactions) + "/X_" + std::to_string(X) + "_Y_" + std::to_string(Y);
    std::string path_histograms = "histograms/prob_" + std::to_string(prob_interactions) + "/X_" + std::to_string(X) + "_Y_" + std::to_string(Y);
    
    unsigned long long total_iterations = num_iterations*num_wl_loops;
    
    create_directory(path_interactions);
    create_directory(path_histograms);

    write(d_interactions, path_interactions + "/interactions_seed_" + std::to_string(seed), X, Y, 1, false);
    
    write_histograms(d_H, path_histograms, len_histogram, seed, total_iterations, E_min);

    return 0;
}