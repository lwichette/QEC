#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <getopt.h>
#include <vector>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <sstream>
#include <iomanip>
#include <thread>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <unistd.h> 
#include "./header/cudamacro.h"
#include <chrono> 

typedef struct
{
    std::vector<int> h_start;
    std::vector<int> h_end;
    long long len_histogram_over_all_walkers;
    int len_interval;
} IntervalResult;

typedef struct
{   
    int X;
    int Y;
    int num_iterations;
    float prob_interactions;
    double alpha;
    double beta;
    int E_min;
    int E_max;
    int num_intervals;
    int walker_per_interval;
    float overlap_decimal;
    int num_iterations_pre_run;
} Options;

void parse_args(int argc, char *argv[], Options *options)
{
    // overlap decimal is more like the reciprocal non overlap parameter here, i.e. 0 as overlap_decimal is full overlap of intervals.

    int opt;

    while (1){
        int option_index = 0;
        static struct option long_options[] = {
            {"X", 1, 0, 'x'},
            {"Y", 1, 0, 'y'},
            {"num_iterations", 1, 0, 'n'},
            {"prob_interactions", 1, 0, 'p'},
            {"alpha", 1, 0, 'a'},
            {"beta", 1, 0, 'b'},
            {"num_intervals", 1, 0, 'i'},
            {"walker_per_interval", 1, 0, 'w'},
            {"overlap_decimal", 1, 0, 'o'},
            {"num_iterations_pre_run", 1, 0, 'r'},
            {0, 0, 0, 0}};
        
        opt = getopt_long(argc, argv, "x:y:n:p:a:b:i:w:o:r:", long_options, &option_index);
        
        if (opt == -1)
            break;
        switch (opt)
        {
        case 'x':
            options->X = std::atoi(optarg);
            break;
        case 'y':
            options->Y = std::atoi(optarg);
            break;
        case 'n':
            options->num_iterations = std::atoi(optarg);
            break;
        case 'p':
            options->prob_interactions = std::atof(optarg);
            break;
        case 'a':
            options->alpha = std::atof(optarg);
            break;
        case 'b':
            options->beta = std::atof(optarg);
            break;
        case 'i':
            options->num_intervals = std::atoi(optarg);
            break;
        case 'w':
            options->walker_per_interval = std::atoi(optarg);
            break;
        case 'o':
            options->overlap_decimal = std::atof(optarg);
            break;
        case 'r':
            options->num_iterations_pre_run = std::atoi(optarg);
            break;
        default:
            fprintf(stderr, "Usage: %s [-i num_intervals] [-m E_min] [-M E_max] [-w walker_per_interval] [-o overlap_decimal] [-r num_iterations]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

int read_histogram(const char *filename, std::vector<int> &h_expected_energy_spectrum, int *E_min, int *E_max){
    std::cout << filename;
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Failed to open file");
        return -1;
    }

    *E_min = INT_MAX;
    *E_max = INT_MIN;
    int value, count;
    h_expected_energy_spectrum.clear();

    int start_writing_zeros = 0;
    while (fscanf(file, "%d %d", &value, &count) != EOF)
    {
        if (count > 0)
        {
            if (value < *E_min)
                *E_min = value;
            if (value > *E_max)
                *E_max = value;

            h_expected_energy_spectrum.push_back(1);
            start_writing_zeros = 1;
        }
        else if (start_writing_zeros != 0)
        {
            h_expected_energy_spectrum.push_back(0);
        }
    }
    fclose(file);
    return 0;
}

IntervalResult generate_intervals(const int E_min, const int E_max, const int num_intervals, const int num_walker, const float overlap_decimal)
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

__global__ void init_lattice(signed char *lattice, const int nx, const int ny, const int num_lattices, const int seed)
{
    const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= nx * ny * num_lattices) return;

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

__global__ void find_spin_config_in_energy_range(signed char *d_lattice, signed char *d_interactions, const int nx, const int ny, const int num_lattices, const int seed, int *d_start, int *d_end, int *d_energy, int *d_offset_lattice)
{
    const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    
    int blockId = blockIdx.x;

    if (tid >= num_lattices) return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);
    
    int accept_spin_config = 0;
    
    while (accept_spin_config == 0){
        if (d_energy[tid] <= d_end[blockId] && d_energy[tid] >= d_start[blockId]){
            // TO DO d_H and d_G update
            accept_spin_config = 1;
        }
        else{
            double randval = curand_uniform(&st);
            randval *= (nx * ny - 1 + 0.999999);
            int random_index = (int)trunc(randval);

            int i = random_index / ny;
            int j = random_index % ny;

            int ipp = (i + 1 < nx) ? i + 1 : 0;
            int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
            int jpp = (j + 1 < ny) ? j + 1 : 0;
            int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;

            signed char energy_diff = -2 * d_lattice[d_offset_lattice[tid] + i * ny + j] * (d_lattice[d_offset_lattice[tid] + inn * ny + j] * d_interactions[nx * ny + inn * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jnn] * d_interactions[i * ny + jnn] + d_lattice[d_offset_lattice[tid] + ipp * ny + j] * d_interactions[nx * ny + i * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jpp] * d_interactions[i * ny + j]);

            d_energy[tid] += energy_diff;
            d_lattice[d_offset_lattice[tid] + i * ny + j] *= -1;
        }
    }
}

__global__ void init_interactions(signed char *interactions, const int nx, const int ny, const int num_lattices, const int seed, const double prob)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    while (tid < nx * ny * 2)
    {

        float randval = curand_uniform(&st);
        signed char val = (randval < prob) ? -1 : 1;

        interactions[tid] = val;

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void calc_energy(signed char *lattice, signed char *interactions, int *d_energy, int *d_offset_lattice, const int nx, const int ny, const int num_lattices)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int energy = 0;

    for (int l = 0; l < nx * ny; l++)
    {

        int i = l / ny;
        int j = l % ny;

        int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
        int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;

        energy += lattice[d_offset_lattice[tid] + i * ny + j] * (lattice[d_offset_lattice[tid] + inn * ny + j] * interactions[nx * ny + inn * ny + j] + lattice[d_offset_lattice[tid] + i * ny + jnn] * interactions[i * ny + jnn]);
    }

    d_energy[tid] = energy;
}

__global__ void check_energy_ranges(int *d_energy, int *d_start, int *d_end)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int check = 1;

    if (d_energy[tid] > d_end[blockIdx.x] || d_energy[tid] < d_start[blockIdx.x])
    {
        check = 0;
    }

    assert(check);
}

__device__ void fisher_yates(int *d_shuffle, int seed, unsigned long long *d_offset_iter)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int offset = blockDim.x * blockIdx.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_offset_iter[tid], &st);

    for (int i = blockDim.x - 1; i > 0; i--)
    {
        double randval = curand_uniform(&st);
        randval *= (i + 0.999999);
        int random_index = (int)trunc(randval);
        d_offset_iter[tid] += 1;

        int temp = d_shuffle[offset + i];
        d_shuffle[offset + i] = d_shuffle[offset + random_index];
        d_shuffle[offset + random_index] = temp;
    }
}

__global__ void init_indices(int *d_indices)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    d_indices[tid] = threadIdx.x;
}

__global__ void replica_exchange(
    int *d_offset_lattice, int *d_energy, int *d_start, int *d_end, int *d_indices,
    float *d_logG, int *d_offset_histogram, bool even, int seed, unsigned long long *d_offset_iter)
{

    if (blockIdx.x == (gridDim.x - 1)) return;
    
    if ((even && (blockIdx.x % 2 != 0)) || (!even && (blockIdx.x % 2 == 0))) return;

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    long long cid = static_cast<long long>(blockDim.x) * (blockIdx.x + 1);

    if (threadIdx.x == 0){
        fisher_yates(d_indices, seed, d_offset_iter);
    }

    __syncthreads();

    cid += d_indices[tid];

    // Check energy ranges
    if (d_energy[tid] > d_end[blockIdx.x + 1] || d_energy[tid] < d_start[blockIdx.x + 1]) return;
    if (d_energy[cid] > d_end[blockIdx.x] || d_energy[tid] < d_start[blockIdx.x]) return;

    double prob = min(1.0, exp(d_logG[d_offset_histogram[tid] + d_energy[tid] - d_start[blockIdx.x]] - d_logG[d_offset_histogram[tid] + d_energy[cid] - d_start[blockIdx.x]]) * exp(d_logG[d_offset_histogram[cid] + d_energy[cid] - d_start[blockIdx.x+1]] - d_logG[d_offset_histogram[cid] + d_energy[tid] - d_start[blockIdx.x+1]]));

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_offset_iter[tid], &st);

    if (curand_uniform(&st) < prob){

        int temp_off = d_offset_lattice[tid];
        int temp_energy = d_energy[tid];

        d_offset_lattice[tid] = d_offset_lattice[cid];
        d_energy[tid] = d_energy[cid];

        d_offset_lattice[cid] = temp_off;
        d_energy[cid] = temp_energy;
    }
    
    d_offset_iter[tid] += 1;
}

__global__ void check_histogram(int *d_H, int *d_offset_histogramm, int *d_end, int *d_start, double *d_factor, int nx, int ny, double alpha, int *d_expected_energy_spectrum, int len_energy_spectrum, int num_walker_total){

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int blockId = blockIdx.x;

    if (tid < num_walker_total){
        int min = INT_MAX;
        double average = 0;
        int len_reduced_energy_spectrum = 0;

        // Here is average and min calculation over all bins in histogram which correspond to values in expected energy spectrum
        for (int i = 0; i < (d_end[blockId] - d_start[blockId] + 1); i++){
            if (d_expected_energy_spectrum[d_start[blockId] + i - d_start[0]] == 1){
                if (d_H[d_offset_histogramm[tid] + i] < min){
                    min = d_H[d_offset_histogramm[tid] + i];
                }
                average += d_H[d_offset_histogramm[tid] + i];
                len_reduced_energy_spectrum += 1;
            }
        }

        if (len_reduced_energy_spectrum > 0)
        {
            average = average / len_reduced_energy_spectrum;

            if (min >= alpha * average)
            {
                d_factor[tid] = sqrt(d_factor[tid]);
                for (int i = 0; i < (d_end[blockId] - d_start[blockId] + 1); i++)
                {
                    d_H[d_offset_histogramm[tid] + i] = 0;
                }
            }
        }
        else
        {
            printf("Error histogram has no sufficient length to check for flatness on walker %lld. \n", tid);
        }
    }
}

__global__ void wang_landau(
    signed char *d_lattice, signed char *d_interactions, int *d_energy,
    int *d_start, int *d_end, int *d_H, float *d_logG, int *d_offset_histogramm, int *d_offset_lattice, const int num_iterations,
    const int nx, const int ny, const int seed, double *factor, unsigned long long *d_offset_iter, int *d_expected_energy_spectrum, int *d_newEnergies, int *foundFlag, 
    const int num_lattices, const double beta
    ){

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int blockId = blockIdx.x;

    if (tid >= num_lattices || factor[tid] <= exp(beta)) return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_offset_iter[tid], &st);

    for (int it = 0; it < num_iterations; it++){
        double randval = curand_uniform(&st);
        randval *= (nx * ny - 1 + 0.999999);
        int random_index = (int)trunc(randval);

        d_offset_iter[tid] += 1;

        int i = random_index / ny;
        int j = random_index % ny;

        int ipp = (i + 1 < nx) ? i + 1 : 0;
        int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
        int jpp = (j + 1 < ny) ? j + 1 : 0;
        int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;

        signed char energy_diff = -2 * d_lattice[d_offset_lattice[tid] + i * ny + j] * (d_lattice[d_offset_lattice[tid] + inn * ny + j] * d_interactions[nx * ny + inn * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jnn] * d_interactions[i * ny + jnn] + d_lattice[d_offset_lattice[tid] + ipp * ny + j] * d_interactions[nx * ny + i * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jpp] * d_interactions[i * ny + j]);

        int d_new_energy = d_energy[tid] + energy_diff;

        // If no new energy is found, set it to 0, else to tid + 1
        foundFlag[tid] = (d_expected_energy_spectrum[d_new_energy - d_start[0]] == 1) ? 0 : tid + 1;

        if (foundFlag[tid] != 0){
            printf("new_energy %d index in spectrum %d \n", d_new_energy, d_new_energy - d_start[0]);
            d_newEnergies[tid] = d_new_energy;
            return;
        }

        int index_old = d_offset_histogramm[tid] + d_energy[tid] - d_start[blockId];

        if (d_new_energy > d_end[blockId] || d_new_energy < d_start[blockId]){
            d_H[index_old] += 1;
            d_logG[index_old] += log(factor[tid]);
        }
        else{

            int index_new = d_offset_histogramm[tid] + d_new_energy - d_start[blockId];

            float prob = min(1.0, exp(d_logG[index_old] - d_logG[index_new]));

            if (curand_uniform(&st) < prob){
                d_lattice[d_offset_lattice[tid] + i * ny + j] *= -1;
                d_H[index_new] += 1;
                d_logG[index_new] += log(factor[tid]);
                d_energy[tid] = d_new_energy;
                d_offset_iter[tid] += 1;
            }
            else{
                d_H[index_old] += 1;
                d_logG[index_old] += log(factor[tid]);
            }
        }
    }
}

__global__ void init_offsets_histogramm(int *d_offset_histogramm, int *d_start, int *d_end)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    // Length of interval equal except last one --> length of array is given by num_threads_per_block * (length of interval + length of last interval)
    if (blockIdx.x == gridDim.x - 1)
    {
        d_offset_histogramm[tid] = (gridDim.x - 1) * blockDim.x * (d_end[0] - d_start[0] + 1) + threadIdx.x * (d_end[gridDim.x - 1] - d_start[gridDim.x - 1] + 1);
    }
    else
    {
        d_offset_histogramm[tid] = tid * (d_end[0] - d_start[0] + 1);
    }
}

__global__ void init_offsets_lattice(int *d_offset_lattice, int nx, int ny)
{
    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    d_offset_lattice[tid] = tid * nx * ny;
}

// It is important that the format of the histogramm file is like energy: count such that we can append a row with new_energy: 1
void handleNewEnergyError(int *new_energies, int *new_energies_flag, char *histogram_file, int num_walkers_total)
{
    std::cerr << "Error: Found new energy:" << std::endl;
    std::ofstream outfile;
    outfile.open(histogram_file, std::ios_base::app);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Could not open file " << histogram_file << std::endl;
        return;
    }

    for (int i = 0; i < num_walkers_total; i++)
    {
        if (new_energies_flag[i] != 0)
        {
            int walker_idx = new_energies_flag[i] - 1;
            int new_energy = new_energies[walker_idx];

            // Write the new energy to the histogram file
            outfile << new_energy << " 1" << std::endl;
        }
    }

    // Close the file
    outfile.close();
}

char *constructHistogramFilePath(float prob_interactions, int X, int Y, int seed)
{
    std::stringstream strstr;

    strstr << "init/prob_" << std::fixed << std::setprecision(6) << prob_interactions;
    strstr << "/X_" << X << "_Y_" << Y;
    strstr << "/seed_" << seed << "/histogram/histogram.txt";

    // Convert the stringstream to a string
    std::string filePathStr = strstr.str();

    // Allocate memory for the char* result and copy the string data to it
    char *filePathCStr = new char[filePathStr.length() + 1];
    std::strcpy(filePathCStr, filePathStr.c_str());

    return filePathCStr;
}

char *constructInteractionFilePath(float prob_interactions, int X, int Y, int seed)
{
    std::stringstream strstr;

    strstr << "init/prob_" << std::fixed << std::setprecision(6) << prob_interactions;
    strstr << "/X_" << X << "_Y_" << Y;
    strstr << "/seed_" << seed << "/interactions/interactions.txt";

    // Convert the stringstream to a string
    std::string filePathStr = strstr.str();

    // Allocate memory for the char* result and copy the string data to it
    char *filePathCStr = new char[filePathStr.length() + 1];
    std::strcpy(filePathCStr, filePathStr.c_str());

    return filePathCStr;
}

void read(std::vector<signed char> &lattice, std::string filename)
{

    std::ifstream inputFile(filename);

    if (!inputFile){
        std::cerr << "Unable to open file " << filename << std::endl;
        return;
    }

    int spin = 0;

    while (inputFile >> spin){
        lattice.push_back(static_cast<signed char>(spin));
    }
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

std::vector<signed char> init_lattice_with_pre_run_result(float prob, int seed, int x, int y, std::vector<int> h_start, std::vector<int> h_end, int num_intervals, int num_walkers_total, int num_walkers_per_interval){
    namespace fs = std::filesystem;
    std::ostringstream oss;
    oss << "init/prob_" << std::fixed << std::setprecision(6) << prob;
    oss << "/X_" << x << "_Y_" << y;
    oss << "/seed_" << seed;
    oss << "/lattice";

    std::string lattice_path = oss.str();
    std::vector<signed char> lattice_over_all_walkers;
    for(int interval_iterator = 0 ; interval_iterator < num_intervals; interval_iterator++){
        std::cout << interval_iterator;
        try {
            for (const auto& entry : fs::directory_iterator(lattice_path)) {
                // Check if the entry is a regular file and has a .txt extension
                if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                    // Extract the number from the filename
                    std::string filename = entry.path().stem().string(); // Get the filename without extension
                    std::regex regex("lattice_(-?\\d+)");
                    std::smatch match;
                    if (std::regex_search(filename, match, regex)) {
                        int number = std::stoi(match[1]);
                        // Check if the number is between interval boundaries
                        if (number >= h_start[interval_iterator] && number <= h_end[interval_iterator]) {
                            std::cout << "Processing file: " << entry.path() << " with energy: " << number << " for interval [" << h_start[interval_iterator] << ", " << h_end[interval_iterator] << std::endl;
                            for(int walker_per_interval_iterator = 0; walker_per_interval_iterator < num_walkers_per_interval;  walker_per_interval_iterator++){
                                read(lattice_over_all_walkers, entry.path().string());
                            }
                            break;
                        } 
                    } else {
                        std::cerr << "Unable to open file: " << entry.path() << std::endl;
                    }
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Filesystem error: " << e.what() << std::endl;
        }
    }
    return lattice_over_all_walkers;
}


/*
To Do:
    - Paths in construct histogram path, and interaction path
    - blockID to blockIdx.x to save storage?
    - d_indices array not needed, could in theory use shared memory in each fisher yates call
    - still the read and write of interactions such that we initialize each WL run with a specific histogram and interaction data
    - Store results and normalize
    - init flag for found new energy with seperate kernel and only update in wang landau inverted to current setting
    - New energies smarter way to update histogram
    - Concatenation of energy density results
*/

int main(int argc, char **argv){

    auto start = std::chrono::high_resolution_clock::now();

    const int seed = 42;

    Options options;
    parse_args(argc, argv, &options);
    
    const int num_walker_total = options.num_intervals * options.walker_per_interval;

    char *histogram_file = constructHistogramFilePath(options.prob_interactions, options.X, options.Y, seed);

    // Energy spectrum from pre_run
    std::vector<int> h_expected_energy_spectrum;
    if (read_histogram(histogram_file, h_expected_energy_spectrum, &options.E_min, &options.E_max) != 0){
        fprintf(stderr, "Error reading histogram file.\n");
        return 1;
    }
    const int len_energy_spectrum = h_expected_energy_spectrum.size();

    // Get interval information
    IntervalResult interval_result = generate_intervals(options.E_min, options.E_max, options.num_intervals, options.walker_per_interval, options.overlap_decimal);
    
    // Start end energies of the intervals
    int *d_start, *d_end;
    CHECK_CUDA(cudaMalloc(&d_start, options.num_intervals * sizeof(*d_start))); 
    CHECK_CUDA(cudaMalloc(&d_end, options.num_intervals * sizeof(*d_end)));
    CHECK_CUDA(cudaMemcpy(d_start, interval_result.h_start.data(), options.num_intervals * sizeof(*d_start), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_end, interval_result.h_end.data(), options.num_intervals * sizeof(*d_start), cudaMemcpyHostToDevice));

    // Histogramm and G array
    int *d_H;
    CHECK_CUDA(cudaMalloc(&d_H, interval_result.len_histogram_over_all_walkers * sizeof(*d_H)));
    CHECK_CUDA(cudaMemset(d_H, 0, interval_result.len_histogram_over_all_walkers * sizeof(*d_H)));

    float *d_logG;
    CHECK_CUDA(cudaMalloc(&d_logG, interval_result.len_histogram_over_all_walkers * sizeof(*d_logG)));
    CHECK_CUDA(cudaMemset(d_logG, 0, interval_result.len_histogram_over_all_walkers * sizeof(*d_logG)));

    // Offset histograms, lattice, seed_iterator
    int *d_offset_histogramm, *d_offset_lattice;
    unsigned long long *d_offset_iter;
    CHECK_CUDA(cudaMalloc(&d_offset_histogramm, num_walker_total * sizeof(*d_offset_histogramm)));
    CHECK_CUDA(cudaMalloc(&d_offset_lattice, num_walker_total * sizeof(*d_offset_lattice)));
    CHECK_CUDA(cudaMalloc(&d_offset_iter, num_walker_total * sizeof(*d_offset_iter)));
    CHECK_CUDA(cudaMemset(d_offset_iter, 0, num_walker_total * sizeof(*d_offset_iter)));
    
    // f Factors for each walker
    std::vector<double> h_factor(num_walker_total, std::exp(1));

    double *d_factor;
    CHECK_CUDA(cudaMalloc(&d_factor, num_walker_total * sizeof(*d_factor)));
    CHECK_CUDA(cudaMemcpy(d_factor, h_factor.data(), num_walker_total * sizeof(*d_factor), cudaMemcpyHostToDevice));

    // Indices used for replica exchange later
    int *d_indices;
    CHECK_CUDA(cudaMalloc(&d_indices, num_walker_total * sizeof(*d_indices)));

    // lattice, interactions
    signed char *d_lattice, *d_interactions;
    CHECK_CUDA(cudaMalloc(&d_lattice, num_walker_total * options.X * options.Y * sizeof(*d_lattice)));
    CHECK_CUDA(cudaMalloc(&d_interactions, options.X * options.Y * 2 * sizeof(*d_interactions)));

    // Hamiltonian of lattices
    int *d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, num_walker_total * sizeof(*d_energy)));
    CHECK_CUDA(cudaMemset(d_energy, 0, num_walker_total * sizeof(*d_energy)));

    // Binary indicator of energies were found or not
    int *d_expected_energy_spectrum;
    CHECK_CUDA(cudaMalloc(&d_expected_energy_spectrum, h_expected_energy_spectrum.size() * sizeof(*d_expected_energy_spectrum)));
    CHECK_CUDA(cudaMemcpy(d_expected_energy_spectrum, h_expected_energy_spectrum.data(), h_expected_energy_spectrum.size() * sizeof(*d_expected_energy_spectrum), cudaMemcpyHostToDevice));

    // To catch energies which are outside of expected spectrum
    int *d_newEnergies, *d_foundNewEnergyFlag; 
    CHECK_CUDA(cudaMalloc(&d_newEnergies, num_walker_total * sizeof(*d_newEnergies)));
    CHECK_CUDA(cudaMalloc(&d_foundNewEnergyFlag, num_walker_total * sizeof(*d_foundNewEnergyFlag)));

    /*
    ----------------------------------------------
    ------------ Actual WL Starts Now ------------
    ----------------------------------------------
    */

    // Initialization of lattices, interactions, offsets and indices
    // init_lattice<<<num_walker_total, options.X*options.Y>>>(d_lattice, options.X, options.Y, num_walker_total, seed);
    init_offsets_lattice<<<options.num_intervals, options.walker_per_interval>>>(d_offset_lattice, options.X, options.Y);
    init_offsets_histogramm<<<options.num_intervals, options.walker_per_interval>>>(d_offset_histogramm, d_start, d_end);
    init_indices<<<options.num_intervals, options.walker_per_interval>>>(d_indices);
    
    char *interaction_file = constructInteractionFilePath(options.prob_interactions, options.X, options.Y, seed);
    std::vector<signed char> h_interactions;
    read(h_interactions, interaction_file);
    CHECK_CUDA(cudaMemcpy(d_interactions, h_interactions.data(), options.X * options.Y * 2 * sizeof(*d_interactions), cudaMemcpyHostToDevice));
    
    
    std::vector<signed char> h_lattice = init_lattice_with_pre_run_result(options.prob_interactions, seed, options.X, options.Y, interval_result.h_start, interval_result.h_end, options.num_intervals, num_walker_total, options.walker_per_interval);
    CHECK_CUDA(cudaMemcpy(d_lattice, h_lattice.data(), num_walker_total * options.X * options.Y * sizeof(*d_lattice), cudaMemcpyHostToDevice));
    // Calculate energy and find right configurations
    calc_energy<<<options.num_intervals, options.walker_per_interval>>>(d_lattice, d_interactions, d_energy, d_offset_lattice, options.X, options.Y, num_walker_total);    
    cudaDeviceSynchronize();
    check_energy_ranges<<<options.num_intervals, options.walker_per_interval>>>(d_energy, d_start, d_end);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time before Wang Landau has started: " << elapsed.count() << " seconds" << std::endl;

    float max_factor = std::exp(1);
    int max_newEnergyFlag = 0;
    
    while (max_factor > std::exp(options.beta)){
        
        printf("Max factor %2f \n", max_factor);

        wang_landau<<<options.num_intervals, options.walker_per_interval>>>(d_lattice, d_interactions, d_energy, d_start, d_end, d_H, d_logG, d_offset_histogramm, d_offset_lattice, options.num_iterations, options.X, options.Y, seed + 3, d_factor, d_offset_iter, d_expected_energy_spectrum, d_newEnergies, d_foundNewEnergyFlag, num_walker_total, options.beta);
        cudaDeviceSynchronize();

        // get max of found new energy flag array to condition break and update the histogramm file with value in new energy array
        thrust::device_ptr<int> d_newEnergyFlag_ptr(d_foundNewEnergyFlag);
        thrust::device_ptr<int> max_newEnergyFlag_ptr = thrust::max_element(d_newEnergyFlag_ptr, d_newEnergyFlag_ptr + num_walker_total);
        max_newEnergyFlag = *max_newEnergyFlag_ptr;

        // If flag shows new energies get the device arrays containing these to the host, update histogramm file and print error message.
        if (max_newEnergyFlag != 0)
        {
            int h_newEnergies[num_walker_total];
            int h_newEnergyFlag[num_walker_total];
            CHECK_CUDA(cudaMemcpy(h_newEnergies, d_newEnergies, num_walker_total * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_newEnergyFlag, d_foundNewEnergyFlag, num_walker_total * sizeof(int), cudaMemcpyDeviceToHost));

            handleNewEnergyError(h_newEnergies, h_newEnergyFlag, histogram_file, num_walker_total);
            return 1;
        }
        
        check_histogram<<<options.num_intervals, options.walker_per_interval>>>(d_H, d_offset_histogramm, d_end, d_start, d_factor, options.X, options.Y, options.alpha, d_expected_energy_spectrum, len_energy_spectrum, num_walker_total);
        cudaDeviceSynchronize();

        // get max factor over walkers for abort condition of while loop
        thrust::device_ptr<double> d_factor_ptr(d_factor);
        thrust::device_ptr<double> max_factor_ptr = thrust::max_element(d_factor_ptr, d_factor_ptr + num_walker_total);
        max_factor = *max_factor_ptr;
        
        replica_exchange<<<options.num_intervals, options.walker_per_interval>>>(d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG, d_offset_histogramm, true, seed + 3, d_offset_iter);
        replica_exchange<<<options.num_intervals, options.walker_per_interval>>>(d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG, d_offset_histogramm, false, seed + 3, d_offset_iter);
    };

    /*
    ---------------------------------------------
    --------------Post Processing ---------------
    ---------------------------------------------
    */

    std::vector<float> h_log_density_per_walker(interval_result.len_histogram_over_all_walkers);
    CHECK_CUDA(cudaMemcpy(h_log_density_per_walker.data(), d_logG, interval_result.len_histogram_over_all_walkers * sizeof(*d_logG), cudaMemcpyDeviceToHost));

    std::ofstream f_log_density;
    f_log_density.open("log_density.txt");

    int index_h_log_g = 0;
    if (f_log_density.is_open())
    {
        for (int i = 0; i < options.num_intervals; i++)
        {

            int start_energy = interval_result.h_start[i];
            int end_energy = interval_result.h_end[i];
            int len_int = interval_result.h_end[i] - interval_result.h_start[i] + 1;

            for (int j = 0; j < options.walker_per_interval; j++)
            {
                for (int k = 0; k < len_int; k++)
                {
                    f_log_density << (int)interval_result.h_start[i] + k << " : " << (float)h_log_density_per_walker[index_h_log_g] << " ,";
                    index_h_log_g += 1;
                }
                f_log_density << std::endl;
            }
        }

    }
    f_log_density.close();
    
}