#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <getopt.h>
#include <vector>
#include <thread>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <unistd.h> // For Sleep
#include "./header/cudamacro.h"

const unsigned int THREADS = 128;

typedef struct
{
    std::vector<int> h_start;
    std::vector<int> h_end;
    long long len_histogram_over_all_walkers;
    int len_interval;
} IntervalResult;

typedef struct
{
    int E_min;
    int E_max;
    int num_intervals;
    int walker_per_interval;
    float overlap_decimal;
    char *histogram_file;
    int num_iterations;
} Options;

void parse_args(int argc, char *argv[], Options *options, int L)
{
    // overlap decimal is more like the reciprocal non overlap parameter here, i.e. 0 as overlap_decimal is full overlap of intervals.

    int opt;
    options->histogram_file = NULL;
    options->walker_per_interval = 10; // Default value for num_walker
    options->overlap_decimal = 0.25;   // 75% overlap of neighboring intervals as default overlap
    options->E_min = -2 * L * L;
    options->E_max = 2 * L * L;
    options->num_intervals = 1;
    options->num_iterations = 1000;

    while (1)
    {
        int option_index = 0;
        static struct option long_options[] = {
            {"Emin", 1, 0, 'm'},
            {"Emax", 1, 0, 'M'},
            {"num_intervals", 1, 0, 'i'},
            {"walker_per_interval", 1, 0, 'w'},
            {"histogram_file", 1, 0, 'f'},
            {"overlap_decimal", 1, 0, 'o'},
            {"num_iterations", 1, 0, 'r'},
            {0, 0, 0, 0}};
        opt = getopt_long(argc, argv, "m:M:i:w:f:o:r:", long_options, &option_index);
        if (opt == -1)
            break;
        switch (opt)
        {
        case 'm':
            options->E_min = std::atoi(optarg);
            break;
        case 'M':
            options->E_max = std::atoi(optarg);
            break;
        case 'i':
            options->num_intervals = std::atoi(optarg);
            break;
        case 'w':
            options->walker_per_interval = std::atoi(optarg);
            break;
        case 'f':
            options->histogram_file = optarg;
            break;
        case 'o':
            options->overlap_decimal = std::atof(optarg);
            break;
        case 'r':
            options->num_iterations = std::atoi(optarg);
            break;
        default:
            fprintf(stderr, "Usage: %s [-i num_intervals] [-m E_min] [-M E_max] [-w walker_per_interval] [-f histogram_file] [-o overlap_decimal] [-r num_iterations]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (!options->histogram_file)
    {
        fprintf(stderr, "No histogram file is given and default E_min E_max were assigned.\n");
        ;
    }
}

int read_histogram(const char *filename, std::vector<int> &nonNullBins, int *E_min, int *E_max)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Failed to open file");
        return -1;
    }

    *E_min = INT_MAX;
    *E_max = INT_MIN;
    int value, count;
    nonNullBins.clear();

    // Use a set to keep track of unique bins
    std::set<int> uniqueBins;

    while (fscanf(file, "%d : %d", &value, &count) != EOF)
    {
        if (count > 0)
        {
            if (value < *E_min)
                *E_min = value;
            if (value > *E_max)
                *E_max = value;
            if (uniqueBins.find(value) == uniqueBins.end())
            {
                nonNullBins.push_back(value);
                uniqueBins.insert(value);
            }
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

__global__ void init_lattice(signed char *lattice, const int nx, const int ny, const int num_lattices, const int seed, const float prob)
{
    const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x; //

    if (tid >= nx * ny * num_lattices)
        return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st); // offset??

    float randval = curand_uniform(&st);
    signed char val = (randval < prob) ? -1 : 1;

    lattice[tid] = val;
}

__global__ void find_spin_config_in_energy_range(signed char *d_lattice, signed char *d_interactions, const int nx, const int ny, const int num_lattices, const int seed, int *d_start, int *d_end, int *d_energy, int *d_offset_lattice)
{
    const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    int blockId = blockIdx.x;

    if (tid >= num_lattices) // only for each walker single thread
        return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);
    int accept_spin_config = 0;
    while (accept_spin_config == 0)
    {
        if (d_energy[tid] <= d_end[blockId] && d_energy[tid] >= d_start[blockId])
        {
            accept_spin_config = 1;
        }
        else
        {
            // Generate random int --> is that actually uniformly?
            double randval = curand_uniform(&st);
            randval *= (nx * ny - 1 + 0.999999);
            int random_index = (int)trunc(randval);

            int i = random_index / ny;
            int j = random_index % ny;

            // Set up periodic boundary conditions
            int ipp = (i + 1 < nx) ? i + 1 : 0;
            int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
            int jpp = (j + 1 < ny) ? j + 1 : 0;
            int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;

            // Nochmal checken
            signed char energy_diff = -2 * d_lattice[d_offset_lattice[tid] + i * ny + j] * (d_lattice[d_offset_lattice[tid] + inn * ny + j] * d_interactions[nx * ny + inn * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jnn] * d_interactions[i * ny + jnn] + d_lattice[d_offset_lattice[tid] + ipp * ny + j] * d_interactions[nx * ny + i * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jpp] * d_interactions[i * ny + j]);

            d_energy[tid] += energy_diff;
            d_lattice[d_offset_lattice[tid] + i * ny + j] = (-1) * d_lattice[d_offset_lattice[tid] + i * ny + j];
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

    tid += blockDim.x * gridDim.x;
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

__device__ void fisher_yates(int *d_shuffle, int seed, int *d_iter)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int offset = blockDim.x * blockIdx.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_iter[tid], &st);

    for (int i = blockDim.x - 1; i > 0; i--)
    {
        double randval = curand_uniform(&st);
        randval *= (i + 0.999999);
        int random_index = (int)trunc(randval);
        d_iter[tid] += 1;

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
    float *d_logG, bool even, int seed, int *d_iter)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    // Check if last block
    if (blockIdx.x == (gridDim.x - 1))
    {
        return;
    }

    // change index
    long long cid = static_cast<long long>(blockDim.x) * (blockIdx.x + 1);

    if (threadIdx.x == 0)
    {
        fisher_yates(d_indices, seed, d_iter);
    }

    // Synchronize

    if (even)
    {
        if (blockIdx.x % 2 != 0)
            return;
    }
    else
    {
        if (blockIdx.x % 2 != 1)
            return;
    }

    cid += d_indices[tid];

    // Check energy ranges
    if (d_energy[tid] > d_end[blockIdx.x + 1] || d_energy[tid] < d_start[blockIdx.x + 1])
        return;
    if (d_energy[cid] > d_end[blockIdx.x] || d_energy[tid] < d_start[blockIdx.x])
        return;

    double prob = min(1.0, d_logG[d_energy[tid]] / d_logG[d_energy[tid]] * d_logG[d_energy[cid]] / d_logG[d_energy[cid]]);

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_iter[tid], &st);

    if (curand_uniform(&st) < prob)
    {

        int temp_off = d_offset_lattice[tid];
        int temp_energy = d_energy[tid];

        d_offset_lattice[tid] = d_offset_lattice[cid];
        d_energy[tid] = d_energy[cid];

        d_offset_lattice[cid] = temp_off;
        d_energy[cid] = temp_energy;

        d_iter[tid] += 1;
    }
}

__global__ void check_histogram(int *d_H, int *d_offset_histogramm, int *d_end, int *d_start, float *d_factor, int nx, int ny, double alpha, int *expected_energy_spectrum, int len_energy_spectrum, int num_walker_total)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int blockId = blockIdx.x;

    if (tid < num_walker_total)
    {
        int min = INT_MAX;

        double average = 0;

        int len_reduced_energy_spectrum = 0;

        // Here is average and min calculation over all bins in histogram which correspond to values in expected energy spectrum
        for (int i = 0; i < (d_end[blockId] - d_start[blockId] + 1); i++) // index range for full histogram on thread
        {
            bool found_energy_flag = false;
            for (int j = 0; j < len_energy_spectrum; j++) // index range for expected energy spectrum
            {
                if (d_start[blockId] + i == expected_energy_spectrum[j])
                {
                    if (d_H[d_offset_histogramm[tid] + i] < min)
                    {
                        min = d_H[d_offset_histogramm[tid] + i];
                    }
                    average += d_H[d_offset_histogramm[tid] + i];
                    len_reduced_energy_spectrum++;
                    found_energy_flag = true;
                }
                if (found_energy_flag)
                {
                    break;
                }
            }
        }

        if (len_reduced_energy_spectrum > 0)
        {
            average = average / len_reduced_energy_spectrum;
        }
        else
        {
            printf("Error histogram has no length - no average is computable.\n");
        }

        if (min >= alpha * average)
        {
            d_factor[tid] = sqrt(d_factor[tid]);
        }
    }
}

__global__ void wang_landau(
    signed char *d_lattice, signed char *d_interactions, int *d_energy,
    int *d_start, int *d_end, int *d_H, float *d_logG, int *d_offset_histogramm, int *d_offset_lattice, const int num_iterations,
    const int nx, const int ny, const int seed, float *factor, int *d_iter, int *d_nonNullEnergies, int *d_newEnergies, int *foundFlag, const int num_lattices, const double beta, int len_energy_spectrum)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int blockId = blockIdx.x;

    if (tid >= num_lattices || factor[tid] <= std::exp(beta)) // for each walker single thread and walker with minimal factor shall stop
        return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_iter[tid], &st);

    for (int it = 0; it < num_iterations; it++)
    {

        // Generate random int --> is that actually uniformly?
        double randval = curand_uniform(&st);
        randval *= (nx * ny - 1 + 0.999999);
        int random_index = (int)trunc(randval);

        d_iter[tid] += 1;

        int i = random_index / ny;
        int j = random_index % ny;

        // Set up periodic boundary conditions
        int ipp = (i + 1 < nx) ? i + 1 : 0;
        int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
        int jpp = (j + 1 < ny) ? j + 1 : 0;
        int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;

        // Nochmal checken
        signed char energy_diff = -2 * d_lattice[d_offset_lattice[tid] + i * ny + j] * (d_lattice[d_offset_lattice[tid] + inn * ny + j] * d_interactions[nx * ny + inn * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jnn] * d_interactions[i * ny + jnn] + d_lattice[d_offset_lattice[tid] + ipp * ny + j] * d_interactions[nx * ny + i * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jpp] * d_interactions[i * ny + j]);

        int d_new_energy = d_energy[tid] + energy_diff;

        // Initialize the found new energy flag with thread id + 1 as if energy match is found it will be set to zero and we can by this map non zero entries to tid where new energy appeared.
        foundFlag[tid] = tid + 1;

        // check for found new energy
        for (int i = 0; i < len_energy_spectrum; i++)
        {
            if (d_new_energy == d_nonNullEnergies[i])
            {
                foundFlag[tid] = 0;
            }
            if (foundFlag[tid] == 0)
            {
                break;
            }
        }
        if (foundFlag[tid] != 0)
        {
            d_newEnergies[tid] = d_new_energy;
            return;
        }

        int index_old = d_offset_histogramm[tid] + d_energy[tid] - d_start[blockId];

        if (d_new_energy > d_end[blockId] || d_new_energy < d_start[blockId])
        {
            d_H[index_old] += 1;
            d_logG[index_old] += log(factor[tid]);
        }
        else
        {

            int index_new = d_offset_histogramm[tid] + d_new_energy - d_start[blockId];

            float prob = exp(d_logG[index_old] - d_logG[index_new]);

            if (curand_uniform(&st) < prob)
            {
                d_lattice[d_offset_lattice[tid] + i * ny + j] *= -1;

                d_H[index_new] += 1;
                d_logG[index_new] += log(factor[tid]);

                d_energy[tid] = d_new_energy;

                d_iter[tid] += 1;
            }
            else
            {
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
            outfile << new_energy << " : 1" << std::endl;
        }
    }

    // Close the file
    outfile.close();
}

int main(int argc, char **argv)
{

    // General parameter which stay regularly unchanged
    const int seed = 42;
    const int num_iterations = 1000; // iteration count after which flattness gets checked and replica exchange executed
    const double alpha = 0.8;        // condition for histogram
    const double beta = 0.1;         // end condition for factor

    // Model parameter
    const int L = 4;
    const float prob_interactions = 0; // prob of error
    const float prob_spins = 0.4;      // prob of down spin

    // Input args parsing
    Options options;
    parse_args(argc, argv, &options, L);
    std::vector<int> nonNullEnergies;

    if (options.histogram_file)
    {
        if (read_histogram(options.histogram_file, nonNullEnergies, &options.E_min, &options.E_max) != 0)
        {
            fprintf(stderr, "Error reading histogram file.\n");
            return 1;
        }
    }
    int len_energy_spectrum = nonNullEnergies.size();
    int num_walker_total = options.num_intervals * options.walker_per_interval;

    // Get interval information
    IntervalResult interval_result = generate_intervals(options.E_min, options.E_max, options.num_intervals, options.walker_per_interval, options.overlap_decimal);

    int *d_start, *d_end;
    CHECK_CUDA(cudaMalloc(&d_start, options.num_intervals * sizeof(*d_start))); // array of start energies of intervals
    CHECK_CUDA(cudaMalloc(&d_end, options.num_intervals * sizeof(*d_end)));     // array of end energies of intervals
    CHECK_CUDA(cudaMemcpy(d_start, interval_result.h_start.data(), options.num_intervals * sizeof(*d_start), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_end, interval_result.h_end.data(), options.num_intervals * sizeof(*d_start), cudaMemcpyHostToDevice));

    // Histogramm and G array
    int *d_H;
    CHECK_CUDA(cudaMalloc(&d_H, interval_result.len_histogram_over_all_walkers * sizeof(*d_H)));
    CHECK_CUDA(cudaMemset(d_H, 0, interval_result.len_histogram_over_all_walkers * sizeof(*d_H)));

    float *d_logG;
    CHECK_CUDA(cudaMalloc(&d_logG, interval_result.len_histogram_over_all_walkers * sizeof(*d_logG)));
    CHECK_CUDA(cudaMemset(d_logG, 0, interval_result.len_histogram_over_all_walkers * sizeof(*d_logG)));

    int *d_offset_histogramm;
    CHECK_CUDA(cudaMalloc(&d_offset_histogramm, num_walker_total * sizeof(*d_offset_histogramm)));

    // offset in big lattice array for each walkers lattice
    int *d_offset_lattice;
    CHECK_CUDA(cudaMalloc(&d_offset_lattice, num_walker_total * sizeof(*d_offset_lattice)));

    float *d_factor;
    CHECK_CUDA(cudaMalloc(&d_factor, num_walker_total * sizeof(*d_factor)));
    float h_factor[num_walker_total];
    for (int i = 0; i < num_walker_total; ++i)
    {
        h_factor[i] = std::exp(1.0);
    }
    cudaMemcpy(d_factor, h_factor, num_walker_total * sizeof(float), cudaMemcpyHostToDevice);

    int *d_indices;
    CHECK_CUDA(cudaMalloc(&d_indices, num_walker_total * sizeof(*d_indices)));

    int *d_iter;
    CHECK_CUDA(cudaMalloc(&d_iter, num_walker_total * sizeof(*d_iter)));
    CHECK_CUDA(cudaMemset(d_iter, 0, num_walker_total * sizeof(*d_iter)));

    // lattice, interactions
    signed char *d_lattice;
    CHECK_CUDA(cudaMalloc(&d_lattice, num_walker_total * L * L * sizeof(*d_lattice)));

    signed char *d_interactions;
    CHECK_CUDA(cudaMalloc(&d_interactions, L * L * 2 * sizeof(*d_interactions)));

    // Hamiltonian of lattices
    int *d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, num_walker_total * sizeof(*d_energy)));
    CHECK_CUDA(cudaMemset(d_energy, 0, num_walker_total * sizeof(*d_energy)));

    // energies with non zero counts on device
    int *d_nonNullEnergies;
    CHECK_CUDA(cudaMalloc((void **)&d_nonNullEnergies, nonNullEnergies.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_nonNullEnergies, nonNullEnergies.data(), nonNullEnergies.size() * sizeof(int), cudaMemcpyHostToDevice));

    int *d_newEnergies; // To catch energies which are outside of expected spectrum
    CHECK_CUDA(cudaMalloc((void **)&d_newEnergies, num_walker_total * sizeof(int)));

    int *d_foundNewEnergyFlag;
    CHECK_CUDA(cudaMalloc(&d_foundNewEnergyFlag, num_walker_total * sizeof(int)));

    const int blocks_init = (L * L * num_walker_total + THREADS - 1) / THREADS; // why this as basic block count as per spin a thread only needed in init, or?

    init_lattice<<<blocks_init, THREADS>>>(d_lattice, L, L, num_walker_total, seed, prob_spins);
    init_interactions<<<blocks_init, THREADS>>>(d_interactions, L, L, num_walker_total, seed + 1, prob_interactions);
    init_offsets_lattice<<<options.num_intervals, options.walker_per_interval>>>(d_offset_lattice, L, L);
    init_offsets_histogramm<<<options.num_intervals, options.walker_per_interval>>>(d_offset_histogramm, d_start, d_end);
    calc_energy<<<options.num_intervals, options.walker_per_interval>>>(d_lattice, d_interactions, d_energy, d_offset_lattice, L, L, num_walker_total);

    find_spin_config_in_energy_range<<<(num_walker_total + options.walker_per_interval - 1) / options.walker_per_interval, options.walker_per_interval>>>(d_lattice, d_interactions, L, L, num_walker_total, seed + 2, d_start, d_end, d_energy, d_offset_lattice);

    check_energy_ranges<<<options.num_intervals, options.walker_per_interval>>>(d_energy, d_start, d_end);

    init_indices<<<options.num_intervals, options.walker_per_interval>>>(d_indices); // for replica exchange

    float max_factor = std::exp(1);
    int max_newEnergyFlag = 0;

    while (max_factor > std::exp(beta))
    {
        // execute wang landau updates with given number of iterations
        wang_landau<<<options.num_intervals, options.walker_per_interval>>>(d_lattice, d_interactions, d_energy, d_start, d_end, d_H, d_logG, d_offset_histogramm, d_offset_lattice, num_iterations, L, L, seed + 3, d_factor, d_iter, d_nonNullEnergies, d_newEnergies, d_foundNewEnergyFlag, num_walker_total, beta, len_energy_spectrum);

        // get max factor over walkers for abort condition of while loop
        cudaDeviceSynchronize();
        thrust::device_ptr<float> d_factor_ptr(d_factor);
        thrust::device_ptr<float> max_factor_ptr = thrust::max_element(d_factor_ptr, d_factor_ptr + num_walker_total);
        max_factor = *max_factor_ptr;

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

            handleNewEnergyError(h_newEnergies, h_newEnergyFlag, options.histogram_file, num_walker_total);
            return 1;
        }

        // check flatness of histogram
        check_histogram<<<options.num_intervals, options.walker_per_interval>>>(d_H, d_offset_histogramm, d_end, d_start, d_factor, L, L, alpha, d_nonNullEnergies, len_energy_spectrum, num_walker_total);

        replica_exchange<<<options.num_intervals, options.walker_per_interval>>>(d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG, true, seed + 4, d_iter);
        replica_exchange<<<options.num_intervals, options.walker_per_interval>>>(d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG, false, seed + 5, d_iter);
    };

    std::vector<int> h_histogram_per_walker(interval_result.len_histogram_over_all_walkers);

    CHECK_CUDA(cudaMemcpy(h_histogram_per_walker.data(), d_H, interval_result.len_histogram_over_all_walkers * sizeof(*d_H), cudaMemcpyDeviceToHost));

    std::vector<int> energies_histogram;

    for (int i = 0; i < options.num_intervals; i++)
    {

        int start_energy = interval_result.h_start[i];
        int end_energy = interval_result.h_end[i];
        int len_int = interval_result.h_end[i] - interval_result.h_start[i] + 1;

        for (int j = 0; j < options.walker_per_interval; j++)
        {
            for (int k = 0; k < len_int; k++)
            {
                energies_histogram.push_back(interval_result.h_start[i] + k);
            }
        }
    }

    std::ofstream f;
    f.open("histogramm_afterRun.txt");

    if (f.is_open())
    {
        for (int i = 0; i < interval_result.len_histogram_over_all_walkers; i++)
        {
            f << (int)energies_histogram[i] << " : " << (int)h_histogram_per_walker[i];
            f << std::endl;
        }
    }
    f.close();

    return 0;
}