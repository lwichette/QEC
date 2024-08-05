#ifndef WLUTILS_H
#define WLUTILS_H

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
#include <chrono>

typedef struct {
    std::vector<int> h_start;
    std::vector<int> h_end;
    long long len_histogram_over_all_walkers;
    int len_interval;
} IntervalResult;

typedef struct {
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
} Options;

typedef struct{
    int new_energy;
    int i;
    int j; 
} RBIM;

void parse_args(int argc, char *argv[], Options *options);

IntervalResult generate_intervals(const int E_min, const int E_max, int num_intervals, int num_walker, float overlap_decimal);

void writeToFile(const std::string& filename, const signed char* data, int nx_w, int ny);

void write(signed char* array, const std::string& filename, long nx, long ny, int num_lattices, bool lattice, const std::vector<int>& energies = std::vector<int>());

void create_directory(std::string path);

void write_histograms(unsigned long long *d_H, std::string path_histograms, int len_histogram, int seed, int E_min);

int read_histogram(const char *filename, std::vector<int> &h_expected_energy_spectrum, int *E_min, int *E_max);

void read(std::vector<signed char> &lattice, std::string filename);

void handleNewEnergyError(int *new_energies, int *new_energies_flag, char *histogram_file, int num_walkers_total);

char *constructFilePath(float prob_interactions, int X, int Y, int seed, std::string type);

std::vector<signed char> get_lattice_with_pre_run_result(float prob, int seed, int x, int y, std::vector<int> h_start, std::vector<int> h_end, int num_intervals, int num_walkers_total, int num_walkers_per_interval);

__global__ void init_lattice(signed char* lattice, const int nx, const int ny, const int num_lattices, const int seed);

__global__ void init_interactions(signed char* interactions, const int nx, const int ny, const int num_lattices, const int seed, const double prob);

__global__ void calc_energy(signed char *lattice, signed char *interactions, int *d_energy, int *d_offset_lattice, const int nx, const int ny, const int num_lattices);

__global__ void calc_energy_pre_run(signed char* lattice, signed char* interactions, int* d_energy, const int nx, const int ny, const int num_lattices);

__global__ void wang_landau_pre_run(signed char *d_lattice, signed char *d_interactions, int *d_energy, unsigned long long *d_H, int* d_iter, int *d_found_interval, signed char *d_store_lattice, const int E_min, const int E_max, const int num_iterations, const int nx, const int ny, const int seed, const int len_interval, const int found_interval);

__global__ void wang_landau(
    signed char *d_lattice, signed char *d_interactions, int *d_energy,
    int *d_start, int *d_end, unsigned long long *d_H, double *d_logG, int *d_offset_histogramm, int *d_offset_lattice, const int num_iterations,
    const int nx, const int ny, const int seed, double *factor, unsigned long long *d_offset_iter, int *d_expected_energy_spectrum, int *d_newEnergies, int *foundFlag, 
    const int num_lattices, const double beta, signed char *d_cond
);

__global__ void check_histogram(unsigned long long *d_H, int *d_offset_histogramm, int *d_end, int *d_start, double *d_factor, int nx, int ny, double alpha, int *d_expected_energy_spectrum, int len_energy_spectrum, int num_walker_total, signed char *d_cond);

__global__ void find_spin_config_in_energy_range(signed char *d_lattice, signed char *d_interactions, const int nx, const int ny, const int num_lattices, const int seed, int *d_start, int *d_end, int *d_energy, int *d_offset_lattice);

__global__ void check_energy_ranges(int *d_energy, int *d_start, int *d_end);

__device__ void fisher_yates(int *d_shuffle, int seed, unsigned long long *d_offset_iter);

__device__ void store_lattice(signed char *d_lattice, int *d_energy, int* d_found_interval, signed char* d_store_lattice, const int E_min, const int nx, const int ny, const long long tid, const int len_interval);

__global__ void init_indices(int *d_indices);

__global__ void init_offsets_histogramm(int *d_offset_histogramm, int *d_start, int *d_end);

__global__ void init_offsets_lattice(int *d_offset_lattice, int nx, int ny);

__global__ void replica_exchange(
    int *d_offset_lattice, int *d_energy, int *d_start, int *d_end, int *d_indices,
    double *d_logG, int *d_offset_histogram, bool even, int seed, unsigned long long *d_offset_iter);

__global__ void print_finished_walker_ratio(double *d_factor, int num_walker_total, const double exp_beta, double *d_finished_walkers_ratio);

__device__ RBIM random_bond_ising(
    signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_offset_lattice, unsigned long long *d_offset_iter, 
    curandStatePhilox4_32_10_t *st, const long long tid, const int nx, const int ny
    );

#endif // UTILS_H