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
#include <thrust/extrema.h>
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
    int seed_histogram;
    int seed_run;
    char logical_error_type;
    int boundary_type;
} Options;

typedef struct{
    int new_energy;
    int i;
    int j; 
} RBIM;

void parse_args(int argc, char *argv[], Options *options);

IntervalResult generate_intervals(const int E_min, const int E_max, int num_intervals, int num_walker, float overlap_decimal);

void writeToFile(const std::string& filename, const signed char* data, int nx_w, int ny);

void write(signed char *array_host, const std::string& filename, long nx, long ny, int num_lattices, bool lattice, const int *energies = NULL);

void create_directory(std::string path);

void write_histograms(unsigned long long *d_H, std::string path_histograms, int len_histogram, int seed, int E_min);

int read_histogram(const char *filename, std::vector<int> &h_expected_energy_spectrum, int *E_min, int *E_max);

void read(std::vector<signed char> &lattice, std::string filename);

void handleNewEnergyError(int *new_energies, int *new_energies_flag, char *histogram_file, int num_walkers_total);

void calc_energy(int blocks, int threads, const int boundary_type, signed char *lattice, signed char *interactions, int *d_energy, int *d_offset_lattice, const int nx, const int ny, const int total_walker, const int walker_per_interactions);

void result_handling(Options options, IntervalResult interval_result, double *d_logG);

char *constructFilePath(float prob_interactions, int X, int Y, int seed, std::string type, char error_class, int boundary_type);

std::vector<signed char> get_lattice_with_pre_run_result(float prob, int seed, int x, int y, std::vector<int> h_start, std::vector<int> h_end, int num_intervals, int num_walkers_total, int num_walkers_per_interval, char error_class, int boundary_type);

__global__ void init_lattice(signed char* lattice, float* d_probs, const int nx, const int ny, const int num_lattices, const int seed);

__global__ void init_interactions(signed char* interactions, const int nx, const int ny, const int num_lattices, const int seed, const double, const char problogical_error_type = 'I');

__global__ void calc_energy_periodic_boundary(signed char *lattice, signed char *interactions, int *d_energy, int *d_offset_lattice, const int nx, const int ny, const int num_lattices, const int walker_per_interactions);

__global__ void calc_energy_open_boundary(signed char *lattice, signed char *interactions, int *d_energy, int *d_offset_lattice, const int nx, const int ny, const int num_lattices, const int walker_per_interactions);

__global__ void wang_landau_pre_run(
    signed char *d_lattice, signed char *d_interactions, int *d_energy, 
    unsigned long long *d_H, unsigned long long* d_iter, int *d_offset_lattice, 
    int *d_found_interval, signed char *d_store_lattice, const int E_min, const int E_max, 
    const int num_iterations, const int nx, const int ny, const int seed, const int len_interval, 
    const int found_interval, int num_walker, const int num_intervals, const int boundary_type, 
    const int walker_per_interactions);

__global__ void wang_landau(
    signed char *d_lattice, signed char *d_interactions, int *d_energy,
    int *d_start, int *d_end, unsigned long long *d_H, double *d_logG, int *d_offset_histogramm, int *d_offset_lattice, const int num_iterations,
    const int nx, const int ny, const int seed, double *factor, unsigned long long *d_offset_iter, int *d_expected_energy_spectrum, int *d_newEnergies, int *foundFlag, 
    const int num_lattices, const double beta, signed char *d_cond, int boundary_type
);

__global__ void check_histogram(unsigned long long *d_H, double *d_log_G, double *d_shared_logG, int *d_offset_histogramm, int *d_end, int *d_start, double *d_factor, int nx, int ny, double alpha, double beta, int *d_expected_energy_spectrum, int len_energy_spectrum, int num_walker_total, signed char *d_cond);

__global__ void find_spin_config_in_energy_range(signed char *d_lattice, signed char *d_interactions, const int nx, const int ny, const int num_lattices, const int seed, int *d_start, int *d_end, int *d_energy, int *d_offset_lattice);

__global__ void check_energy_ranges(int *d_energy, int *d_start, int *d_end);

__device__ void fisher_yates(int *d_shuffle, int seed, unsigned long long *d_offset_iter);

__global__ void redistribute_g_values(int num_intervals, long long len_histogram_over_all_walkers, int num_walker_per_interval,  double *d_log_G, double *d_shared_logG, int *d_end, int *d_start, double *d_factor, double beta, int *d_expected_energy_spectrum, signed char *d_cond);  

__global__ void calc_average_log_g(int num_intervals, long long len_histogram_over_all_walkers, int num_walker_per_interval,  double *d_log_G, double *d_shared_logG, int *d_end, int *d_start, int *d_expected_energy_spectrum, signed char *d_cond);

__device__ void store_lattice(signed char *d_lattice, int *d_energy, int* d_found_interval, signed char* d_store_lattice, const int E_min, const int nx, const int ny, const long long tid, const int len_interval, const int num_intervals, const int int_id);

__global__ void init_indices(int *d_indices);

__global__ void init_offsets_histogramm(int *d_offset_histogramm, int *d_start, int *d_end);

__global__ void init_offsets_lattice(int *d_offset_lattice, int nx, int ny, int num_lattices);

__global__ void replica_exchange(
    int *d_offset_lattice, int *d_energy, int *d_start, int *d_end, int *d_indices,
    double *d_logG, int *d_offset_histogram, bool even, int seed, unsigned long long *d_offset_iter);

__global__ void print_finished_walker_ratio(double *d_factor, int num_walker_total, const double exp_beta, double *d_finished_walkers_ratio);

__device__ RBIM periodic_boundary_random_bond_ising(
    signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_offset_lattice, unsigned long long *d_offset_iter, 
    curandStatePhilox4_32_10_t *st, const long long tid, const int nx, const int ny, const int interaction_offset
    );

__device__ RBIM open_boundary_random_bond_ising(signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_offset_lattice, unsigned long long *d_offset_iter, curandStatePhilox4_32_10_t *st, const long long tid, const int nx, const int ny, const int interaction_offset);

#endif // UTILS_H