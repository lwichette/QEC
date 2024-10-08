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
    std::vector<int> E_min;
    std::vector<int> E_max;
    int num_intervals;
    int walker_per_interval;
    float overlap_decimal;
    int seed_histogram;
    int seed_run;
    char logical_error_type;
    int boundary_type;
    int num_interactions;
    int replica_exchange_offset;
    int task_id;
} Options;

typedef struct
{
    int new_energy;
    int i;
    int j;
} RBIM;

typedef struct
{
    double new_energy;
    int i;
    int j;
    bool color;
} RBIM_eight_vertex;

void parse_args(int argc, char *argv[], Options *options);

IntervalResult generate_intervals(const int E_min, const int E_max, int num_intervals, int num_walker, float overlap_decimal);

template <typename T>
inline void writeToFile(const std::string &filename, const T *data, int nx_w, int ny)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        for (int i = 0; i < nx_w; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                file << static_cast<int>(data[i * ny + j]) << " ";
            }
            file << std::endl;
        }
    }
    else
    {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
    file.close();

    return;
}

template <>
inline void writeToFile<double>(const std::string &filename, const double *data, int nx_w, int ny)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << std::fixed << std::setprecision(10);
        for (int i = 0; i < nx_w; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                file << data[i * ny + j] << " ";
            }
            file << std::endl;
        }
    }
    else
    {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
    file.close();

    return;
}

void write(signed char *array_host, const std::string &filename, long nx, long ny, int num_lattices, bool lattice, const int *energies = NULL); // non eight vertex write with int energies

template <typename T>
inline void write(
    T *array_host, const std::string &filename, long num_rows, long num_cols,
    int num_lattices, bool is_lattice, const double *energies = NULL)
{
    if (num_lattices == 1)
    {
        writeToFile(filename + ".txt", array_host, num_rows, num_cols);
    }
    else
    {
        for (int l = 0; l < num_lattices; l++)
        {
            int offset = l * num_rows * num_cols;

            if (energies)
            {
                if (energies[l] == 0 && array_host[offset] == 0)
                {
                    continue;
                }
            }

            std::string file_suffix = (!energies) ? std::to_string(l) : std::to_string(energies[l]);
            writeToFile(filename + "_energy_" + file_suffix + ".txt", array_host + offset, num_rows, num_cols);
        }
    }

    return;
}

__inline__ __device__ void fisher_yates(int *d_shuffle, int seed, unsigned long long *d_offset_iter)
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

    return;
}

template <typename EnergyType>
__global__ void replica_exchange(
    int *d_offset_lattice, EnergyType *d_energy, int *d_start, int *d_end, int *d_indices,
    double *d_logG, int *d_offset_histogram, bool even, int seed,
    unsigned long long *d_offset_iter, const int num_intervals,
    const int walker_per_interactions, int *d_cond_interaction)
{

    // if last block in interaction return
    if (blockIdx.x % num_intervals == (num_intervals - 1))
        return;

    // if even only even blocks if odd only odd blocks
    if ((even && (blockIdx.x % 2 != 0)) || (!even && (blockIdx.x % 2 == 0)))
        return;

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    const int int_id = tid / walker_per_interactions;

    if (d_cond_interaction[int_id] == -1)
        return;

    long long cid = static_cast<long long>(blockDim.x) * (blockIdx.x + 1);

    if (threadIdx.x == 0)
    {
        fisher_yates(d_indices, seed, d_offset_iter);
    }

    __syncthreads();

    cid += d_indices[tid];

    if (d_energy[tid] > d_end[blockIdx.x + 1] || d_energy[tid] < d_start[blockIdx.x + 1])
        return;
    if (d_energy[cid] > d_end[blockIdx.x] || d_energy[cid] < d_start[blockIdx.x])
        return;

    double prob = min(1.0, exp(d_logG[d_offset_histogram[tid] + static_cast<int>(d_energy[tid]) - d_start[blockIdx.x]] - d_logG[d_offset_histogram[tid] + static_cast<int>(d_energy[cid]) - d_start[blockIdx.x]]) * exp(d_logG[d_offset_histogram[cid] + static_cast<int>(d_energy[cid]) - d_start[blockIdx.x + 1]] - d_logG[d_offset_histogram[cid] + static_cast<int>(d_energy[tid]) - d_start[blockIdx.x + 1]]));

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_offset_iter[tid], &st);

    if (curand_uniform(&st) < prob)
    {

        int temp_off = d_offset_lattice[tid];
        EnergyType temp_energy = d_energy[tid];

        d_offset_lattice[tid] = d_offset_lattice[cid];
        d_energy[tid] = d_energy[cid];

        d_offset_lattice[cid] = temp_off;
        d_energy[cid] = temp_energy;

        // if (tid == 0 || cid == 0)
        // {
        //     printf("replica exchange: cid %d tid %lld e_cid %.2f e_tid %.2f\n", cid, tid, d_energy[tid], temp_energy);
        // }
    }

    d_offset_iter[tid] += 1;

    return;
}

template <typename T>
__global__ void check_energy_ranges(T *d_energy, int *d_start, int *d_end, int total_walker)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= total_walker)
        return;

    int check = 1;

    int energy_int;
    if (std::is_same<T, double>::value)
    {
        // Round the double valued energy to the nearest int as binning allows for out of bounds by rounding to nearest bin
        energy_int = __double2int_rn(static_cast<double>(d_energy[tid]));
    }
    else
    {
        energy_int = static_cast<int>(d_energy[tid]);
    }

    if (energy_int > d_end[blockIdx.x] || energy_int < d_start[blockIdx.x])
    {
        check = 0;
        printf("tid=%lld energy=%.2f start_interval=%d end_interval=%d  \n", tid, static_cast<double>(d_energy[tid]), d_start[blockIdx.x], d_end[blockIdx.x]);
    }

    assert(check);

    return;
}

void read(std::vector<signed char> &lattice, std::string filename);

void read(std::vector<double> &lattice, std::string filename);

void create_directory(std::string path);

void write_histograms(unsigned long long *d_H, std::string path_histograms, int len_histogram, int seed, int E_min);

std::vector<signed char> read_histogram(std::string filename, std::vector<int> &E_min, std::vector<int> &E_max);

double logSumExp(const std::vector<std::map<int, double>> &data);

void rescaleMapValues(std::vector<std::map<int, double>> &data, double X, double Y);

void read(std::vector<signed char> &lattice, std::string filename);

void handleNewEnergyError(int *new_energies, int *new_energies_flag, char *histogram_file, int num_walkers_total);

void calc_energy(int blocks, int threads, const int boundary_type, signed char *lattice, signed char *interactions, int *d_energy, int *d_offset_lattice, const int nx, const int ny, const int total_walker, const int walker_per_interactions);

void result_handling(Options options, std::vector<double> h_logG, std::vector<int> h_start, std::vector<int> h_end, int int_id);

void result_handling_stitched_histogram(
    Options options, std::vector<double> h_logG,
    std::vector<int> h_start, std::vector<int> h_end, int int_id);

void check_interactions_finished(
    signed char *d_cond, int *d_cond_interactions,
    int *d_offset_intervals, int num_intervals, int num_interactions);

void cut_overlapping_histogram_parts(
    std::vector<std::map<int, double>> &interval_data,
    const std::vector<int> &stitching_keys);

std::tuple<int, double> find_stitching_keys(const std::map<int, double> &current_interval, const std::map<int, double> &next_interval);

std::string constructFilePath(float prob_interactions, int X, int Y, int seed, std::string type, char error_class, int boundary_type, int task_id);

std::vector<signed char> get_lattice_with_pre_run_result(float prob, int seed, int x, int y, std::vector<int> h_start, std::vector<int> h_end, int num_intervals, int num_walkers_total, int num_walkers_per_interval, char error_class, int boundary_type, int task_id);

__global__ void init_lattice(signed char *lattice, float *d_probs, const int nx, const int ny, const int num_lattices, const int seed);

__global__ void init_interactions(signed char *interactions, const int nx, const int ny, const int num_lattices, const int seed, const double, const char problogical_error_type = 'I');

__global__ void calc_energy_periodic_boundary(signed char *lattice, signed char *interactions, int *d_energy, int *d_offset_lattice, const int nx, const int ny, const int num_lattices, const int walker_per_interactions);

__global__ void calc_energy_open_boundary(signed char *lattice, signed char *interactions, int *d_energy, int *d_offset_lattice, const int nx, const int ny, const int num_lattices, const int walker_per_interactions);

__global__ void calc_energy_cylinder(signed char *lattice, signed char *interactions, int *d_energy, int *d_offset_lattice, const int nx, const int ny, const int num_lattices, const int walker_per_interactions);

__global__ void wang_landau_pre_run(
    signed char *d_lattice, signed char *d_interactions, int *d_energy,
    unsigned long long *d_H, unsigned long long *d_iter, int *d_offset_lattice,
    int *d_found_interval, signed char *d_store_lattice, const int E_min, const int E_max,
    const int num_iterations, const int nx, const int ny, const int seed, const int len_interval,
    const int found_interval, int num_walker, const int num_intervals, const int boundary_type,
    const int walker_per_interactions);

__global__ void wang_landau(
    signed char *d_lattice, signed char *d_interactions, int *d_energy,
    int *d_start, int *d_end, unsigned long long *d_H, double *d_logG,
    int *d_offset_histogramm, int *d_offset_lattice, const int num_iterations,
    const int nx, const int ny, const int seed, double *factor,
    unsigned long long *d_offset_iter, signed char *d_expected_energy_spectrum,
    int *d_newEnergies, int *foundFlag, const int num_lattices,
    const double beta, signed char *d_cond, int boundary_type,
    const int walker_per_interactions, const int num_intervals,
    int *d_offset_energy_spectrum, int *d_cond_interactions);

__global__ void check_histogram(
    unsigned long long *d_H, double *d_log_G, double *d_shared_logG, int *d_offset_histogramm,
    int *d_end, int *d_start, double *d_factor, int nx, int ny, double alpha, double beta,
    signed char *d_expected_energy_spectrum, int *d_len_energy_spectrum, int num_walker_total, signed char *d_cond,
    const int walker_per_interactions, const int num_intervals, int *d_offset_energy_spectrum,
    int *d_cond_interactions);

__global__ void find_spin_config_in_energy_range(signed char *d_lattice, signed char *d_interactions, const int nx, const int ny, const int num_lattices, const int seed, int *d_start, int *d_end, int *d_energy, int *d_offset_lattice);

__global__ void redistribute_g_values(
    int num_intervals_per_interaction, int *d_len_histograms, int num_walker_per_interval,
    double *d_log_G, double *d_shared_logG, int *d_end, int *d_start, double *d_factor,
    double beta, signed char *d_expected_energy_spectrum, signed char *d_cond,
    int *d_offset_histogram, int num_interactions, long long *d_offset_shared_logG,
    int *d_cond_interactions, int total_len_histogram);

__global__ void calc_average_log_g(
    int num_intervals_per_interaction, int *d_len_histograms,
    int num_walker_per_interval, double *d_log_G,
    double *d_shared_logG, int *d_end, int *d_start,
    signed char *d_expected_energy_spectrum, signed char *d_cond,
    int *d_offset_histogram, int *d_offset_energy_spectrum,
    int num_interactions, long long *d_offset_shared_logG,
    int *d_cond_interactions, int total_len_histogram);

__global__ void initialize_Gaussian_error_rates(double *d_prob_i, double *d_prob_x, double *d_prob_y, double *d_prob_z, int num_qubits, int num_interactions, double error_rate_mean, double error_rate_variance, unsigned long long seed);

__global__ void initialize_coupling_factors(double *prob_i_err, double *prob_x_err, double *prob_y_err, double *prob_z_err, int num_qubits, int num_interactions, int histogram_scale, double *d_J_i, double *d_J_x, double *d_J_y, double *d_J_z);

// Overload for int type (no color argument)
__device__ void store_lattice(
    signed char *d_lattice, int *d_energy, int *d_found_interval, signed char *d_store_lattice,
    const int E_min, const int nx, const int ny, const long long tid, const int len_interval,
    const int num_interval, const int int_id);

// Overload for double type (with color argument)
__device__ void store_lattice(
    signed char *d_lattice_r, signed char *d_lattice_b, double *d_energy, int *d_found_interval, signed char *d_store_lattice_r, signed char *d_store_lattice_b,
    const int E_min, const int nx, const int ny, const long long tid, const int len_interval,
    const int num_interval, const int int_id);

__global__ void init_indices(int *d_indices, int total_walker);

__global__ void init_offsets_histogramm(int *d_offset_histogramm, int *d_start, int *d_end, int *d_len_histograms, int num_intervals, int total_walker);

__global__ void init_offsets_lattice(int *d_offset_lattice, int nx, int ny, int num_lattices);

__global__ void print_finished_walker_ratio(double *d_factor, int num_walker_total, const double exp_beta, double *d_finished_walkers_ratio);

__global__ void generate_pauli_errors(int *pauli_errors, const int num_qubits, const int X, const int num_interactions, const unsigned long seed, const double p_I, const double p_X, const double p_Y, const double p_Z, const bool x_horizontal_error, const bool x_vertical_error, const bool z_horizontal_error, const bool z_vertical_error);

__global__ void generate_pauli_errors(int *pauli_errors, const int num_qubits, const int X, const int num_interactions, const unsigned long seed, const double *p_I, const double *p_X, const double *p_Y, const double *p_Z, const bool x_horizontal_error, const bool x_vertical_error, const bool z_horizontal_error, const bool z_vertical_error);

__global__ void get_interaction_from_commutator(int *pauli_errors, double *int_X, double *int_Y, double *int_Z, const int num_qubits, const int num_interactions, double J_X, double J_Y, double J_Z);

__global__ void get_interaction_from_commutator(int *pauli_errors, double *int_X, double *int_Y, double *int_Z, const int num_qubits, const int num_interactions, double *J_X, double *J_Y, double *J_Z);

__global__ void initialize_Gaussian_error_rates(double *prob_i_err, double *prob_x_err, double *prob_y_err, double *prob_z_err, int num_qubits, int num_interactions, int histogram_scale, double *d_J_i, double *d_J_x, double *d_J_y, double *d_J_z);

__global__ void init_interactions_eight_vertex(double *int_X, double *int_Y, double *int_Z, const int num_qubits, const int num_interactions, int X, int Y, double *int_r, double *int_b, double *d_interactions_down_four_body, double *d_interactions_right_four_body);

__global__ void calc_energy_eight_vertex(double *energy_out, signed char *lattice_b, signed char *lattice_r, double *interactions_b, double *interactions_r, double *interactions_four_body_right, double *interactions_four_body_down, const int num_qubits, const int X, const int Y, const int num_lattices, const int num_lattices_x_interaction, int *d_offset_lattice_per_walker);

__global__ void wang_landau_pre_run_eight_vertex(
    signed char *d_lattice_b, signed char *d_lattice_r, double *d_interactions_b, double *d_interactions_r, double *d_interactions_right_four_body, double *d_interactions_down_four_body, double *d_energy, unsigned long long *d_H, unsigned long long *d_iter,
    int *d_found_interval, signed char *d_store_lattice_b, signed char *d_store_lattice_r, const int E_min, const int E_max,
    const int num_iterations, const int num_qubits, const int X, const int Y, const int seed, const int len_interval, const int found_interval,
    const int num_walker, const int num_interval, const int walker_per_interaction, int *d_offset_lattice_per_walker);

__global__ void wang_landau_eight_vertex(
    signed char *d_lattice_b, signed char *d_lattice_r, double *d_interactions_b, double *d_interactions_r, double *d_interactions_right_four_body, double *d_interactions_down_four_body, double *d_energy, int *d_start, int *d_end, unsigned long long *d_H,
    double *d_logG, int *d_offset_histogramm, int *d_offset_lattice, const int num_iterations, const int nx, const int ny,
    const int seed, double *factor, unsigned long long *d_offset_iter, signed char *d_expected_energy_spectrum, double *d_newEnergies, int *foundFlag,
    const int num_lattices, const double beta, signed char *d_cond, const int walker_per_interactions, const int num_intervals,
    int *d_offset_energy_spectrum, int *d_cond_interaction, const int walker_per_interval);

__global__ void check_sums(int *d_cond_interactions, int num_intervals, int num_interactions);

__device__ RBIM periodic_boundary_random_bond_ising(signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_offset_lattice, unsigned long long *d_offset_iter, curandStatePhilox4_32_10_t *st, const long long tid, const int nx, const int ny, const int interaction_offset);

__device__ RBIM open_boundary_random_bond_ising(signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_offset_lattice, unsigned long long *d_offset_iter, curandStatePhilox4_32_10_t *st, const long long tid, const int nx, const int ny, const int interaction_offset);

__device__ double calc_energy_periodic_eight_vertex(signed char *lattice_b, signed char *lattice_r, double *interactions_b, double *interactions_r, double *interactions_four_body_right, double *interactions_four_body_down, const int num_qubits, const int X, const int Y, const int num_lattices_x_interaction, int *d_offset_lattice_per_walker);

__device__ int scalar_commutator(int pauli1, int pauli2);

__device__ double calc_energy_periodic_eight_vertex(signed char *lattice_b, signed char *lattice_r, double *interactions_b, double *interactions_r, double *interactions_four_body_right, double *interactions_four_body_down, const int num_qubits, const int X, const int Y, const int num_lattices_x_interaction);

__device__ RBIM cylinder_random_bond_ising(signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_offset_lattice, unsigned long long *d_offset_iter, curandStatePhilox4_32_10_t *st, const long long tid, const int nx, const int ny, const int interaction_offset);

__device__ RBIM_eight_vertex eight_vertex_periodic_wl_step(
    signed char *d_lattice_b, signed char *d_lattice_r, double *d_interactions_b, double *d_interactions_r, double *d_interactions_four_body_right, double *d_interactions_four_body_down, double *d_energy, unsigned long long *d_offset_iter,
    curandStatePhilox4_32_10_t *st, const long long tid, const int num_qubits, const int X, const int Y, const int num_lattices, const int num_lattices_x_interaction, int *d_offset_lattice_per_walker);

__device__ int commutator(int pauli1, int pauli2);

std::string eight_vertex_histogram_path(
    bool is_qubit_specific_noise, float error_mean, float error_variance,
    int X, int Y, int seed_hist, bool x_horizontal_error, bool x_vertical_error,
    bool z_horizontal_error, bool z_vertical_error, float prob_x_err, float prob_y_err, float prob_z_err, int task_id);

std::string eight_vertex_interaction_path(
    bool is_qubit_specific_noise, float error_mean, float error_variance,
    int X, int Y, int seed_hist, bool x_horizontal_error, bool x_vertical_error,
    bool z_horizontal_error, bool z_vertical_error, std::string interaction_type, float prob_x_err, float prob_y_err, float prob_z_err, int task_id);

std::map<std::string, std::vector<signed char>> get_lattice_with_pre_run_result_eight_vertex(
    bool is_qubit_specific_noise, float error_mean, float error_variance, bool x_horizontal_error, bool x_vertical_error, bool z_horizontal_error, bool z_vertical_error,
    int X, int Y, std::vector<int> h_start, std::vector<int> h_end, int num_intervals, int num_walkers_per_interval, int seed_hist, float prob_x_err, float prob_y_err, float prob_z_err, int task_id);

void eight_vertex_result_handling_stitched_histogram(
    Options options, std::vector<double> h_logG, float error_mean, float error_variance,
    float prob_x, float prob_y, float prob_z, std::vector<int> h_start, std::vector<int> h_end, int int_id,
    bool isQubitSpecificNoise, bool x_horizontal_error, bool x_vertical_error, bool z_horizontal_error,
    bool z_vertical_error);

__global__ void reset_d_cond(signed char *d_cond, double *d_factor, int total_intervals, double beta, int walker_per_interval);

#endif // WLUTILS_H
