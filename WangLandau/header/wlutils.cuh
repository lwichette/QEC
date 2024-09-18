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
    std::vector<int> h_start, std::vector<int> h_end, int int_id, int X, int Y);

void rescale_intervals_for_concatenation(std::vector<std::map<int, double>> &interval_data, const std::vector<int> &stitching_keys);

void check_interactions_finished(
    signed char *d_cond, int *d_cond_interactions,
    int *d_offset_intervals, int num_intervals, int num_interactions,
    void *d_temp_storage, size_t &temp_storage_bytes);

void cut_overlapping_histogram_parts(
    std::vector<std::map<int, double>> &interval_data,
    const std::vector<int> &stitching_keys);

int find_stitching_keys(const std::map<int, double> &current_interval, const std::map<int, double> &next_interval);

std::string constructFilePath(float prob_interactions, int X, int Y, int seed, std::string type, char error_class, int boundary_type);

std::vector<signed char> get_lattice_with_pre_run_result(float prob, int seed, int x, int y, std::vector<int> h_start, std::vector<int> h_end, int num_intervals, int num_walkers_total, int num_walkers_per_interval, char error_class, int boundary_type);

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

__global__ void check_energy_ranges(int *d_energy, int *d_start, int *d_end, int total_walker);

__device__ void fisher_yates(int *d_shuffle, int seed, unsigned long long *d_offset_iter);

__global__ void redistribute_g_values(
    int num_intervals_per_interaction, int *d_len_histograms, int num_walker_per_interval,
    double *d_log_G, double *d_shared_logG, int *d_end, int *d_start, double *d_factor,
    double beta, signed char *d_expected_energy_spectrum, signed char *d_cond,
    int *d_offset_histogram, int num_interactions, long long *d_offset_shared_logG,
    int *d_cond_interactions);

__global__ void calc_average_log_g(
    int num_intervals_per_interaction, int *d_len_histograms,
    int num_walker_per_interval, double *d_log_G,
    double *d_shared_logG, int *d_end, int *d_start,
    signed char *d_expected_energy_spectrum, signed char *d_cond,
    int *d_offset_histogram, int *d_offset_energy_spectrum,
    int num_interactions, long long *d_offset_shared_logG,
    int *d_cond_interactions);

__global__ void initialize_Gaussian_error_rates(double *d_prob_i, double *d_prob_x, double *d_prob_y, double *d_prob_z, int num_qubits, int num_interactions, double error_rate_mean, double error_rate_variance, unsigned long long seed);

__global__ void initialize_coupling_factors(double *prob_i_err, double *prob_x_err, double *prob_y_err, double *prob_z_err, int num_qubits, int num_interactions, int histogram_scale, double *d_J_i, double *d_J_x, double *d_J_y, double *d_J_z);

// Overload for int type (no color argument)
__device__ void store_lattice(
    signed char *d_lattice, int *d_energy, int *d_found_interval, signed char *d_store_lattice,
    const int E_min, const int nx, const int ny, const long long tid, const int len_interval,
    const int num_interval, const int int_id);

// Overload for double type (with color argument)
__device__ void store_lattice(
    signed char *d_lattice, double *d_energy, int *d_found_interval, signed char *d_store_lattice,
    const int E_min, const int nx, const int ny, const long long tid, const int len_interval,
    const int num_interval, const int int_id, bool color);

__global__ void init_indices(int *d_indices, int total_walker);

__global__ void init_offsets_histogramm(int *d_offset_histogramm, int *d_start, int *d_end, int *d_len_histograms, int num_intervals, int total_walker);

__global__ void init_offsets_lattice(int *d_offset_lattice, int nx, int ny, int num_lattices);

__global__ void replica_exchange(
    int *d_offset_lattice, int *d_energy, int *d_start, int *d_end, int *d_indices,
    double *d_logG, int *d_offset_histogram, bool even, int seed,
    unsigned long long *d_offset_iter, const int num_intervals,
    const int walker_per_interactions, int *d_cond_interactions);

__global__ void print_finished_walker_ratio(double *d_factor, int num_walker_total, const double exp_beta, double *d_finished_walkers_ratio);

__global__ void generate_pauli_errors(int *pauli_errors, const int num_qubits, const int X, const int num_interactions, const unsigned long seed, const double p_I, const double p_X, const double p_Y, const double p_Z, const bool x_horizontal_error, const bool x_vertical_error, const bool z_horizontal_error, const bool z_vertical_error);

__global__ void generate_pauli_errors(int *pauli_errors, const int num_qubits, const int X, const int num_interactions, const unsigned long seed, const double *p_I, const double *p_X, const double *p_Y, const double *p_Z, const bool x_horizontal_error, const bool x_vertical_error, const bool z_horizontal_error, const bool z_vertical_error);

__global__ void get_interaction_from_commutator(int *pauli_errors, double *int_X, double *int_Y, double *int_Z, const int num_qubits, const int num_interactions, double J_X, double J_Y, double J_Z);

__global__ void get_interaction_from_commutator(int *pauli_errors, double *int_X, double *int_Y, double *int_Z, const int num_qubits, const int num_interactions, double *J_X, double *J_Y, double *J_Z);

__global__ void initialize_Gaussian_error_rates(double *prob_i_err, double *prob_x_err, double *prob_y_err, double *prob_z_err, int num_qubits, int num_interactions, int histogram_scale, double *d_J_i, double *d_J_x, double *d_J_y, double *d_J_z);

__global__ void init_interactions_eight_vertex(double *int_X, double *int_Y, double *int_Z, const int num_qubits, const int num_interactions, int X, int Y, double *int_r, double *int_b, double *d_interactions_down_four_body, double *d_interactions_right_four_body);

__global__ void calc_energy_eight_vertex(double *energy_out, signed char *lattice_b, signed char *lattice_r, double *interactions_b, double *interactions_r, double *interactions_four_body_right, double *interactions_four_body_down, const int num_qubits, const int X, const int Y, const int num_lattices, const int num_lattices_x_interaction);

__global__ void wang_landau_pre_run_eight_vertex(
    signed char *d_lattice_b, signed char *d_lattice_r, double *d_interactions_b, double *d_interactions_r, double *d_interactions_right_four_body, double *d_interactions_down_four_body, double *d_energy, unsigned long long *d_H, unsigned long long *d_iter,
    int *d_found_interval, signed char *d_store_lattice_b, signed char *d_store_lattice_r, const int E_min, const int E_max,
    const int num_iterations, const int num_qubits, const int X, const int Y, const int seed, const int len_interval, const int found_interval,
    const int num_walker, const int num_interval, const int walker_per_interaction);

__global__ void check_sums(int *d_cond_interactions, int num_intervals, int num_interactions);

__global__ void test_eight_vertex_periodic_wl_step(
    signed char *d_lattice_b, signed char *d_lattice_r, double *d_interactions_b, double *d_interactions_r, double *d_interactions_four_body_right, double *d_interactions_four_body_down, double *d_energy, unsigned long long *d_offset_iter, const int num_qubits, const int X, const int Y, const int num_lattices, const int num_lattices_x_interaction);

__device__ RBIM periodic_boundary_random_bond_ising(signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_offset_lattice, unsigned long long *d_offset_iter, curandStatePhilox4_32_10_t *st, const long long tid, const int nx, const int ny, const int interaction_offset);

__device__ RBIM open_boundary_random_bond_ising(signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_offset_lattice, unsigned long long *d_offset_iter, curandStatePhilox4_32_10_t *st, const long long tid, const int nx, const int ny, const int interaction_offset);

__device__ double calc_energy_periodic_eight_vertex(signed char *lattice_b, signed char *lattice_r, double *interactions_b, double *interactions_r, double *interactions_four_body_right, double *interactions_four_body_down, const int num_qubits, const int X, const int Y, const int num_lattices_x_interaction);

__device__ int scalar_commutator(int pauli1, int pauli2);

__device__ double calc_energy_periodic_eight_vertex(signed char *lattice_b, signed char *lattice_r, double *interactions_b, double *interactions_r, double *interactions_four_body_right, double *interactions_four_body_down, const int num_qubits, const int X, const int Y, const int num_lattices_x_interaction);

__device__ RBIM cylinder_random_bond_ising(signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_offset_lattice, unsigned long long *d_offset_iter, curandStatePhilox4_32_10_t *st, const long long tid, const int nx, const int ny, const int interaction_offset);

__device__ RBIM_eight_vertex eight_vertex_periodic_wl_step(
    signed char *d_lattice_b, signed char *d_lattice_r, double *d_interactions_b, double *d_interactions_r, double *d_interactions_four_body_right, double *d_interactions_four_body_down, double *d_energy, unsigned long long *d_offset_iter,
    curandStatePhilox4_32_10_t *st, const long long tid, const int num_qubits, const int X, const int Y, const int num_lattices, const int num_lattices_x_interaction);

__device__ int commutator(int pauli1, int pauli2);

std::string eight_vertex_histogram_path(
    bool is_qubit_specific_noise, float error_mean, float error_variance,
    int X, int Y, int seed_hist, bool x_horizontal_error, bool x_vertical_error,
    bool z_horizontal_error, bool z_vertical_error, float prob_x_err, float prob_y_err, float prob_z_err);

std::string eight_vertex_interaction_path(
    bool is_qubit_specific_noise, float error_mean, float error_variance,
    int X, int Y, int seed_hist, bool x_horizontal_error, bool x_vertical_error,
    bool z_horizontal_error, bool z_vertical_error, std::string interaction_type, float prob_x_err, float prob_y_err, float prob_z_err);

#endif // WLUTILS_H
