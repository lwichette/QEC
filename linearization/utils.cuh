#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>

void write_bonds(signed char* interactions, std::string filename, long nx, long ny);

__global__ void init_randombond(signed char* interactions, const float* __restrict__ interaction_randvals, const long long nx, const long long ny, const float p);

void init_interactions_with_seed(signed char* interactions, const long long seed, const long long nx, const long long ny, const float p);

__global__ void init_spins(signed char* lattice, const float* __restrict__ randvals, const long long nx, const long long ny);

void init_spins_with_seed(signed char* lattice_b, signed char* lattice_w, const long long seed, const long long nx, const long long ny);

__global__ void B2_lattices(signed char* lattice_b, signed char* lattice_w, const float *wave_vector, thrust::complex<float> *sum,  int nx, int ny);

template<bool is_black> 
__global__ void calc_energy(float* sum, signed char* lattice, signed char* __restrict__ op_lattice, signed char* interactions, const long long nx, const long long ny, const float coupling_constant);

__global__ void abs_square(thrust::complex<float> *d_store_sum, const int num_iterations);

__global__ void exp_beta(float *d_store_energy, float inv_temp, const int num_iterations, const int L);

void write_lattice(signed char *lattice_b, signed char *lattice_w, std::string filename, long long nx, long long ny);

void calculate_B2(signed char *lattice_b, signed char *lattice_w, thrust::complex<float> *d_store_sum, float *d_wave_vector, int i, const long nx, const long ny);

void calculate_energy(signed char *lattice_b, signed char *lattice_w, signed char *d_interactions, float *d_store_energy, float coupling_constant, int i, long nx, long ny);

__global__ void weighted_energies(float *d_weighted_energies, float *d_store_energy, thrust::complex<float> *d_store_sum, float *d_partition_function, const int num_iterations);

void calculate_weighted_energies(float *d_error_weight, float *d_store_energy, thrust::complex<float> *d_store_sum, float *d_partition_function, const int num_iterations, const int blocks, const int j);

template<bool is_black>
__global__ void update_lattice(signed char* lattice, signed char* __restrict__ op_lattice, const float* __restrict__ randvals, signed char* interactions, const float inv_temp, const long long nx, const long long ny, const float coupling_constant);

void update(signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, signed char* interactions, float inv_temp, long long nx, long long ny, float coupling_constant);

float calc_psi(float *d_error_weight_0, float *d_error_weight_k, const int num_iterations_error, const int nx);

int create_results_folder(char* results);
                            
#endif
