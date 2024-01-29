#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>

__global__ void init_randombond(signed char* interactions, const float* __restrict__ interaction_randvals, const long long nx, const long long ny, const int num_lattices, const float p);

__global__ void init_spins(signed char* lattice, const float* __restrict__ randvals, const long long nx, const long long ny, const int num_lattices);

void init_interactions_with_seed(signed char* interactions, curandGenerator_t interaction_rng, float* interaction_randvals, const long long nx, const long long ny, const int num_lattices, const float p, const int blocks);

void initialize_spins(signed char* lattice_b, signed char* lattice_w, curandGenerator_t lattice_rng, float* lattice_randvals, const long long nx, const long long ny, const int num_lattices, bool up, const int blocks, std::string filename);

void write_lattice(signed char *lattice_b, signed char *lattice_w, std::string filename, long long nx, long long ny, const int num_lattices);

void write_bonds(signed char* interactions, std::string filename, long nx, long ny, const int num_lattices);

template<bool is_black>
__global__ void update_lattice(signed char* lattice, signed char* __restrict__ op_lattice, const float* __restrict__ randvals, signed char* interactions, const float *inv_temp, const long long nx, const long long ny, const int num_lattices, const float *coupling_constant);

void update(signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, signed char* interactions, float *inv_temp, long long nx, long long ny, const int num_lattices, float *coupling_constant, const int blocks);

__global__ void B2_lattices(signed char* lattice_b, signed char* lattice_w, const float *wave_vector, thrust::complex<float> *sum,  int nx, int ny, int num_lattices);

template<bool is_black>
__global__ void calc_energy(float* sum, signed char* lattice, signed char* __restrict__ op_lattice, signed char* interactions, const long long nx, const long long ny, const int num_lattices, const float *coupling_constant);

void calculate_B2(thrust::complex<float> *d_sum, signed char *lattice_b, signed char *lattice_w, thrust::complex<float> *d_store_sum, float *d_wave_vector, const long nx, const long ny, const int num_lattices, const int blocks);

void calculate_energy(float* d_energy, signed char *lattice_b, signed char *lattice_w, signed char *d_interactions, float *d_store_energy, float *coupling_constant, const int nx, const int ny, const int num_lattices, const int blocks);

__global__ void abs_square(thrust::complex<float> *d_store_sum, const int num_lattices);

__global__ void exp_beta(float *d_store_energy, float *inv_temp, const int num_lattices, const int L);

__global__ void weighted_energies(float *d_weighted_energies, float *d_store_energy, thrust::complex<float> *d_store_sum, float *d_partition_function, const int num_lattices, const int num_iterations);

void calculate_weighted_energies(float* d_weighted_energies, float *d_error_weight, float *d_store_energy, thrust::complex<float> *d_store_sum, float *d_partition_function, const int num_lattices, const int num_iterations_error, const int blocks, const int e);

int create_results_folder(char* results);

template<bool is_black>
__global__ void update_lattice_ob(signed char* lattice, signed char* __restrict__ op_lattice, const float* __restrict__ randvals, signed char* interactions, const float *inv_temp, const long long nx, const long long ny, const int num_lattices, const float *coupling_constant);

void update_ob(signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, signed char* interactions, float *inv_temp, long long nx, long long ny, const int num_lattices, float *coupling_constant, const int blocks, float *d_energy);

void combine_cross_subset_hamiltonians_to_whole_lattice_hamiltonian(float* d_energy,  float *d_store_energy, const int nx, const int ny, const int num_lattices);

void calculate_energy_ob(float* d_energy, signed char *lattice_b, signed char *lattice_w, signed char *d_interactions, float *d_store_energy, float *coupling_constant, const int loc, const int nx, const int ny, const int num_lattices, const int blocks);

template<bool is_black>
__global__ void calc_energy_ob(float* sum, signed char* lattice, signed char* __restrict__ op_lattice, signed char* interactions, const long long nx, const long long ny, const int num_lattices, const float *coupling_constant);

__global__ void init_spins_up(signed char* lattice, const long long nx, const long long ny, const int num_lattices);

__global__ void incremental_summation_of_product_of_magnetization_and_boltzmann_factor(float *d_store_energy, thrust::complex<float> *d_store_sum_0, thrust::complex<float> *d_store_sum_k, const int num_lattices, float *d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector, float *d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector);

__global__ void incremental_summation_of_partition_function(float *d_store_energy, const int num_lattices, float *d_store_partition_function);

#endif