#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>

__global__ void init_spins_up(signed char* lattice, const long long nx, const long long ny, const int num_lattices);

__global__ void init_randombond(signed char* interactions, const float* __restrict__ interaction_randvals, const long long nx, const long long ny, const int num_lattices, const float p);

__global__ void init_spins(signed char* lattice, const float* __restrict__ randvals, const long long nx, const long long ny, const int num_lattices);

void init_interactions_with_seed(signed char* interactions, curandGenerator_t interaction_rng, float* interaction_randvals, const long long nx, const long long ny, const int num_lattices, const float p, const int blocks);

void initialize_spins(signed char* lattice_b, signed char* lattice_w, curandGenerator_t lattice_rng, float* lattice_randvals, const long long nx, const long long ny, const int num_lattices, bool up, const int blocks, bool read_lattice, std::string filename_b, std::string filename_w);

void write_updated_lattices(signed char *lattice_b, signed char *lattice_w, const long long nx, const long long ny, const int num_lattices, std::string lattice_b_file_name, std::string lattice_w_file_name);

void write_lattice_to_disc(signed char *lattice_b, signed char *lattice_w, std::string filename, const long long nx, const long long ny, const int num_lattices);

void write_bonds(signed char* interactions, std::string filename, const long nx, const long ny, const int num_lattices);

template<bool is_black>
__global__ void update_lattice(signed char* lattice, signed char* __restrict__ op_lattice, const float* __restrict__ randvals, const signed char* interactions, const double *inv_temp, const long long nx, const long long ny, const int num_lattices, double *d_energy);

void update(signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, const signed char* interactions, const double *inv_temp, const long long nx, const long long ny, const int num_lattices, const int blocks, double *d_energy);

__global__ void B2_lattices(signed char* lattice_b, signed char* lattice_w, const double *wave_vector, thrust::complex<double> *sum, const int nx, const int ny, const int num_lattices);

void calculate_B2(thrust::complex<double> *d_sum, signed char *lattice_b, signed char *lattice_w, thrust::complex<double> *d_store_sum, const double *d_wave_vector, const long nx, const long ny, const int num_lattices, const int blocks);

__global__ void abs_square(thrust::complex<double> *d_store_sum, const int num_lattices);

int create_results_folder(char* results);

template<bool is_black>
__global__ void update_lattice_ob(signed char* lattice, signed char* __restrict__ op_lattice, const float* __restrict__ randvals, const signed char* interactions, const double *inv_temp, const long long nx, const long long ny, const int num_lattices, double *d_energy);

void update_ob(signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, const signed char* interactions, const double *inv_temp, const long long nx, const long long ny, const int num_lattices, const int blocks, double *d_energy);

__global__ void incrementalSumMagnetization(thrust::complex<double> *d_store_sum_0, thrust::complex<double> *d_store_sum_k, const int num_lattices, double *d_storeIncrementalSumMag_0, double *d_storeIncrementalSumMag_k);

#endif