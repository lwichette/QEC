#include <cstdlib>
#include <fstream>
#include <iostream>
#include <time.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>
#include <cmath>
#include <math.h>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <filesystem>
#include "defines.h"
#include "utils.cuh"
#include "cudamacro.h"

using namespace std;

void *d_temp_nx = NULL;
size_t temp_storage_nx = 0;

void *d_temp_nx_thrust = NULL;
size_t temp_storage_nx_thrust = 0;

void *d_temp_nis = NULL;
size_t temp_storage_nis = 0;

void *d_temp_nie = NULL;
size_t temp_storage_nie = 0;

// Initialize lattice spins
__global__ void init_spins_up(
    signed char* lattice, const long long nx, const long long ny, const int num_lattices
){
    const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (tid >= nx * ny * num_lattices) return;

    lattice[tid] = 1;
}

__global__ void init_randombond(
    signed char* interactions, const float* __restrict__ interaction_randvals,
    const long long nx, const long long ny, const int num_lattices, const float p
){

        const long long tid = static_cast<long long>(threadIdx.x + blockIdx.x * blockDim.x);

        if (tid >= 2*nx*ny*num_lattices) return;

        float bondrandval = interaction_randvals[tid];
        signed char bondval = (bondrandval<p)? -1 : 1;
        interactions[tid] = bondval;
}

// Initialize lattice spins
__global__ void init_spins(
    signed char* lattice, const float* __restrict__ randvals,
    const long long nx, const long long ny, const int num_lattices
){
        const long long  tid = static_cast<long long>(blockDim.x * blockIdx.x + threadIdx.x);
        if (tid >= nx * ny * num_lattices) return;

        float randval = randvals[tid];
        signed char val = (randval < 0.5f) ? -1 : 1;
        lattice[tid] = val;
}

void init_interactions_with_seed(
    signed char* interactions, curandGenerator_t interaction_rng, float* interaction_randvals,
    const long long nx, const long long ny, const int num_lattices, const float p, const int blocks
){
    CHECK_CURAND(curandGenerateUniform(interaction_rng,interaction_randvals, num_lattices*nx*ny*2));
    init_randombond<<<blocks, THREADS>>>(interactions, interaction_randvals, nx, ny, num_lattices, p);
}

void initialize_spins(
    signed char* lattice_b, signed char* lattice_w, curandGenerator_t lattice_rng, float* lattice_randvals,
    const long long nx, const long long ny, const int num_lattices, bool up, const int blocks, bool read_lattice, std::string filename_b, std::string filename_w
){

    if (read_lattice){
        // Initialization of black lattice
        if (std::filesystem::exists(filename_b.c_str())){

            std::vector<signed char> charVector_b;

            // black lattice read
            std::ifstream inFile_b(filename_b);

            int value;

            if (!inFile_b.is_open()) {
                std::cerr << "Error opening file for reading." << std::endl;
                return;  // Return an empty vector in case of an error
            }

            // Read each value from the file
            while (inFile_b >> value) {
                // Reverse the mapping: 0 to -1 and 1 to 1
                charVector_b.push_back((value == 0) ? -1 : 1);
            }
            // Close the file
            inFile_b.close();

            cout << "Initialized black lattice with preceeding results. ";
            // // printing initialized elements
            // for (const auto& element : charVector_b) {
            //     cout << static_cast<int>(element) << " ";
            // }
            cout << endl;

            CHECK_CUDA(cudaMemcpy(lattice_b, charVector_b.data(), num_lattices * nx * ny /2 * sizeof(*lattice_b), cudaMemcpyHostToDevice));
        }
        else{
            std::cerr << "Error opening file for reading. Could not find black lattice file." << std::endl;
            return;  // Return an empty vector in case of an error
        }

        // Initialization of white lattice
        if (std::filesystem::exists(filename_w)){

            std::vector<signed char> charVector_w;

            // black lattice read
            std::ifstream inFile_w(filename_w);

            int value;

            if (!inFile_w.is_open()) {
                std::cerr << "Error opening file for reading." << std::endl;
                return;  // Return an empty vector in case of an error
            }

            // Read each value from the file
            while (inFile_w >> value) {
                // Reverse the mapping: 0 to -1 and 1 to 1
                charVector_w.push_back((value == 0) ? -1 : 1);
            }

            // Close the file
            inFile_w.close();

            cout << "Initialized white lattice with preceeding results. ";
            // // printing initialized elements
            // for (const auto& element : charVector_w) {
            //     cout << static_cast<int>(element) << " ";
            // }
            cout << endl;

            CHECK_CUDA(cudaMemcpy(lattice_w, charVector_w.data(), num_lattices * nx * ny /2 * sizeof(*lattice_w), cudaMemcpyHostToDevice));
        }
        else{
            std::cerr << "Error opening file for reading. Could not find white lattice file." << std::endl;
            return;  // Return an empty vector in case of an error
        }
    }
    else {
        if (up){
            init_spins_up<<<blocks,THREADS>>>(lattice_b, nx, ny/2, num_lattices);
            init_spins_up<<<blocks,THREADS>>>(lattice_w, nx, ny/2, num_lattices);
        }
        else{
            //Initialize the arrays for white and black lattice
            CHECK_CURAND(curandGenerateUniform(lattice_rng, lattice_randvals, nx*ny/2*num_lattices));
            init_spins<<<blocks, THREADS>>>(lattice_b, lattice_randvals, nx, ny/2, num_lattices);
            //Initialize the arrays for white and black lattice
            CHECK_CURAND(curandGenerateUniform(lattice_rng, lattice_randvals, nx*ny/2*num_lattices));
            init_spins<<<blocks, THREADS>>>(lattice_w, lattice_randvals, nx, ny/2, num_lattices);
        }
    }
}

void write_updated_lattices(signed char *lattice_b, signed char *lattice_w, const long long nx, const long long ny, const int num_lattices, std::string lattice_b_file_name, std::string lattice_w_file_name){
    printf("Writing updated lattices to %s and %s \n", lattice_b_file_name.c_str(), lattice_w_file_name.c_str());
    // copy to host
    std::vector<signed char> h_lattice_b(num_lattices * nx * ny / 2);
    std::vector<signed char> h_lattice_w(num_lattices * nx * ny / 2);
    CHECK_CUDA(cudaMemcpy(h_lattice_b.data(), lattice_b, num_lattices * nx * ny / 2 * sizeof(*lattice_b), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_lattice_w.data(), lattice_w, num_lattices * nx * ny / 2 * sizeof(*lattice_w), cudaMemcpyDeviceToHost));
    // Open a file for writing the black lattice
    std::ofstream outFile_b(lattice_b_file_name, std::ios::out | std::ios::binary | std::ios::trunc);
    // Check if the file is open
    if (!outFile_b.is_open()) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }
    // Write each element of the black lattice vector to the file
    for (const auto& element : h_lattice_b) {
        char mappedValue = (element == 1) ? '1' : '0';
        outFile_b << mappedValue;
    }
    // Close the black lattice file
    outFile_b.close();
    // Open a file for writing the white lattice
    std::ofstream outFile_w(lattice_w_file_name, std::ios::out | std::ios::binary | std::ios::trunc);
    // Check if the file is open
    if (!outFile_w.is_open()) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }
    // Write each element of the white lattice vector to the file
    for (const auto& element : h_lattice_w) {
        char mappedValue = (element == 1) ? '1' : '0';
        outFile_w << mappedValue;
    }
    // Close the white lattice file
    outFile_w.close();
}

void write_lattice_to_disc(signed char *lattice_b, signed char *lattice_w, std::string filename, const long long nx, const long long ny, const int num_lattices) {
    printf("Writing lattice to %s...\n", filename.c_str());

    std::vector<signed char> lattice_h(nx*ny);
    std::vector<signed char> lattice_w_h(nx*ny/2);
    std::vector<signed char> lattice_b_h(nx*ny/2);

    cudaMemcpy(lattice_b_h.data(), lattice_b, nx * ny/2 * sizeof(*lattice_b), cudaMemcpyDeviceToHost);
    cudaMemcpy(lattice_w_h.data(), lattice_w, nx * ny/2 * sizeof(*lattice_w), cudaMemcpyDeviceToHost);

    int offset;

    for (int l = 0; l < num_lattices; l++){

        offset = l*nx*ny/2;

        for (int i = 0; i < nx; i++){
            for (int j=0; j < ny/2; j++){
                if (i%2 == 0){
                    lattice_h[i*ny+2*j+1] = lattice_w_h[offset + i*ny/2+j];
                    lattice_h[i*ny+2*j] = lattice_b_h[offset + i*ny/2+j];
                }
                else{
                    lattice_h[i*ny+2*j] = lattice_w_h[offset + i*ny/2+j];
                    lattice_h[i*ny+2*j+1] = lattice_b_h[offset + i*ny/2+j];
                }
            }
        }

        std::ofstream f;
        f.open(filename + std::string(".txt"));

        if (f.is_open()) {
            for (int i = 0; i < nx; i++) {
                for (int j = 0; j < ny; j++) {
                    f << (int)lattice_h[i * ny + j] << " ";
                }
                f << std::endl;
            }
        }
        f.close();
    }
}

void write_bonds(signed char* interactions, std::string filename, const long nx, const long ny, const int num_lattices){
    printf("Writing bonds to %s ...\n", filename.c_str());

    std::vector<signed char> interactions_host(2*nx*ny*num_lattices);

    CHECK_CUDA(cudaMemcpy(interactions_host.data(),interactions, 2*num_lattices*nx*ny*sizeof(*interactions), cudaMemcpyDeviceToHost));

    int offset;

    for (int l=0; l<num_lattices; l++){

        offset = l*nx*ny*2;

        std::ofstream f;
        f.open(filename + std::to_string(l) + std::string(".txt"));
        if (f.is_open()) {
            for (int i = 0; i < 2*nx; i++) {
                for (int j = 0; j < ny; j++) {
                    f << (int)interactions_host[offset + i * ny + j] << " ";
                }
                f << std::endl;
            }
        }
        f.close();
    }
}

template<bool is_black>
__global__ void update_lattice(
    signed char* lattice, signed char* __restrict__ op_lattice, const float* __restrict__ randvals, const signed char* interactions,
    const double *inv_temp, const long long nx, const long long ny, const int num_lattices, double* d_energy
) {

    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= nx*ny*num_lattices) return;

    // Calculate in which lattice we are
    int l_id = tid/(nx*ny);

    // Project tid back to single lattice
    int tid_sl = tid - l_id*nx*ny;

    int i = tid_sl/ny;
    int j = tid_sl%ny;

    // Set up periodic boundary conditions
    int ipp = (i + 1 < nx) ? i + 1 : 0;
    int inn = (i - 1 >= 0) ? i - 1: nx - 1;
    int jpp = (j + 1 < ny) ? j + 1 : 0;
    int jnn = (j - 1 >= 0) ? j - 1: ny - 1;

    int joff;
    int jcouplingoff;
    int icouplingpp;
    int icouplingnn;

    int offset = l_id * nx * ny;
    int offset_i = l_id * nx * ny * 4;

    if (!is_black) {
        icouplingpp = offset_i + 2*(nx-1)*ny + 2*(ny*(i+1) + j) + (i+1)%2;
        icouplingnn = offset_i + 2*(nx-1)*ny + 2*(ny*(inn+1) + j) + (i+1)%2;

        joff = (i % 2) ? jnn : jpp;

        if (i % 2) {
            jcouplingoff = offset_i + 2 * (i * ny + joff) + 1;
        } else {
            if (j + 1 >= ny) {
                jcouplingoff = offset_i + 2 * (i * ny + j + 1) - 1;
            } else {
                jcouplingoff = offset_i + 2 * (i * ny + joff) - 1;
            }
        }
    } else {
        icouplingpp = offset_i + 2*(nx-1)*ny + 2*(ny*(i+1) + j) + i%2;
        icouplingnn = offset_i + 2*(nx-1)*ny + 2*(ny*(inn+1) + j) + i%2;

        joff = (i % 2) ? jpp : jnn;

        if (i % 2) {
            if (j+1 >= ny) {
                jcouplingoff = offset_i + 2 * (i * ny + j + 1) - 1;
            } else {
                jcouplingoff = offset_i + 2 * (i * ny + joff) - 1;
            }
        } else {
            jcouplingoff = offset_i + 2 * (i * ny + joff) + 1;
        }
    }

    // Compute sum of nearest neighbor spins times the coupling
    signed char nn_sum = op_lattice[offset + inn*ny + j]*interactions[icouplingnn] + op_lattice[offset + i*ny + j]*interactions[offset_i + 2*(i*ny + j)]
                        + op_lattice[offset + ipp*ny + j]*interactions[icouplingpp] + op_lattice[offset + i*ny + joff]*interactions[jcouplingoff];


    signed char lij = lattice[offset + i*ny + j];

    // set device energy for each temp and each spin on lattice
    d_energy[tid]=inv_temp[l_id]*nn_sum*lij;

    // Determine whether to flip spin
    float acceptance_ratio = exp(-2 * d_energy[tid]);
    if (randvals[offset + i*ny + j] < acceptance_ratio) {
        lattice[offset + i*ny + j] = -lij;
        d_energy[tid] *= -1;
    }
}

void update(
    signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, const signed char* interactions,
    const double *inv_temp, const long long nx, const long long ny, const int num_lattices, const int blocks, double *d_energy
) {

    // Update black
    CHECK_CURAND(curandGenerateUniform(rng, randvals, num_lattices*nx*ny/2));
    update_lattice<true><<<blocks, THREADS>>>(lattice_b, lattice_w, randvals, interactions, inv_temp, nx, ny/2, num_lattices, d_energy);

    // Update white
    CHECK_CURAND(curandGenerateUniform(rng, randvals, num_lattices*nx*ny/2));
    update_lattice<false><<<blocks, THREADS>>>(lattice_w, lattice_b, randvals, interactions, inv_temp, nx, ny/2, num_lattices, d_energy);
}

__global__ void B2_lattices(
    signed char *lattice_b, signed char *lattice_w, const double *wave_vector,
    thrust::complex<double> *sum, const int nx, const int ny, const int num_lattices
){

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= nx*ny*num_lattices) return;

    int lattice_id = tid/(nx*ny);
    int offset = lattice_id*nx*ny;
    int tid_single_lattice = tid - offset;

    int i = tid_single_lattice/ny;
    int j = tid_single_lattice%ny;

    int b_orig_j;
    int w_orig_j;

    if (i%2==0){
        b_orig_j = 2*j +1;
        w_orig_j = 2*j;
    }

    else{
        b_orig_j = 2*j;
        w_orig_j = 2*j + 1;
    }

    thrust::complex<float> imag = thrust::complex<float>(0, 1.0f);

    float dot_b = wave_vector[0]*i + wave_vector[1]*b_orig_j;
    float dot_w = wave_vector[0]*i + wave_vector[1]*w_orig_j;

    sum[tid] = lattice_b[tid]*exp(imag*dot_b) + lattice_w[tid]*exp(imag*dot_w);
}

void calculate_B2(
    thrust::complex<double> *d_sum, signed char *lattice_b, signed char *lattice_w, thrust::complex<double> *d_store_sum, const double *d_wave_vector, const long nx, const long ny, const int num_lattices, const int blocks
){
    // Calculate B2 and reduce sum
    B2_lattices<<<blocks, THREADS>>>(lattice_b, lattice_w, d_wave_vector, d_sum, nx, ny/2, num_lattices);

    // cant we parallalize this too?
    for (int i=0; i<num_lattices; i++){

        if (temp_storage_nx_thrust == 0){
            CHECK_CUDA(cub::DeviceReduce::Sum(d_temp_nx_thrust, temp_storage_nx_thrust, d_sum + i*nx*ny/2, &d_store_sum[i], nx*ny/2));
            CHECK_CUDA(cudaMalloc(&d_temp_nx_thrust, temp_storage_nx_thrust));
        }

        CHECK_CUDA(cub::DeviceReduce::Sum(d_temp_nx_thrust, temp_storage_nx_thrust, d_sum + i*nx*ny/2, &d_store_sum[i], nx*ny/2));
    }
}

__global__ void abs_square(thrust::complex<double> *d_store_sum, const int num_lattices){

    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= num_lattices) return;

    d_store_sum[tid] = thrust::abs(d_store_sum[tid]) * thrust::abs(d_store_sum[tid]);
}

int create_results_folder(char* results){
    struct stat sb;

    if (stat(results, &sb) == 0){
        std::cout << "Results already exist, check file name";
        return 0;
    }
    else{
        mkdir(results, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        return 1;
    }
}

template<bool is_black>
__global__ void update_lattice_ob(
    signed char* lattice, signed char* __restrict__ op_lattice, const float* __restrict__ randvals, const signed char* interactions,
    const double *inv_temp, const long long nx, const long long ny, const int num_lattices, double* d_energy
){

    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= nx*ny*num_lattices) return;

    // Calculate in which lattice we are
    int l_id = tid/(nx*ny);

    // Project tid back to single lattice
    int tid_sl = tid - l_id*nx*ny;

    int i = tid_sl/ny;
    int j = tid_sl%ny;

    // Set up periodic boundary conditions
    int ipp = (i + 1 < nx) ? i + 1 : 0;
    int inn = (i - 1 >= 0) ? i - 1: nx - 1;
    int jpp = (j + 1 < ny) ? j + 1 : 0;
    int jnn = (j - 1 >= 0) ? j - 1: ny - 1;

    int joff;
    int jcouplingoff;
    int icouplingpp;
    int icouplingnn;

    int offset = l_id * nx * ny;
    int offset_i = l_id * nx * ny * 4;

    int c_up = 1-inn/(nx-1);
    int c_down = 1-(i+1)/nx;
    int c_side;

    if (!is_black) {
        icouplingpp = offset_i + 2*(nx-1)*ny + 2*(ny*(i+1) + j) + (i+1)%2;
        icouplingnn = offset_i + 2*(nx-1)*ny + 2*(ny*(inn+1) + j) + (i+1)%2;

        joff = (i % 2) ? jnn : jpp;

        if (i % 2) {
            
            jcouplingoff = offset_i + 2 * (i * ny + joff) + 1;

            c_side = 1 - jnn/(ny-1);

        } else {
            
            c_side = 1 - (j+1)/ny;

            if (j + 1 >= ny) {
                jcouplingoff = offset_i + 2 * (i * ny + j + 1) - 1;
            } 
            else {
                jcouplingoff = offset_i + 2 * (i * ny + joff) - 1;
            }
        }
    }
    else {

        icouplingpp = offset_i + 2*(nx-1)*ny + 2*(ny*(i+1) + j) + i%2;
        icouplingnn = offset_i + 2*(nx-1)*ny + 2*(ny*(inn+1) + j) + i%2;

        joff = (i % 2) ? jpp : jnn;

        if (i % 2) {
            
            c_side = 1-(j+1)/ny;

            if (j+1 >= ny) {
                jcouplingoff = offset_i + 2 * (i * ny + j + 1) - 1;
            } 
            else {
                jcouplingoff = offset_i + 2 * (i * ny + joff) - 1;
            }
        } 
        else {
            c_side = 1-jnn/(ny-1);
            jcouplingoff = offset_i + 2 * (i * ny + joff) + 1;
        }
    }

    signed char nn_sum = op_lattice[offset + inn*ny + j]*interactions[icouplingnn]*c_up + op_lattice[offset + i*ny + j]*interactions[offset_i + 2*(i*ny + j)]
                        + op_lattice[offset + ipp*ny + j]*interactions[icouplingpp]*c_down + op_lattice[offset + i*ny + joff]*interactions[jcouplingoff]*c_side;

    // Determine whether to flip spin

    // The exponent is exactly what calc_energy_ob does and which is calles again to store energy over same iterator in update loop. Instead here should be filled the energy array directly

    signed char lij = lattice[offset + i*ny + j];

    // set device energy for each temp and each spin on lattice
    d_energy[tid]=inv_temp[l_id]*nn_sum*lij;

    float acceptance_ratio = exp(-2*d_energy[tid]);
    if (randvals[offset + i*ny + j] < acceptance_ratio) {
        lattice[offset + i*ny + j] = -lij;
        d_energy[tid] *= -1;
    }
}

void update_ob(
    signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, const signed char* interactions,
    const double *inv_temp, const long long nx, const long long ny, const int num_lattices, const int blocks, double *d_energy
) {

    // Update black and store final Hamiltonian value for each cross term in d_energy
    CHECK_CURAND(curandGenerateUniform(rng, randvals, num_lattices*nx*ny/2));
    update_lattice_ob<true><<<blocks, THREADS>>>(lattice_b, lattice_w, randvals, interactions, inv_temp, nx, ny/2, num_lattices, d_energy);

    // Update white and store final Hamiltonian value for each cross term in d_energy
    CHECK_CURAND(curandGenerateUniform(rng, randvals, num_lattices*nx*ny/2));
    update_lattice_ob<false><<<blocks, THREADS>>>(lattice_w, lattice_b, randvals, interactions, inv_temp, nx, ny/2, num_lattices, d_energy);
}

__global__ void incrementalSumMagnetization(thrust::complex<double> *d_store_sum_0, thrust::complex<double> *d_store_sum_k, const int num_lattices, double *d_storeIncrementalSumMag_0, double *d_storeIncrementalSumMag_k){

    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= num_lattices) return;

    d_storeIncrementalSumMag_0[tid] += d_store_sum_0[tid].real();
    d_storeIncrementalSumMag_k[tid] += d_store_sum_k[tid].real();

}