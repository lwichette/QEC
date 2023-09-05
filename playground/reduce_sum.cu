#include <cstdlib>
#include <iostream>
#include <time.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>

using namespace std;

#define TCRIT 1
#define THREADS 128

    

// Initialize lattice spins
__global__ void init_spins(signed char* lattice, const long long nx, const long long ny) {
        
    const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (tid >= nx * ny) return;
        
    lattice[tid] = 1;
}

__global__ void B2_lattices(signed char* lattice_b, signed char* lattice_w, const float *wave_vector, thrust::complex<float> *sum,  int nx, int ny){
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    int i = tid/ny;
    int j = tid%ny;

    if (i>=nx || j >= ny) return;

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

int main(void){
    long nx = 4;
    long ny = 4;  

    int blocks = (nx*ny*2 + THREADS -1)/THREADS;

    // Allocate the wave vector and copy it to GPU memory
    float *wave_vector = (float *)malloc(2*sizeof(float));
    wave_vector[0] = 0;
    wave_vector[1] = 0;

    float *d_wave_vector;
    cudaMalloc(&d_wave_vector, 2 * sizeof(*d_wave_vector));
    cudaMemcpy(d_wave_vector, wave_vector, 2*sizeof(float), cudaMemcpyHostToDevice);

    // Setup black and white lattice arrays on device
    signed char *lattice_b, *lattice_w;
    cudaMalloc(&lattice_b, nx * ny/2 * sizeof(*lattice_b));
    cudaMalloc(&lattice_w, nx * ny/2 * sizeof(*lattice_w));

    init_spins<<<blocks, THREADS>>>(lattice_b, nx, ny);
    init_spins<<<blocks, THREADS>>>(lattice_w, nx, ny);

    thrust::complex<float> *d_sum;
    cudaMalloc(&d_sum, nx*ny/2*sizeof(*d_sum));

    B2_lattices<<<blocks, THREADS>>>(lattice_b, lattice_w, d_wave_vector, d_sum, nx, ny/2);

    // Reduce sum
    thrust::complex<float>* d_out = NULL;
    cudaMalloc(&d_out, sizeof(*d_out));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_out, nx*ny/2);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_out, nx*ny/2);
    
    thrust::complex<float>* hostsum = (thrust::complex<float> *)malloc(sizeof(thrust::complex<float>));
    
    cudaMemcpy(hostsum, d_out, sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    cout << *hostsum;
}