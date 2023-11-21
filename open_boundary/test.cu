#include <cstdlib>
#include <fstream>
#include <iostream>
#include <time.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>
#include <thrust/reduce.h>
#include<thrust/device_vector.h>
#include <cmath>
#include <math.h>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <filesystem>
#include <boost/program_options.hpp>

#include "../header/defines.h"
#include "../header/utils.cuh"
#include "../header/cudamacro.h"

using namespace std;


__global__ void init_keys(int *d_keys, const int num_lattices, const int nx, const int ny){
    
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= nx*ny*num_lattices) return;

    d_keys[tid] = tid/(nx*ny);
}

int main(int argc, char **argv){
    /*
    const int N = 7;
    std::vector<int> B = {9, 8, 7, 6, 5, 4, 3}; // input values
    std::vector<int> C(4);                         // output keys
    std::vector<int> D(4);                         // output values

    int *d_A, *d_B;
    cudaMalloc(&d_A, N*sizeof(int));
    cudaMalloc(&d_B, N*sizeof(int));
    cudaMemcpy(d_A, A.data(), N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N*sizeof(int), cudaMemcpyHostToDevice);

    int *d_C, *d_D;
    cudaMalloc(&d_C, 4*sizeof(int));
    cudaMalloc(&d_D, 4*sizeof(int));

    thrust::pair<int*,int*> new_end;
    new_end = thrust::reduce_by_key(thrust::device, d_A, d_A + N, d_B, d_C, d_D);

    cudaMemcpy(C.data(), d_C, 4*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(D.data(), d_D, 4*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0; i<4; i++){
        cout << C[i] << endl;
        cout << D[i] << endl;
    }
    */

    const int num_lattices = 4;
    const int L = 12;

    const int threads = 128;
    int blocks = (L * L * num_lattices + threads - 1) / threads;

    std::vector<int> B(num_lattices*L);

    for (int i=0; i<num_lattices*L; i++){
        B[i] = i/L;
    }

    int *d_keys;
    cudaMalloc(&d_keys, num_lattices*L*sizeof(int));

    init_keys<<<blocks, threads>>>(d_keys, num_lattices, L, L);

    int *d_B;
    cudaMalloc(&d_B, num_lattices*L*sizeof(int));
    cudaMemcpy(d_B, B.data(), num_lattices*L*sizeof(int), cudaMemcpyHostToDevice);

    int *d_C, *d_D;
    cudaMalloc(&d_C, num_lattices*sizeof(int));
    cudaMalloc(&d_D, num_lattices*sizeof(int));
    
    std::vector<int> C(num_lattices);
    std::vector<int> D(num_lattices);
    
    thrust::pair<int*,int*> new_end;
    new_end = thrust::reduce_by_key(thrust::device, d_keys, d_keys + num_lattices*L, d_B, d_C, d_D);

    cudaMemcpy(C.data(), d_C, num_lattices*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(D.data(), d_D, num_lattices*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0; i < num_lattices; i++){
        cout << C[i] << endl;
        cout << D[i] << endl;
    }
}