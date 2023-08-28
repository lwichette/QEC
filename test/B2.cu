#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <iostream>
#include <cub/cub.cuh>

using namespace std;
/**
* CUDA Kernel Device code
*
* Computes the vector addition of A and B into C. The 3 vectors have the same
* number of elements numElements.
*/
const int threadsPerBlock = 256;

__global__ void
B2(const signed char *A, const float *B, thrust::complex<float> *C, int ny, int nx)
{
    /*
    Calculates the inner sum of eq B2. Sum of blocks and absolute value, square needs to be done on the host.
    */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    thrust::complex<float> imag = thrust::complex<float>(0, 1.0f);

    if (tid < nx*ny){       
        int i = tid/ny;
        int j = tid%ny;
        
        float dot = B[0]*i + B[1]*j;
        C[tid] = A[tid]*exp(imag*dot);
    }
}

/**
* Host main routine
*/
int main(void){
    // Print the vector length to be used, and compute its size
    int nx = 50;
    int ny = 50;
    
    // Launch the Vector Add CUDA Kernel
    int blocksPerGrid =(nx*ny + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate the host input vector A
    signed char *h_A = (signed char *)malloc(nx*ny*sizeof(signed char));

    // Initialize the host input vectors
    for (int i = 0; i < nx*ny; ++i){
        if (i<nx*ny/2) {
            h_A[i] = 1;
        }
        else {
            h_A[i] = -1;
        }
    }

    // Allocate the host input vector B
    float *h_B = (float *)malloc(2*sizeof(float));
    h_B[0] = 1;
    h_B[1] = 0;

    // Allocate the host output vector C
    thrust::complex<float> *h_C = (thrust::complex<float> *)malloc(nx*ny*sizeof(thrust::complex<float>));

    // Allocate the device input vector A
    signed char *d_A = NULL;
    cudaMalloc(&d_A, nx*ny*sizeof(signed char));

    // Allocate the device input vector B
    float *d_B = NULL;
    cudaMalloc(&d_B, 2*sizeof(float));

    // Allocate the device output vector C
    thrust::complex<float> *d_C = NULL;
    cudaMalloc(&d_C, nx*ny*sizeof(thrust::complex<float>));

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    cudaMemcpy(d_A, h_A, nx*ny*sizeof(signed char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 2*sizeof(float), cudaMemcpyHostToDevice);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, ny, nx);

    // Reduce sum
    thrust::complex<float>* d_out = NULL;
    cudaMalloc(&d_out, sizeof(thrust::complex<float>));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_C, d_out, nx*ny);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_C, d_out, nx*ny);

    thrust::complex<float>* hostsum = (thrust::complex<float> *)malloc(sizeof(thrust::complex<float>));
    cudaMemcpy(hostsum, d_out, sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);
    
    cout << *hostsum;

    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C); 
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}
