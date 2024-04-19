#include <cuda_runtime.h>
#include <iostream>

__device__ unsigned short atomicOr(unsigned short* address, unsigned short val)
{
    unsigned int* address_as_uint = (unsigned int *)((char *)address - ((size_t)address & 2));
    unsigned int int_val = ((size_t)address & 2) ? (((unsigned int)val << 16)) : val;
    
    unsigned int int_old = atomicOr(address_as_uint, int_val);

    if((size_t)address & 2) {
        return (unsigned short)(int_old >> 16);
    }
    else{
        return (unsigned short)(int_old & 0xffff);
    }
}

__global__ void test(unsigned short* address, unsigned short val){
    atomicOr(address, val);
}

void test_atomicOr() {
    unsigned short h_data = 0x1222;
    unsigned short h_value = 0x5532;
    unsigned short expected_result = h_data | h_value;

    unsigned short *d_data;
    cudaMalloc(&d_data, sizeof(unsigned short));
    cudaMemcpy(d_data, &h_data, sizeof(unsigned short), cudaMemcpyHostToDevice);

    // Call the kernel function
    test<<<1,1>>>(d_data, h_value);

    // Copy the result back to host memory
    cudaMemcpy(&h_data, d_data, sizeof(unsigned short), cudaMemcpyDeviceToHost);

    // Check if the result matches the expected value
    if (h_data == expected_result){
        printf("Results correct");
    }

    cudaFree(d_data);
}

int main() {
    test_atomicOr();
    return 0;
}
