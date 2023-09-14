#include <cstdlib>
#include <iostream>
#include <time.h>
#include <fstream>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>
#include <vector>

using namespace std;

#define THREADS 128

void sum_with_index(float *d_array, int num_values, float* d_res, int i){
  // Variables used for sum reduction
  void *d_temp = NULL;
  size_t temp_storage = 0;

  if (i==-1){
    // Sum reduction
    cub::DeviceReduce::Sum(d_temp, temp_storage, d_array, d_res, num_values);
    cudaMalloc(&d_temp, temp_storage);
    cub::DeviceReduce::Sum(d_temp, temp_storage, d_array, d_res, num_values);
  }
  else{
    // Sum reduction
    cub::DeviceReduce::Sum(d_temp, temp_storage, d_array, &d_res[i], num_values);
    cudaMalloc(&d_temp, temp_storage);
    cub::DeviceReduce::Sum(d_temp, temp_storage, d_array, &d_res[i], num_values);
  }
  
}

float* sum_one(float *d_array, int num_values){
  // Variables used for sum reduction
  void *d_temp = NULL;
  size_t temp_storage = 0;

  float *d_sum;
  cudaMalloc(&d_sum, sizeof(float));
  
  // Sum reduction
  cub::DeviceReduce::Sum(d_temp, temp_storage, d_array, d_sum, num_values);
  cudaMalloc(&d_temp, temp_storage);
  cub::DeviceReduce::Sum(d_temp, temp_storage, d_array, d_sum, num_values);
  
  return d_sum;
}

int main(int argc, char **argv) {

  int num_values = 10;
  
  float *h_array = (float *)malloc(num_values*sizeof(float));

  for (int i=0; i<num_values; i++){
    h_array[i] = i;
  }

  float *d_array;
  cudaMalloc(&d_array, num_values*sizeof(float));
  
  cudaMemcpy(d_array, h_array, num_values*sizeof(float), cudaMemcpyHostToDevice);

  float* d_res_single;
  cudaMalloc(&d_res, num_values*sizeof(float));
  
  sum_with_index(d_array, num_values, d_res, -1);
  
  float *h_sum = (float *)malloc(num_values*sizeof(float));
  cudaMemcpy(h_sum, d_res, num_values*sizeof(float), cudaMemcpyDeviceToHost);

  printf("%f", h_sum[0]);
}