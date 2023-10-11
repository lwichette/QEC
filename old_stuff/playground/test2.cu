#include <cstdlib>
#include <iostream>
#include <time.h>
#include <fstream>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>
#include <vector>
#include <sys/stat.h>

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

void create_array(int n){
  std::vector<float> test(n);
}

__global__ void initialize(float *d_test, int num_values){
  
  const long long tid = static_cast<long long>(threadIdx.x + blockIdx.x * blockDim.x);

  if (tid>=num_values) return;

  d_test[tid] = 3000;

}
int main(int argc, char **argv) {
  char* results = "results/test3";

  // Structure which would store the metadata
  struct stat sb;

  if (stat(results, &sb) == 0){
      cout << "Results already exist, check file name";
      return 0;
  }
  else{
      mkdir(results, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }
}