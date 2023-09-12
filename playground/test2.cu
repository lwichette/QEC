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



int main(int argc, char **argv) {

  std::vector< int > psi_l;

  for(int i=0; i<10; i++){
    psi_l.push_back(i);
  }

  for(int i=)
}