#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <getopt.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_kernel.h>
#include "cudamacro.h"
#include "utils.h"
#include <iostream>
#include <sys/stat.h>
#include <thrust/complex.h>
#include <fstream>
#include <iomanip>


using namespace std;

#define THREADS 128

#define BLOCK_X (2)
#define BLOCK_Y (8)

#define BMULT_X (1)
#define BMULT_Y (1)

