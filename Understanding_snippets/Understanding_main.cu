/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Mauro Bisson <maurob@nvidia.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <sys/types.h>
 #include <getopt.h>
 #include <unistd.h>
 #include <cuda_runtime.h>
 #include <curand_kernel.h>
 #include "cudamacro.h" /* for time() */
 #include "utils.h"
 #include <iostream>
 #include <sys/stat.h>
 #include <thrust/complex.h>
 #include <fstream>
 #include <iomanip>
 
 using namespace std;
 
 #define DIV_UP(a,b)     (((a)+((b)-1))/(b))
 
 #define THREADS  128
 
 // Bits per spin
 #define BIT_X_SPIN (4)
 
 // MIN & MAX Operator
 #define MIN(a,b)	(((a)<(b))?(a):(b))
 #define MAX(a,b)	(((a)>(b))?(a):(b))
 
 // 2048+: 16, 16, 2, 1
 //  1024: 16, 16, 1, 2
 //   512:  8,  8, 1, 1
 //   256:  4,  8, 1, 1
 //   128:  2,  8, 1, 1
 // 2*SPIN_X_WORD*2*BLOCK_X*BMULT_X
 // BMULT_X Block Multiple X Direction
 
 // Thread size block x and y
 #define BLOCK_X (2)
 #define BLOCK_Y (8)
 
 // Unclear
 #define BMULT_X (1)
 #define BMULT_Y (1)
 
 // Maximum number of GPUs
 #define MAX_GPU	(256)
 
 __device__ __forceinline__ unsigned int __mypopc(const unsigned int x) {
     return __popc(x);
 }
 
 __device__ __forceinline__ unsigned long long int __mypopc(const unsigned long long int x) {
     return __popcll(x);
 }
 
 enum {C_BLACK, C_WHITE};
 
 __device__ __forceinline__ uint2 __mymake_int2(const unsigned int x,
                                        const unsigned int y) {
     return make_uint2(x, y);
 }
 
 __device__ __forceinline__ ulonglong2 __mymake_int2(const unsigned long long x,
                                             const unsigned long long y) {
     return make_ulonglong2(x, y);
 }