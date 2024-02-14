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
#include <sys/stat.h>
#include <getopt.h>
#include <unistd.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cudamacro.h" /* for time() */
#include "utils.h"
#include <iostream>
#include <thrust/complex.h>
#include <cub/cub.cuh>
#include <cmath>
#include <ctime>

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

// Unclear
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

void *d_temp = NULL;
size_t temp_storage = 0;

void *d_temp_complex = NULL;
size_t temp_storage_complex = 0;

template<int BDIM_X,
	 int BDIM_Y,
	 int LOOP_X,
	 int LOOP_Y,
	 int BITXSP,
	 int COLOR,
	 typename INT_T,
	 typename INT2_T>
__global__  void latticeInit_k(const int devid,
			       const long long seed,
                               const int it,
                               const long long begY,
                               const long long dimX, // ld
                                     INT2_T *__restrict__ vDst) {

	// i linearized y position in blocks and threads
	// j linearized x position in blocks and threads
	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + threadIdx.y;
	const int __j = blockIdx.x*BDIM_X*LOOP_X + threadIdx.x;

	// calculate number of spins per word
	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	// get thread id
	const long long tid = ((devid*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x)*BDIM_X*BDIM_Y +
	                       threadIdx.y*BDIM_X + threadIdx.x;

	// Random number generator
	curandStatePhilox4_32_10_t st;
	// Unclear what exactly
	curand_init(seed, tid, static_cast<long long>(2*SPIN_X_WORD)*LOOP_X*LOOP_Y*(2*it+COLOR), &st);

	// tmp 2D array of type unsigned long long of size (1x2)
	INT2_T __tmp[LOOP_Y][LOOP_X];

	// Initialize array with (0,0)
	#pragma unroll //compiler more efficient
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__tmp[i][j] = __mymake_int2(INT_T(0),INT_T(0));
		}
	}

	//INT = Unsigned long long
	//INT2 == (ull, ull)
	// BIT X SP = 4
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			#pragma unroll
			for(int k = 0; k < 8*sizeof(INT_T); k += BITXSP) {
				// Logical or plus shifting --> Initialize spins to up or down
				if (curand_uniform(&st) < 0.5f) {
					__tmp[i][j].x |= INT_T(1) << k;
				}
				if (curand_uniform(&st) < 0.5f) {
					__tmp[i][j].y |= INT_T(1) << k;
				}
			}
		}
	}

	// Set values in overall array
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			vDst[(begY + __i + i*BDIM_Y)*dimX + __j+j*BDIM_X] = __tmp[i][j];
		}
	}
	return;
}

template<int BDIM_X,
	 int BDIM_Y,
	 int LOOP_X,
	 int LOOP_Y,
	 int BITXSP,
	 typename INT_T,
	 typename INT2_T>
__global__  void hamiltInitB_k(const int devid,
			       const float tgtProb,
			       const long long seed,
                               const long long begY,
                               const long long dimX, // ld
                                     INT2_T *__restrict__ hamB) {

	// i column index in block thread picture, j row index in block thread picture
	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + threadIdx.y;
	const int __j = blockIdx.x*BDIM_X*LOOP_X + threadIdx.x;

	// Thread id
	const long long tid = ((devid*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x)*BDIM_X*BDIM_Y +
	                       threadIdx.y*BDIM_X + threadIdx.x;

	// Random number generator
	curandStatePhilox4_32_10_t st;
	curand_init(seed, tid, 0, &st);

	// array of tuples of size (1,2) unsigned long long
	// Set entries to zero tuples of unsigned long long
	INT2_T __tmp[LOOP_Y][LOOP_X];
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__tmp[i][j] = __mymake_int2(INT_T(0),INT_T(0));
		}
	}

	// For each black spin, randomly generate 4 interactions
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			#pragma unroll
			for(int k = 0; k < 8*sizeof(INT_T); k += BITXSP) {
				#pragma unroll
				for(int l = 0; l < BITXSP; l++) {
					if (curand_uniform(&st) < tgtProb) {
						__tmp[i][j].x |= INT_T(1) << (k+l);
					}
					if (curand_uniform(&st) < tgtProb) {
						__tmp[i][j].y |= INT_T(1) << (k+l);
					}
				}
			}
		}
	}

	// Fill array with the interaction terms
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			hamB[(begY + __i+i*BDIM_Y)*dimX + __j+j*BDIM_X] = __tmp[i][j];
		}
	}
	return;
}

template<int BDIM_X,
	 int BDIM_Y,
	 int LOOP_X,
	 int LOOP_Y,
	 int BITXSP,
	 typename INT_T,
	 typename INT2_T>
__global__ void hamiltInitW_k(const int xsl,
			      const int ysl,
			      const long long begY,
		              const long long dimX,
		              const INT2_T *__restrict__ hamB,
		                    INT2_T *__restrict__ hamW) {

	// Thread id x and y position
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	// row and column index of block-thread image
	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + tidy;
	const int __j = blockIdx.x*BDIM_X*LOOP_X + tidx;

	// Array of unsigned long long tuples
	// Load corresponding interactions from hamB
	INT2_T __me[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__me[i][j] = hamB[(begY+__i+i*BDIM_Y)*dimX + __j+j*BDIM_X];
		}
	}

	// Initialize arrays for up/side/down neighbors for white words
	INT2_T __up[LOOP_Y][LOOP_X];
	INT2_T __ct[LOOP_Y][LOOP_X];
	INT2_T __dw[LOOP_Y][LOOP_X];
	INT2_T __sd[LOOP_Y][LOOP_X];

	// the 4 bits of me codify: <upJ, downJ, leftJ, rightJ>
	// 0x888888 --> 100010001000 ...
	// Get first bit in every group of four and shift it by one to the right for up array
	// i.e. get up neighbor of black spin and store it at second position
	// 0x44444 --> 010001000100
	// get second bit in every group of four and shift it by one to the left for down array
	// get down neighbor of black spin and store it at first position
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__up[i][j].x = (__me[i][j].x & 0x8888888888888888ull) >> 1;
			__up[i][j].y = (__me[i][j].y & 0x8888888888888888ull) >> 1;

			__dw[i][j].x = (__me[i][j].x & 0x4444444444444444ull) << 1;
			__dw[i][j].y = (__me[i][j].y & 0x4444444444444444ull) << 1;
		}
	}

	// check row parity
	const int readBack = !(__i%2); // this kernel reads only BLACK Js

	// 8*8 = 64 bits per word
	const int BITXWORD = 8*sizeof(INT_T);

	if (!readBack) {
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {

				// 0x22222 --> 001000100010
				// get third bit in every group of four and shift it by one to the right for ct array
				// i.e. get leftJ and move it to the fourth position
				__ct[i][j].x = (__me[i][j].x & 0x2222222222222222ull) >> 1;
				__ct[i][j].y = (__me[i][j].y & 0x2222222222222222ull) >> 1;

				// 0x1111 --> 000100010001
				// get fourth bit in every group of four and shift it by (BITXSP + 1) to the left or right ,i.e. to the third position in the
				// prior/next 4 bit group or by (BITXWORD-BITXSP - 1) to the right and perform logical or with already existing ct
				// ct contains then at every 3rd and 4th position in the 4 bits an entry
				__ct[i][j].x |= (__me[i][j].x & 0x1111111111111111ull) << (BITXSP+1);
				__ct[i][j].y |= (__me[i][j].x & 0x1111111111111111ull) >> (BITXWORD-BITXSP - 1);
				__ct[i][j].y |= (__me[i][j].y & 0x1111111111111111ull) << (BITXSP+1);

				// get fourth bit of every four bit group and shift it by 59 to the right -- > Only one entry at the 63th bit
				// set __sd[i][j] = 0
				__sd[i][j].x = (__me[i][j].y & 0x1111111111111111ull) >> (BITXWORD-BITXSP - 1);
				__sd[i][j].y = 0;
			}
		}
	} else {
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				// Get fourth bit in every group of four and shift it by one to the left, i.e. to the third position
				__ct[i][j].x = (__me[i][j].x & 0x1111111111111111ull) << 1;
				__ct[i][j].y = (__me[i][j].y & 0x1111111111111111ull) << 1;

				// Right part: Get third bit in every group of four and shift it to the fourth position in the next group of four (first and third line)
				// (Second line): Get third bit in every group of four and shift it by 59 to the left --> Only last third bit is at the fourth bit location
				// Logical or: Perform logical or of existing ct with right part to have entries at third and fourth position in each block of four
				__ct[i][j].y |= (__me[i][j].y & 0x2222222222222222ull) >> (BITXSP+1);
				__ct[i][j].x |= (__me[i][j].y & 0x2222222222222222ull) << (BITXWORD-BITXSP - 1);
				__ct[i][j].x |= (__me[i][j].x & 0x2222222222222222ull) >> (BITXSP+1);

				// Get every third bit and shift it by 59 to the left --> last third bit to the fourth position in sd[i][j].y
				__sd[i][j].y = (__me[i][j].x & 0x2222222222222222ull) << (BITXWORD-BITXSP - 1);
				__sd[i][j].x = 0;
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {

		// calc row index
		const int yoff = begY+__i + i*BDIM_Y;

		// upOff if we are at a boarder of a lattice then take last row, else take row -1
		const int upOff = ( yoff   %ysl) == 0 ? yoff+ysl-1 : yoff-1;
		// downOff: if we are at a lower boarder of a sublattice, take first row, else take row + 1
		const int dwOff = ((yoff+1)%ysl) == 0 ? yoff-ysl+1 : yoff+1;

		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {

			// get column index
			const int xoff = __j + j*BDIM_X;

			// perform logical or with given binary entry
			// yoff*dimX + xoff = pointer at given tuple word lattice entry
			// upOff*dimX + xoff = pointer to upper neighbor
			// dwOff*dimX + xoff = pointer to down neighbor
			// subsequently fills all 64 bits with the up, down, left, right neighbors
			atomicOr(&hamW[yoff*dimX + xoff].x, __ct[i][j].x);
			atomicOr(&hamW[yoff*dimX + xoff].y, __ct[i][j].y);

			atomicOr(&hamW[upOff*dimX + xoff].x, __up[i][j].x);
			atomicOr(&hamW[upOff*dimX + xoff].y, __up[i][j].y);

			atomicOr(&hamW[dwOff*dimX + xoff].x, __dw[i][j].x);
			atomicOr(&hamW[dwOff*dimX + xoff].y, __dw[i][j].y);


			// Depending on which row we are in
			// Check whether we are at bordering columns of sublattices
			// Get column of left right neighbor and perform bitwise or
			const int sideOff = readBack ? (  (xoff   %xsl) == 0 ? xoff+xsl-1 : xoff-1 ):
						       ( ((xoff+1)%xsl) == 0 ? xoff-xsl+1 : xoff+1);

			atomicOr(&hamW[yoff*dimX + sideOff].x, __sd[i][j].x);
			atomicOr(&hamW[yoff*dimX + sideOff].y, __sd[i][j].y);
		}
	}
	return;
}

template<int BDIM_X,
	 int BDIM_Y,
	 int TILE_X,
	 int TILE_Y,
	 int FRAME_X,
	 int FRAME_Y,
	 typename INT2_T>
__device__ void loadTile(const int slX,
			 const int slY,
			 const long long begY,
			 const long long dimX,
			 const INT2_T *__restrict__ v,
			       INT2_T tile[][TILE_X+2*FRAME_X]) {

	// x,y block indices
	const int blkx = blockIdx.x;
	const int blky = blockIdx.y;

	// x,y thread indices
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	// TILE_X = BLOCK_X*BMULT_X, TILE = [16,32]
	// X and Y startpoint, Y offset by begY depending on GPU
	const int startX =        blkx*TILE_X;
	const int startY = begY + blky*TILE_Y;

	// Loop over BMULT_Y and BMULT_X
	// For each block load Spinwords of size 16x32 in tiles
	#pragma unroll
	for(int j = 0; j < TILE_Y; j += BDIM_Y) {
		// yoffset for current thread idy
		int yoff = startY + j + tidy;

		#pragma unroll
		for(int i = 0; i < TILE_X; i += BDIM_X) {
			// xoffset for current thread idx
			const int xoff = startX + i + tidx;
			tile[FRAME_Y + j + tidy][FRAME_X + i + tidx] = v[yoff*dimX + xoff];
		}
	}

	// if tidy == 0
	if (tidy == 0) {
		// if beginning of Y % size of sublattice == 0 --> if we are at start of a new sublattice
		// set offset to last row, else to startY - 1
		int yoff = (startY % slY) == 0 ? startY+slY-1 : startY-1;

		#pragma unroll
		// Loop over BMULT_Y
		// Get up neighbors
		for(int i = 0; i < TILE_X; i += BDIM_X) {
			const int xoff = startX + i + tidx;
			tile[0][FRAME_X + i + tidx] = v[yoff*dimX + xoff];
		}

		// Down neighbors
		yoff = ((startY+TILE_Y) % slY) == 0 ? startY+TILE_Y - slY : startY+TILE_Y;

		#pragma unroll
		for(int i = 0; i < TILE_X; i += BDIM_X) {
			const int xoff = startX + i+tidx;
			tile[FRAME_Y + TILE_Y][FRAME_X + i + tidx] = v[yoff*dimX + xoff];
		}

		// the other branch in slower so skip it if possible
		// if BLOCK_X <= TILE_Y
		if (BDIM_X <= TILE_Y) {
			// Find left neighbors
			int xoff = (startX % slX) == 0 ? startX+slX-1 : startX-1;

			#pragma unroll
			for(int j = 0; j < TILE_Y; j += BDIM_X) {
				yoff = startY + j+tidx;
				tile[FRAME_Y + j+tidx][0] = v[yoff*dimX + xoff];
			}

			// right neighbors
			xoff = ((startX+TILE_X) % slX) == 0 ? startX+TILE_X - slX : startX+TILE_X;

			#pragma unroll
			for(int j = 0; j < TILE_Y; j += BDIM_X) {
				yoff = startY + j+tidx;
				tile[FRAME_Y + j+tidx][FRAME_X + TILE_X] = v[yoff*dimX + xoff];
			}
		} else {
			// get left and right neighbors
			if (tidx < TILE_Y) {
				int xoff = (startX % slX) == 0 ? startX+slX-1 : startX-1;

				yoff = startY + tidx;
				tile[FRAME_Y + tidx][0] = v[yoff*dimX + xoff];;

				xoff = ((startX+TILE_X) % slX) == 0 ? startX+TILE_X - slX : startX+TILE_X;
				tile[FRAME_Y + tidx][FRAME_X + TILE_X] = v[yoff*dimX + xoff];
			}
		}
	}
	return;
}

template<int BDIM_X,
	 int BDIM_Y,
	 int LOOP_X,
	 int LOOP_Y,
	 int BITXSP,
	 int COLOR,
	 typename INT_T,
	 typename INT2_T>
__global__
void spinUpdate_open_bdry(const int devid,
		      const long long seed,
		      const int it,
		      const int slX, // sublattice size X of one color (in words or word tuples??)
		      const int slY, // sublattice size Y
		      const long long begY,
		      const long long dimX, // ld
		      const float vExp[][5],
		      const INT2_T *__restrict__ jDst,
		      const INT2_T *__restrict__ vSrc,
		            INT2_T *__restrict__ vDst) {

	// calc how many spins per word
	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	// x and y location in Thread lattice
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	// Initialize shared memory of Block size + neighbors
	__shared__ INT2_T shTile[BDIM_Y*LOOP_Y+2][BDIM_X*LOOP_X+2];

	// Load spin tiles of opposite lattice
	loadTile<BDIM_X, BDIM_Y,
		 BDIM_X*LOOP_X,
		 BDIM_Y*LOOP_Y,
		 1, 1, INT2_T>(slX, slY, begY, dimX, vSrc, shTile);

	// __shExp[cur_s{0,1}][sum_s{0,1}] = __expf(-2*cur_s{-1,+1}*F{+1,-1}(sum_s{0,1})*INV_TEMP)
	// Shared memory to store Exp
	__shared__ float __shExp[2][5];

	// for small lattices BDIM_X/Y may be smaller than 2/5
	// Load exponentials into shared memory
	#pragma unroll
	for(int i = 0; i < 2; i += BDIM_Y) {
		#pragma unroll
		for(int j = 0; j < 5; j += BDIM_X) {
			if (i+tidy < 2 && j+tidx < 5) {
				__shExp[i+tidy][j+tidx] = vExp[i+tidy][j+tidx];
			}
		}
	}
	__syncthreads();

	// get i and j location in block/thread grid
	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + tidy;
	const int __j = blockIdx.x*BDIM_X*LOOP_X + tidx;

	// calculate thread id
	const long long tid = ((devid*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x)*BDIM_X*BDIM_Y +
	                       threadIdx.y*BDIM_X + threadIdx.x;

	// array of size BMULT_Y x BMULT_X of unsigned long long
	INT2_T __me[LOOP_Y][LOOP_X];

	// Store spin words in array
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__me[i][j] = vDst[(begY+__i+i*BDIM_Y)*dimX + __j+j*BDIM_X];
		}
	}

	// initialize up, down center arrays
	INT2_T __up[LOOP_Y][LOOP_X];
	INT2_T __ct[LOOP_Y][LOOP_X];
	INT2_T __dw[LOOP_Y][LOOP_X];

	// Load up, down, center neighbors from other word lattice
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			// up has +0 in y direction index as shift by additional row in load tile. Same row plus one accordingly and only one down a plus two.
			// same x direction thread goes to plus one by additional entry in x direction in loadTile, too.
			__up[i][j] = shTile[i*BDIM_Y +   tidy][j*BDIM_X + 1+tidx];
			__ct[i][j] = shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 1+tidx];
			__dw[i][j] = shTile[i*BDIM_Y + 2+tidy][j*BDIM_X + 1+tidx];
		}
	}

	// BDIM_Y is power of two so row parity won't change across loops
	// Check which color and whether row (__i) is even or odd
	// Example: black lattice, even row --> readBack = 1
	const int readBack = (COLOR == C_BLACK) ? !(__i%2) : (__i%2);

	// Load missing side neighbors
	INT2_T __sd[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			// Hence with read back we are missing left neighbor and without readback missing right neighbor in center tile
			__sd[i][j] = (readBack) ? shTile[i*BDIM_Y + 1+tidy][j*BDIM_X +   tidx]:
						  shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 2+tidx];
		}
	}

	// if read back true: Left neighbor of most left spin entry in me must be deduced from rightest spin in sd array and combined with remaining spins from ct.
	// if read back false: right neighbor of most right spin entry in me must be deduced from leftest spin in sd array and combined with remaining spins from ct.

	// Where we ended
	// Rearrange left and right neighbors and update __sd[i,j] by filling it with the "right" neighbors
	// which become left neighbors
	if (readBack) {
		#pragma unroll
		// (BLACK LATTICE) Shift __sd such that it contains the left neighbors of the corresponding __me word
		// (BLACK LATTICE) __ct then contains the right neighbors
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__sd[i][j].x = (__ct[i][j].x << BITXSP) | (__sd[i][j].y >> (8*sizeof(__sd[i][j].y)-BITXSP)); // looks like furthest spin on the left side is in binary rep most at most right position!
				__sd[i][j].y = (__ct[i][j].y << BITXSP) | (__ct[i][j].x >> (8*sizeof(__ct[i][j].x)-BITXSP)); // only the x word needs left neighbor from the sd array. the y word gets its remaining spin from the x word.
			}
		}
	} else {
		// (BLACK LATTICE) Shift __sd such that it contains the right neighbors of the corresponding __me word
		// (BLACK LATTICE) __ct then contains the left neighbors
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__sd[i][j].y = (__ct[i][j].y >> BITXSP) | (__sd[i][j].x << (8*sizeof(__sd[i][j].x)-BITXSP));
				__sd[i][j].x = (__ct[i][j].x >> BITXSP) | (__ct[i][j].y << (8*sizeof(__ct[i][j].y)-BITXSP));
			}
		}
	}

	// When Hamiltonian is used
	if (jDst != NULL) {
		// Initialize array of size (1,2) to store the interaction terms
		INT2_T __J[LOOP_Y][LOOP_X];

		// Load interactions for current word tuple we are in
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__J[i][j] = jDst[(begY+__i+i*BDIM_Y)*dimX + __j+j*BDIM_X];
			}
		}


		// Now the idea is to apply and operation with 0x088088..ull with 0 at positions of open boundary in correct direction and correct spin position in the word
		// after the XOR and the shift the result looks like 0000|0001|0001|.. for each direction the results are than summed.
		// Hence, by setting the 4 bit group to 0000 would result in not regarding this hamiltonian term.
		// to set a 4 bit group to zero it is sufficient to execute a bitwise & with 0x088088..ull with 0 at the place of spins at the boundary
		// There can be the case of all spins inside a word at the boundary 0x0000..ull or only one spin in direction left or right 0x0888..ull and 0x8888..0ull.
		// The left right direction choice is dependent on the color and row parity.

		// apply them
		// the 4 bits of J codify: <upJ, downJ, leftJ, rightJ>
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {

				// Perform bitwise or operation
				// Column of left side gets the first bit in every group of four which is then shifted by 3 to the right left because spins are only
				// at the fourth location
				// XOR is then performed to change sign of spins

				__up[i][j].x ^= (__J[i][j].x & 0x8888888888888888ull) >> 3;
				__up[i][j].y ^= (__J[i][j].y & 0x8888888888888888ull) >> 3;

				// get down interaction and shift it to the right place
				__dw[i][j].x ^= (__J[i][j].x & 0x4444444444444444ull) >> 2;
				__dw[i][j].y ^= (__J[i][j].y & 0x4444444444444444ull) >> 2;

				if (readBack) {
					// __sd[][] holds "left" spins
					// __ct[][] holds "right" spins
					// get left interaction and shift it to the right position
					__sd[i][j].x ^= (__J[i][j].x & 0x2222222222222222ull) >> 1; // the shift is executed before the or operation!
					__sd[i][j].y ^= (__J[i][j].y & 0x2222222222222222ull) >> 1;

					// get right interaction and shift it to the right position
					__ct[i][j].x ^= (__J[i][j].x & 0x1111111111111111ull);
					__ct[i][j].y ^= (__J[i][j].y & 0x1111111111111111ull);


					// if me word at left boundary of sublattice - only j=0 should give true here but I still have to check for it's value.
					// only if black and even row possible left neighbor spin is boundary or if white and odd row left neighbor is boundary
					if((__j+j*BDIM_X)%(slX)==0){
						__sd[i][j].x &= 0x1111111111111110ull; // maps most right spin values in array to zero which should be interaction term for most left spin in lattice
					}

				} else {
					// __ct[][] holds "left" spins
					// __sd[][] holds "right" spins
					// get left interaction and shift it to the right position and perform XOR
					__ct[i][j].x ^= (__J[i][j].x & 0x2222222222222222ull) >> 1;
					__ct[i][j].y ^= (__J[i][j].y & 0x2222222222222222ull) >> 1;

					// get right interaction and perform XOR
					__sd[i][j].x ^= (__J[i][j].x & 0x1111111111111111ull);
					__sd[i][j].y ^= (__J[i][j].y & 0x1111111111111111ull);

					// only if black and odd row or white even row the right neighbor may be a sublattice boundary
					if((__j+j*BDIM_X+1)%(slX)==0){
						__sd[i][j].y &= 0x0111111111111111ull; // maps most left spin values in array to zero which should be interaction term for most right spin in lattice
					}
				}
			}
		}
	}

	curandStatePhilox4_32_10_t st;
	curand_init(seed, tid, static_cast<long long>(2*SPIN_X_WORD)*LOOP_X*LOOP_Y*(2*it+COLOR), &st);

	// Add binaries up but why though
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			// Check whether current row index on grid is multiple of sublattice dimension y. In this case the investigated word spins lay all at an upper edge of a sublattice.
			// Hence, only the dw, sd, ct boundaries shall be included in this case.
			if((__i+i*BDIM_Y)%slY==0){
					__dw[i][j].x += __sd[i][j].x;
					__ct[i][j].x += __dw[i][j].x;

					__dw[i][j].y += __sd[i][j].y;
					__ct[i][j].y += __dw[i][j].y;
			}
			// Check whether current row index on grid plus one is multiple of sublattice dimension y. In this case the investigated word spins lay all at an lower edge of a sublattice.
			// Hence, only the up, sd, ct boundaries shall be included in this case.
			else if((__i+i*BDIM_Y+1)%slY==0){
					__ct[i][j].x += __up[i][j].x;
					__ct[i][j].x += __sd[i][j].x;

					__ct[i][j].y += __up[i][j].y;
					__ct[i][j].y += __sd[i][j].y;
			}
			// For left an right boundaries was taken care of a step beforehand and thus one can sum over all neighbors here altough the word spins may include left or right boundaries of a sublattice.
			else{
				__ct[i][j].x += __up[i][j].x;
				__dw[i][j].x += __sd[i][j].x;
				__ct[i][j].x += __dw[i][j].x;

				__ct[i][j].y += __up[i][j].y;
				__dw[i][j].y += __sd[i][j].y;
				__ct[i][j].y += __dw[i][j].y;
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			#pragma unroll
			for(int z = 0; z < 8*sizeof(INT_T); z += BITXSP) {

				// do we need other exponential or random uniform distribution for boundary terms to be correct???

				//__src tuple, perform bitwise operation with 4 bits of __me and 1111
				// Extract information whether spin is up or down --> results in 0 or 1
				const int2 __src = make_int2((__me[i][j].x >> z) & 0xF,
							     (__me[i][j].y >> z) & 0xF);

				// __sum tuple, perform bitwise operation with 4 bits of __ct and 1111
				// Get number of up neighbors for each spin contained in the words --> results in range zero to 4
				const int2 __sum = make_int2((__ct[i][j].x >> z) & 0xF,
							     (__ct[i][j].y >> z) & 0xF);

				// Create unsigned long long 1
				const INT_T ONE = static_cast<INT_T>(1);

				// perform logical XOR on the bits containing the spins
				// updates the spins from -1 to 1 or vice versa
				if (curand_uniform(&st) <= __shExp[__src.x][__sum.x]) {
					__me[i][j].x ^= ONE << z;
				}
				if (curand_uniform(&st) <= __shExp[__src.y][__sum.y]) {
					__me[i][j].y ^= ONE << z;
				}
			}
		}
	}

	// Store updated spins in the lattice
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			vDst[(begY + __i+i*BDIM_Y)*dimX + __j+j*BDIM_X] = __me[i][j];
		}
	}
	return;
}

template<int BDIM_X,
	 int BDIM_Y,
	 int LOOP_X,
	 int LOOP_Y,
	 int BITXSP,
	 int COLOR,
	 typename INT_T,
	 typename INT2_T>
__global__
void spinUpdateV_2D_k(const int devid,
		      const long long seed,
		      const int it,
		      const int slX, // sublattice size X of one color (in words)
		      const int slY, // sublattice size Y of one color
		      const long long begY,
		      const long long dimX, // ld
		      const float vExp[][5],
		      const INT2_T *__restrict__ jDst,
		      const INT2_T *__restrict__ vSrc,
		            INT2_T *__restrict__ vDst) {

	// calc how many spins per word
	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	// x and y location in Thread lattice
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	// Initialize shared memory of Block size + neighbors
	__shared__ INT2_T shTile[BDIM_Y*LOOP_Y+2][BDIM_X*LOOP_X+2];

	// Load spin tiles of opposite lattice
	loadTile<BDIM_X, BDIM_Y,
		 BDIM_X*LOOP_X,
		 BDIM_Y*LOOP_Y,
		 1, 1, INT2_T>(slX, slY, begY, dimX, vSrc, shTile);

	// __shExp[cur_s{0,1}][sum_s{0,1}] = __expf(-2*cur_s{-1,+1}*F{+1,-1}(sum_s{0,1})*INV_TEMP)
	// Shared memory to store Exp
	__shared__ float __shExp[2][5];

	// for small lattices BDIM_X/Y may be smaller than 2/5
	// Load exponentials into shared memory
	#pragma unroll
	for(int i = 0; i < 2; i += BDIM_Y) {
		#pragma unroll
		for(int j = 0; j < 5; j += BDIM_X) {
			if (i+tidy < 2 && j+tidx < 5) {
				__shExp[i+tidy][j+tidx] = vExp[i+tidy][j+tidx];
			}
		}
	}
	__syncthreads();

	// get i and j location in block/thread grid
	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + tidy;
	const int __j = blockIdx.x*BDIM_X*LOOP_X + tidx;

	// calculate thread id
	const long long tid = ((devid*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x)*BDIM_X*BDIM_Y +
	                       threadIdx.y*BDIM_X + threadIdx.x;

	// array of size BMULT_Y x BMULT_X of unsigned long long
	INT2_T __me[LOOP_Y][LOOP_X];

	// Store spin words in array
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__me[i][j] = vDst[(begY+__i+i*BDIM_Y)*dimX + __j+j*BDIM_X];
		}
	}

	// initialize up, down center arrays
	INT2_T __up[LOOP_Y][LOOP_X];
	INT2_T __ct[LOOP_Y][LOOP_X];
	INT2_T __dw[LOOP_Y][LOOP_X];

	// Load up, down, center neighbors from other word lattice
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__up[i][j] = shTile[i*BDIM_Y +   tidy][j*BDIM_X + 1+tidx];
			__ct[i][j] = shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 1+tidx];
			__dw[i][j] = shTile[i*BDIM_Y + 2+tidy][j*BDIM_X + 1+tidx];
		}
	}

	// BDIM_Y is power of two so row parity won't change across loops
	// Check which color and whether row (__i) is even or odd
	// Example: black lattice, even row --> readBack = 1
	const int readBack = (COLOR == C_BLACK) ? !(__i%2) : (__i%2);

	// Load missing side neighbors
	INT2_T __sd[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__sd[i][j] = (readBack) ? shTile[i*BDIM_Y + 1+tidy][j*BDIM_X +   tidx]:
						  shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 2+tidx];
		}
	}

	// Rearrange left and right neighbors and update __sd[i,j] by filling it with the "right" neighbors
	// which become left neighbors
	if (readBack) {
		#pragma unroll
		// (BLACK LATTICE) Shift __sd such that it contains the left neighbors of the corresponding __me word
		// (BLACK LATTICE) __ct then contains the right neighbors
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__sd[i][j].x = (__ct[i][j].x << BITXSP) | (__sd[i][j].y >> (8*sizeof(__sd[i][j].y)-BITXSP));
				__sd[i][j].y = (__ct[i][j].y << BITXSP) | (__ct[i][j].x >> (8*sizeof(__ct[i][j].x)-BITXSP));
			}
		}
	} else {
		// (BLACK LATTICE) Shift __sd such that it contains the right neighbors of the corresponding __me word
		// (BLACK LATTICE) __ct then contains the left neighbors
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__sd[i][j].y = (__ct[i][j].y >> BITXSP) | (__sd[i][j].x << (8*sizeof(__sd[i][j].x)-BITXSP));
				__sd[i][j].x = (__ct[i][j].x >> BITXSP) | (__ct[i][j].y << (8*sizeof(__ct[i][j].y)-BITXSP));
			}
		}
	}

	// When Hamiltonian is used
	if (jDst != NULL) {
		// Initialize array of size (1,2) to store the interaction terms
		INT2_T __J[LOOP_Y][LOOP_X];

		// Load interactions for current word tuple we are in
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__J[i][j] = jDst[(begY+__i+i*BDIM_Y)*dimX + __j+j*BDIM_X];
			}
		}

		// apply them
		// the 4 bits of J codify: <upJ, downJ, leftJ, rightJ>
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {

				// Perform bitwise or operation
				// Column of left side gets the first bit in every group of four which is then shifted by 3 to the right left because spins are only
				// at the fourth location
				// XOR is then performed to change sign of spins
				__up[i][j].x ^= (__J[i][j].x & 0x8888888888888888ull) >> 3;
				__up[i][j].y ^= (__J[i][j].y & 0x8888888888888888ull) >> 3;

				// get down interaction and shift it to the right place
				__dw[i][j].x ^= (__J[i][j].x & 0x4444444444444444ull) >> 2;
				__dw[i][j].y ^= (__J[i][j].y & 0x4444444444444444ull) >> 2;

				if (readBack) {
					// __sd[][] holds "left" spins
					// __ct[][] holds "right" spins
					// get left interaction and shift it to the right position
					__sd[i][j].x ^= (__J[i][j].x & 0x2222222222222222ull) >> 1;
					__sd[i][j].y ^= (__J[i][j].y & 0x2222222222222222ull) >> 1;

					// get right interaction and shift it to the right position
					__ct[i][j].x ^= (__J[i][j].x & 0x1111111111111111ull);
					__ct[i][j].y ^= (__J[i][j].y & 0x1111111111111111ull);
				} else {
					// __ct[][] holds "left" spins
					// __sd[][] holds "right" spins
					// get left interaction and shift it to the right position and perform XOR
					__ct[i][j].x ^= (__J[i][j].x & 0x2222222222222222ull) >> 1;
					__ct[i][j].y ^= (__J[i][j].y & 0x2222222222222222ull) >> 1;

					// get right interaction and perform XOR
					__sd[i][j].x ^= (__J[i][j].x & 0x1111111111111111ull);
					__sd[i][j].y ^= (__J[i][j].y & 0x1111111111111111ull);
				}
			}
		}
	}

	curandStatePhilox4_32_10_t st;
	curand_init(seed, tid, static_cast<long long>(2*SPIN_X_WORD)*LOOP_X*LOOP_Y*(2*it+COLOR), &st);

	// Add binaries up but why though
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			// __ct contains at the end number of neighboring up spins in binary for the two words x and y
			__ct[i][j].x += __up[i][j].x;
			__dw[i][j].x += __sd[i][j].x;
			__ct[i][j].x += __dw[i][j].x;

			__ct[i][j].y += __up[i][j].y;
			__dw[i][j].y += __sd[i][j].y;
			__ct[i][j].y += __dw[i][j].y;
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			#pragma unroll
			for(int z = 0; z < 8*sizeof(INT_T); z += BITXSP) {

				//__src tuple, perform bitwise operation with 4 bits of __me and 1111
				// Extract information whether spin is up or down --> results in 0 or 1
				const int2 __src = make_int2((__me[i][j].x >> z) & 0xF,
							     (__me[i][j].y >> z) & 0xF);

				// __sum tuple, perform bitwise operation with 4 bits of __ct and 1111
				// Get number of up neighbors for each spin contained in the words --> results in range zero to 4
				const int2 __sum = make_int2((__ct[i][j].x >> z) & 0xF,
							     (__ct[i][j].y >> z) & 0xF);

				// Create unsigned long long 1
				const INT_T ONE = static_cast<INT_T>(1);

				// perform logical XOR on the bits containing the spins
				// updates the spins from -1 to 1 or vice versa
				if (curand_uniform(&st) <= __shExp[__src.x][__sum.x]) {
					__me[i][j].x ^= ONE << z;
				}
				if (curand_uniform(&st) <= __shExp[__src.y][__sum.y]) {
					__me[i][j].y ^= ONE << z;
				}
			}
		}
	}

	// Store updated spins in the lattice
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			vDst[(begY + __i+i*BDIM_Y)*dimX + __j+j*BDIM_X] = __me[i][j];
		}
	}
	return;
}

template<int BDIM_X,
	 int WSIZE,
	 typename T>
__device__ __forceinline__ T __block_sum(T v) {

	__shared__ T sh[BDIM_X/WSIZE];

	const int lid = threadIdx.x%WSIZE;
	const int wid = threadIdx.x/WSIZE;

	#pragma unroll
	for(int i = WSIZE/2; i; i >>= 1) {
		v += __shfl_down_sync(0xFFFFFFFF, v, i);
	}
	if (lid == 0) sh[wid] = v;

	__syncthreads();
	if (wid == 0) {
		v = (lid < (BDIM_X/WSIZE)) ? sh[lid] : 0;

		#pragma unroll
		for(int i = (BDIM_X/WSIZE)/2; i; i >>= 1) {
			v += __shfl_down_sync(0xFFFFFFFF, v, i);
		}
	}
	__syncthreads();
	return v;
}

// to be optimized
template<int BDIM_X,
	 int BITXSP,
         typename INT_T,
	 typename SUM_T>
__global__ void getMagn_k(const long long n, // llen
			  const INT_T *__restrict__ v, // black_d
			        SUM_T *__restrict__ sum) {

	// Get number of spins per word
	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	// nth = blockDim.x*gridDim.x???
	const long long nth = static_cast<long long>(blockDim.x)*gridDim.x;
	const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

	// counter for positive and negative
	SUM_T __cntP = 0;
	SUM_T __cntN = 0;

	// Loop over all entries until end of array is found
	for(long long i = 0; i < n; i += nth) {
		// Check if still in range
		if (i+tid < n) {
			// counts the number of non_zero bits in v[i+tid]
			// Add up correspondingly
			const int __c = __mypopc(v[i+tid]);
			__cntP += __c;
			__cntN += SPIN_X_WORD - __c;
		}
	}

	__cntP = __block_sum<BDIM_X, 32>(__cntP);
	__cntN = __block_sum<BDIM_X, 32>(__cntN);

	if (threadIdx.x == 0) {
		atomicAdd(sum+0, __cntP);
		atomicAdd(sum+1, __cntN);
	}
	return;
}

static void countSpins(const int ndev,
		       const int redBlocks,
		       const size_t llen,
		       const size_t llenLoc,
		       const unsigned long long *black_d,
		       const unsigned long long *white_d,
			     unsigned long long **sum_d,
			     unsigned long long *bsum,
			     unsigned long long *wsum) {

	if (ndev == 1) {
		CHECK_CUDA(cudaMemset(sum_d[0], 0, 2*sizeof(**sum_d)));
		getMagn_k<THREADS, BIT_X_SPIN><<<redBlocks, THREADS>>>(llen, black_d, sum_d[0]);
		CHECK_ERROR("getMagn_k");
		CHECK_CUDA(cudaDeviceSynchronize());
	}

	else {
		for(int i = 0; i < ndev; i++) {

			CHECK_CUDA(cudaSetDevice(i));
			CHECK_CUDA(cudaMemset(sum_d[i], 0, 2*sizeof(**sum_d)));
			getMagn_k<THREADS, BIT_X_SPIN><<<redBlocks, THREADS>>>(llenLoc, black_d + i*llenLoc, sum_d[i]);
			getMagn_k<THREADS, BIT_X_SPIN><<<redBlocks, THREADS>>>(llenLoc, white_d + i*llenLoc, sum_d[i]);
			CHECK_ERROR("getMagn_k");
		}
	}

	bsum[0] = 0;
	wsum[0] = 0;

	unsigned long long  sum_h[MAX_GPU][2];

	for(int i = 0; i < ndev; i++) {
		CHECK_CUDA(cudaMemcpy(sum_h[i], sum_d[i], 2*sizeof(**sum_h), cudaMemcpyDeviceToHost));
		bsum[0] += sum_h[i][0];
		wsum[0] += sum_h[i][1];
	}
	return;
}

static void dumpLattice(const char *fprefix,
			const int ndev,
			const int Y,
			const size_t lld,
		        const size_t llen,
		        const size_t llenLoc,
		        const unsigned long long *v_d) {

	char fname[256];

	if (ndev == 1) {
		unsigned long long *v_h = (unsigned long long *)Malloc(llen*sizeof(*v_h));
		CHECK_CUDA(cudaMemcpy(v_h, v_d, llen*sizeof(*v_h), cudaMemcpyDeviceToHost));

		unsigned long long *black_h = v_h;
		unsigned long long *white_h = v_h + llen/2;

		snprintf(fname, sizeof(fname), "%s0.txt", fprefix);
		FILE *fp = Fopen(fname, "w");

		for(int i = 0; i < Y; i++) {
			for(int j = 0; j < lld; j++) {
				unsigned long long __b = black_h[i*lld + j];
				unsigned long long __w = white_h[i*lld + j];

				for(int k = 0; k < 8*sizeof(*v_h); k += BIT_X_SPIN) {
					if (i&1) {
						fprintf(fp, "%llX",  (__w >> k) & 0xF);
						fprintf(fp, "%llX",  (__b >> k) & 0xF);
					} else {
						fprintf(fp, "%llX",  (__b >> k) & 0xF);
						fprintf(fp, "%llX",  (__w >> k) & 0xF);
					}
				}
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
		free(v_h);
	} else {
		#pragma omp parallel for schedule(static)
		for(int d = 0; d < ndev; d++) {
			const unsigned long long *black_h = v_d +          d*llenLoc;
			const unsigned long long *white_h = v_d + llen/2 + d*llenLoc;

			snprintf(fname, sizeof(fname), "%s%d.txt", fprefix, d);
			FILE *fp = Fopen(fname, "w");

			for(int i = 0; i < Y; i++) {
				for(int j = 0; j < lld; j++) {
					unsigned long long __b = black_h[i*lld + j];
					unsigned long long __w = white_h[i*lld + j];

					for(int k = 0; k < 8*sizeof(*black_h); k += BIT_X_SPIN) {
						if (i&1) {
							fprintf(fp, "%llX",  (__w >> k) & 0xF);
							fprintf(fp, "%llX",  (__b >> k) & 0xF);
						} else {
							fprintf(fp, "%llX",  (__b >> k) & 0xF);
							fprintf(fp, "%llX",  (__w >> k) & 0xF);
						}
					}
				}
				fprintf(fp, "\n");
			}
			fclose(fp);
		}
	}
	return;
}

template<int BDIM_X,
	 int BDIM_Y,
	 int LOOP_X,
	 int LOOP_Y,
	 int BITXSP,
	 typename INT_T,
	 typename INT2_T>
__global__ void calculate_average_magnetization(const int devid,
			const int slX,
			const int slY,
			const long long begY,
			const long long dimX,
			const INT2_T *__restrict__ v_white,
			const INT2_T *__restrict__ v_black,
			const thrust::complex<float> *exp,
			const int blocks_per_slx,
			const int blocks_per_sly,
			int *sum_per_block,
			thrust::complex<float> *c_sum_per_block) {

	// calc how many spins per word
	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	// x and y location in Thread lattice
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + tidy;

	// Initialize shared memory of Block size + neighbors
	__shared__ INT2_T shTile_w[BDIM_Y*LOOP_Y+2][BDIM_X*LOOP_X+2];
	__shared__ INT2_T shTile_b[BDIM_Y*LOOP_Y+2][BDIM_X*LOOP_X+2];

	// Store sum
	__shared__ int sum[BDIM_Y*BDIM_X];
	__shared__ thrust::complex<float> c_sum[BDIM_Y*BDIM_X];

	// Load spin tiles of lattice
	loadTile<BDIM_X, BDIM_Y,
		 BDIM_X*LOOP_X,
		 BDIM_Y*LOOP_Y,
		 1, 1, INT2_T>(slX, slY, begY, dimX, v_white, shTile_w);

	__syncthreads();

	loadTile<BDIM_X, BDIM_Y,
		 BDIM_X*LOOP_X,
		 BDIM_Y*LOOP_Y,
		 1, 1, INT2_T>(slX, slY, begY, dimX, v_black, shTile_b);

	__syncthreads();

	// array of size BMULT_Y x BMULT_X of unsigned long long
	INT2_T __me_w[LOOP_Y][LOOP_X];
	INT2_T __me_b[LOOP_Y][LOOP_X];

	// Store spin words in array
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__me_w[i][j] = shTile_w[1 + tidy + i*BDIM_Y][1 + tidx + j*BDIM_X];
			__me_b[i][j] = shTile_b[1 + tidy + i*BDIM_Y][1 + tidx + j*BDIM_X];
		}
	}

	int __cntP = 0;
	int __cntN = 0;

	thrust::complex<float> run_sum = thrust::complex<float>(0.0, 0.0);

	// Store spin words in array
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		const int __sli = (__i+i*BDIM_Y) % slY;
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			const int __cx = __popcll(__me_w[i][j].x) + __popcll(__me_b[i][j].x);
			const int __cy = __popcll(__me_w[i][j].y) + __popcll(__me_b[i][j].y);

			__cntP += (__cx + __cy);
			__cntN += 4*SPIN_X_WORD - __cx - __cy;

			int spin_sum = __cx + __cy - (4*SPIN_X_WORD - __cx - __cy);
			run_sum += exp[__sli]*spin_sum;
			//run_sum = cuCaddf(cuCmulf(exp[__sli], make_cuFloatComplex(spin_sum, 0)), run_sum);
		}
	}

	sum[tidy*BDIM_X + tidx] = __cntP - __cntN;
	c_sum[tidy*BDIM_X +tidx] = run_sum;

	__syncthreads();

	for (int s = blockDim.y*blockDim.x/2; s>0; s >>= 1){
		if (tidy*BDIM_X + tidx < s){
			sum[tidy*BDIM_X + tidx] += sum[tidy*BDIM_X + tidx + s];
			c_sum[tidy*BDIM_X + tidx] += c_sum[tidy*BDIM_X + tidx + s];
		}
		__syncthreads();
	}

	if ((tidx == 0) & (tidy == 0)){
		const int current_x = blockIdx.x/blocks_per_slx;
		const int current_y = blockIdx.y/blocks_per_sly;
		const int offset = (current_y*gridDim.x/blocks_per_slx + current_x)*blocks_per_slx*blocks_per_sly;

		const int block_lin_y = blockIdx.y%blocks_per_sly;
		const int block_lin_x = blockIdx.x%blocks_per_slx;

		sum_per_block[offset + block_lin_y*blocks_per_slx + block_lin_x] = sum[0];
		c_sum_per_block[offset + block_lin_y*blocks_per_slx + block_lin_x] = c_sum[0];
	}
}

__global__ void calculate_incremental_susceptibility(const int blocks_per_slx,
				const int blocks_per_sly,
				const int num_lattices,
				const int *d_sums_per_block,
				const thrust::complex<float> *d_weighted_sums_per_block,
				int *d_store_sum,
				float *d_sus_k){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = tid*blocks_per_slx*blocks_per_sly;

	if (tid >= num_lattices) return;

	int sum = 0;
	thrust::complex<float> c_sum = thrust::complex<float> (0.0f, 0.0f);

	for (int i=0; i < blocks_per_slx*blocks_per_sly; i++){
		sum += d_sums_per_block[offset + i];
		c_sum += d_weighted_sums_per_block[offset + i];
	}

	d_store_sum[tid] += pow(abs(sum), 2);
	d_sus_k[tid] += thrust::abs(c_sum)*thrust::abs(c_sum);
}


__global__ void incremental_susceptibility(int *d_store_sum,
			thrust::complex<float> *d_store_weighted_sum,
			int *d_inc_sus_0,
			float *d_inc_sus_k,
			int num_lattices){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= num_lattices) return;

	d_inc_sus_0[tid] += pow(abs(d_store_sum[tid]),2);
	d_inc_sus_k[tid] += thrust::abs(d_store_weighted_sum[tid])*thrust::abs(d_store_weighted_sum[tid]);
}
/*
cuComplex cuExp (cuComplex z){
	float exp_real = expf(z.x)*cosf(z.y);
	float exp_imag = expf(z.x)*sinf(z.y);

	return make_cuComplex(exp_real,exp_imag);
}
*/


int main(int argc, char **argv) {

	// v_d whole lattice
	// black_d black lattice --> white_d white lattice
	unsigned long long *v_d=NULL;
	unsigned long long *black_d=NULL;
	unsigned long long *white_d=NULL;

	// Interaction terms
	unsigned long long *ham_d=NULL;
	unsigned long long *hamB_d=NULL;
	unsigned long long *hamW_d=NULL;

	// Time related stuff
	cudaEvent_t start, stop;
    float et;

	// Number of spins per word
	// Bits per word / Bits per Spin
	// 16 spins per word (Theorie)
	const int SPIN_X_WORD = (8*sizeof(*v_d)) / BIT_X_SPIN;

	// Lattice sizes per GPU
	int X = 0;
	int Y = 0;

	// Write results to txt file
	int dumpOut = 0;

	// number of iterations to run
	int nsteps, nwarmup;

	// Random number seed
	unsigned long long seed = 42;

	// number of GPUs
	int ndev = 1;

	// Temperature in absolute units
	float temp  = -1.0f;

	// Probabilties that links connecting any two spins are anti-ferromagnetic
	// Probability for interactions
	int useGenHamilt = 1;
	float hamiltPerc1 = 0.0f;

	// Should we use sublattices or not
	int useSubLatt = 0;

	// Size of sublattices per GPU
	int XSL = 0;
	int YSL = 0;

	// number of sublattices along X and Y per GPU
	int NSLX = 1;
	int NSLY = 1;

	int och;
    while (1) {
        int option_index = 0;
        static struct option long_options[] = {
            {"X", required_argument, 0, 'x'},
            {"Y", required_argument, 0, 'y'},
			{"XSL", required_argument, 0, 1},
			{"YSL", required_argument, 0, 2},
			{"prob", required_argument, 0, 'p'},
			{"nw", required_argument, 0, 'w'},
            {"nit", required_argument, 0, 'n'},
			{"temp", required_argument, 0, 't'},
            {"ndev", required_argument, 0, 'd'},
			{"out", no_argument, 0, 'o'},
            {0, 0, 0, 0}
        };

        och = getopt_long(argc, argv, "x:y:p:w:n:t:do:", long_options, &option_index);
        if (och == -1)
            break;

        switch (och) {
			case   0:// handles long opts with non-NULL flag field
				break;
            case 'x':
				X = atoi(optarg);
                break;
            case 'y':
                Y = atoi(optarg);
                break;
            case 'p':
                hamiltPerc1 = atof(optarg);
                break;
			case 'w':
                nwarmup = atoi(optarg);
                break;
			case 'n':
                nsteps = atoi(optarg);
                break;
			case 't':
				temp = atoi(optarg);
				break;
			case 'd':
				ndev = atoi(optarg);
				break;
			case 'o':
				dumpOut = 1;
				break;
			case 1:
				useSubLatt = 1;
				XSL = atoi(optarg);
				break;
			case 2:
				useSubLatt = 1;
				YSL = atoi(optarg);
				break;
			case '?':
				exit(EXIT_FAILURE);

			default:
				fprintf(stderr, "unknown option: %c\n", och);
				exit(EXIT_FAILURE);
        }
    }

	// check if X or Y are zero
	if (!X || !Y) {
		// check if X is zero
		if (!X) {
			// if Y is not zero and ! Y % 2*S then set x=y
			// X is minimal size 2*SPIN_X_WORD ...
			if (Y && !(Y % (2*SPIN_X_WORD*2*BLOCK_X*BMULT_X))) {
				X = Y;
			}
			// else set X equal to this
			else {
				X = 2*SPIN_X_WORD*2*BLOCK_X*BMULT_X;
			}
		}
		// if Y is zero
		if (!Y) {
			// if x is divisable by BLOCK_Y*BMULT_Y, set Y=X
			if (!(X%(BLOCK_Y*BMULT_Y))) {
				Y = X;
			}
			// else set Y = BLOCK_Y*BMULT_Y
			else {
				Y = BLOCK_Y*BMULT_Y;
			}
		}
	}

	// Check input dimension of X
	if (!X || (X%2) || ((X/2)%(SPIN_X_WORD*2*BLOCK_X*BMULT_X))) {
		fprintf(stderr, "\nPlease specify an X dim multiple of %d\n\n", 2*SPIN_X_WORD*2*BLOCK_X*BMULT_X);
		exit(EXIT_FAILURE);
	}

	// Check input dimension of Y
	if (!Y || (Y%(BLOCK_Y*BMULT_Y))) {
		fprintf(stderr, "\nPlease specify a Y dim multiple of %d\n\n", BLOCK_Y*BMULT_Y);
		exit(EXIT_FAILURE);
	}

	// Check if we want to use sublattices
	if (useSubLatt) {
		// Same as above but for sublattice sizes
		if (!XSL || !YSL) {
			if (!XSL) {
				if (YSL && !(YSL % (2*SPIN_X_WORD*2*BLOCK_X*BMULT_X))) {
					XSL = YSL;
				} else {
					XSL = 2*SPIN_X_WORD*2*BLOCK_X*BMULT_X;
				}
			}
			if (!YSL) {
				if (!(XSL%(BLOCK_Y*BMULT_Y))) {
					YSL = XSL;
				} else {
					YSL = BLOCK_Y*BMULT_Y;
				}
			}
		}

		// X has to be multiple of XSL, XSL has to be even and != 0, XSL multiple of SPIN_X_WORD
		if ((X%XSL) || !XSL || (XSL%2) || ((XSL/2)%(SPIN_X_WORD*2*BLOCK_X*BMULT_X))) {
			fprintf(stderr,
				"\nPlease specify an X sub-lattice dim multiple of %d and divisor of %d\n\n",
				2*SPIN_X_WORD*2*BLOCK_X*BMULT_X, X);
			exit(EXIT_FAILURE);
		}
		// Y multiple of YSL, YSL != 0, Y multiple of Block_Y*..
		if ((Y%YSL) || !YSL || (YSL%(BLOCK_Y*BMULT_Y))) {
			fprintf(stderr,
				"\nPlease specify a Y sub-lattice dim multiple of %d divisor of %d\n\n",
				BLOCK_Y*BMULT_Y, Y);
			exit(EXIT_FAILURE);
		}

		// Set number of Sublattices per GPU
		NSLX = X / XSL;
		NSLY = Y / YSL;
	}

	// If no sublattice
	else {
		// XSL column size, YSL row size of all lattices over all GPUs
		XSL = X;
		YSL = Y*ndev;

		NSLX = 1;
		NSLY = 1;
	}

	// get GPU properties for each GPU
	cudaDeviceProp props;

	printf("\nUsing GPUs:\n");
	for(int i = 0; i < ndev; i++) {
		CHECK_CUDA(cudaGetDeviceProperties(&props, i));
		printf("\t%2d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
			i, props.name, props.multiProcessorCount,
			props.maxThreadsPerMultiProcessor,
			props.major, props.minor,
			props.ECCEnabled?"on":"off");
	}
	printf("\n");
	// we assums all gpus to be the same so we'll later
	// use the props filled for the last GPU...

	// If we use more than one gpu
	if (ndev > 1) {
		// Check if GPU allows concurrent managed memory access
		for(int i = 0; i < ndev; i++) {
			int attVal = 0;
			CHECK_CUDA(cudaDeviceGetAttribute(&attVal, cudaDevAttrConcurrentManagedAccess, i));
			if (!attVal) {
				fprintf(stderr,
					"error: device %d does not support concurrent managed memory access!\n", i);
				exit(EXIT_FAILURE);
			}
		}

		// print number of GPUs
		printf("GPUs direct access matrix:\n       ");
		for(int i = 0; i < ndev; i++) {
			printf("%4d", i);
		}

		int missingLinks = 0;
		printf("\n");
		for(int i = 0; i < ndev; i++) {
			printf("GPU %2d:", i);
			// Set index of GPUs
			CHECK_CUDA(cudaSetDevice(i));
			for(int j = 0; j < ndev; j++) {
				int access = 1;
				// Check if GPU i can access memory of GPU j
				if (i != j) {
					CHECK_CUDA(cudaDeviceCanAccessPeer(&access, i, j));
					// if access, then enable memory access for peer GPU
					if (access) {
						CHECK_CUDA(cudaDeviceEnablePeerAccess(j, 0));
					} else {
						missingLinks++;
					}
				}
				printf("%4c", access ? 'V' : 'X');
			}
			printf("\n");
		}
		printf("\n");
		// If missing Links, abort
		if (missingLinks) {
			fprintf(stderr,
				"error: %d direct memory links among devices missing\n",
				missingLinks);
			exit(EXIT_FAILURE);
		}
	}

	// Size of X-dimension of the word lattice per GPU
	size_t lld = (X/2)/SPIN_X_WORD;

	// length of a single color section per GPU (Word lattice)
	size_t llenLoc = static_cast<size_t>(Y)*lld;

	// total lattice length (all GPUs, all colors)
	size_t llen = 2ull*ndev*llenLoc;

	// Create blocks and Threads grid
	// lld/2 Tupel in Initialization?
	dim3 grid(DIV_UP(lld/2, BLOCK_X*BMULT_X),
		  DIV_UP(    Y, BLOCK_Y*BMULT_Y));

	// Creates a CUDA block with Block_X threads in the x dimension and block_Y threads in the y dimension
	dim3 block(BLOCK_X, BLOCK_Y);

	int blocks_per_slx = (XSL/2)/SPIN_X_WORD/2/(BLOCK_X*BMULT_X);
	int blocks_per_sly = YSL/(BLOCK_Y*BMULT_Y);

	// print stuff
	printf("Run configuration:\n");
	printf("\tspin/word: %d\n", SPIN_X_WORD);
	printf("\tspins: %zu\n", llen*SPIN_X_WORD);
	printf("\tseed: %llu\n", seed);
	printf("\titerations: %d\n", nsteps);
	printf("\tblock (X, Y): %d, %d\n", block.x, block.y);
	printf("\ttile  (X, Y): %d, %d\n", BLOCK_X*BMULT_X, BLOCK_Y*BMULT_Y);
	printf("\tgrid  (X, Y): %d, %d\n", grid.x, grid.y);

	printf("\tusing Hamiltonian buffer, setting links to -1 with prob %G\n", hamiltPerc1);

	printf("\n");
	if (useSubLatt) {
		printf("\tusing sub-lattices:\n");
		printf("\t\tno. of sub-lattices per GPU: %8d\n", NSLX*NSLY);
		printf("\t\tno. of sub-lattices (total): %8d\n", ndev*NSLX*NSLY);
		printf("\t\tsub-lattices size:           %7d x %7d\n\n", XSL, YSL);
	}
	printf("\tlocal lattice size:      %8d x %8d\n",      Y, X);
	printf("\ttotal lattice size:      %8d x %8d\n", ndev*Y, X);
	printf("\tlocal lattice shape: 2 x %8d x %8zu (%12zu %s)\n",      Y, lld, llenLoc*2, sizeof(*v_d) == 4 ? "uints" : "ulls");
	printf("\ttotal lattice shape: 2 x %8d x %8zu (%12zu %s)\n", ndev*Y, lld,      llen, sizeof(*v_d) == 4 ? "uints" : "ulls");
	printf("\tmemory: %.2lf MB (%.2lf MB per GPU)\n", (llen*sizeof(*v_d))/(1024.0*1024.0), llenLoc*2*sizeof(*v_d)/(1024.0*1024.0));

	// Maximum block number
	const int redBlocks = MIN(DIV_UP(llen, THREADS),
				  (props.maxThreadsPerMultiProcessor/THREADS)*props.multiProcessorCount);

	// How many spins are up/down
	unsigned long long cntPos;
	unsigned long long cntNeg;

	// pointer array of length MAX_GPU
	unsigned long long *sum_d[MAX_GPU];

	// if only one GPU
	if (ndev == 1) {
		//Allocate memory of size equal to whole lattice and set to 0
		CHECK_CUDA(cudaMalloc(&v_d, llen*sizeof(*v_d)));
		CHECK_CUDA(cudaMemset(v_d, 0, llen*sizeof(*v_d)));

		// allocate two unsigned long longs
		CHECK_CUDA(cudaMalloc(&sum_d[0], 2*sizeof(**sum_d)));

		// Allocate hamiltonian array and set to 0
		CHECK_CUDA(cudaMalloc(&ham_d, llen*sizeof(*ham_d)));
		CHECK_CUDA(cudaMemset(ham_d, 0, llen*sizeof(*ham_d)));

	// More than one GPU
	} else {
		// Allocate memory accessible by all GPUs
		CHECK_CUDA(cudaMallocManaged(&v_d, llen*sizeof(*v_d), cudaMemAttachGlobal));
		CHECK_CUDA(cudaMallocManaged(&ham_d, llen*sizeof(*ham_d), cudaMemAttachGlobal));

		printf("\nSetting up multi-gpu configuration:\n"); fflush(stdout);
		//#pragma omp parallel for schedule(static)

		// Loop over devices
		for(int i = 0; i < ndev; i++) {

			CHECK_CUDA(cudaSetDevice(i));

			// Allocate 2 elements for each entry in sum_d and set it to zero
			CHECK_CUDA(cudaMalloc(sum_d+i,     2*sizeof(**sum_d)));
        	CHECK_CUDA(cudaMemset(sum_d[i], 0, 2*sizeof(**sum_d)));

			// divide v_d into regions for black and white lattices
			CHECK_CUDA(cudaMemAdvise(v_d +            i*llenLoc, llenLoc*sizeof(*v_d), cudaMemAdviseSetPreferredLocation, i));
			CHECK_CUDA(cudaMemAdvise(v_d + (llen/2) + i*llenLoc, llenLoc*sizeof(*v_d), cudaMemAdviseSetPreferredLocation, i));

			//Same as above
			CHECK_CUDA(cudaMemAdvise(ham_d +            i*llenLoc, llenLoc*sizeof(*ham_d), cudaMemAdviseSetPreferredLocation, i));
			CHECK_CUDA(cudaMemAdvise(ham_d + (llen/2) + i*llenLoc, llenLoc*sizeof(*ham_d), cudaMemAdviseSetPreferredLocation, i));

			// black boundaries up/down
			CHECK_CUDA(cudaMemAdvise(v_d +            i*llenLoc,             lld*sizeof(*v_d), cudaMemAdviseSetAccessedBy, (i+ndev-1)%ndev));
			CHECK_CUDA(cudaMemAdvise(v_d +            i*llenLoc + (Y-1)*lld, lld*sizeof(*v_d), cudaMemAdviseSetAccessedBy, (i+ndev+1)%ndev));

			// white boundaries up/down
			CHECK_CUDA(cudaMemAdvise(v_d + (llen/2) + i*llenLoc,             lld*sizeof(*v_d), cudaMemAdviseSetAccessedBy, (i+ndev-1)%ndev));
			CHECK_CUDA(cudaMemAdvise(v_d + (llen/2) + i*llenLoc + (Y-1)*lld, lld*sizeof(*v_d), cudaMemAdviseSetAccessedBy, (i+ndev+1)%ndev));

			// Set black/white to all 0s
			CHECK_CUDA(cudaMemset(v_d +            i*llenLoc, 0, llenLoc*sizeof(*v_d)));
			CHECK_CUDA(cudaMemset(v_d + (llen/2) + i*llenLoc, 0, llenLoc*sizeof(*v_d)));

			CHECK_CUDA(cudaMemset(ham_d +            i*llenLoc, 0, llenLoc*sizeof(*ham_d)));
			CHECK_CUDA(cudaMemset(ham_d + (llen/2) + i*llenLoc, 0, llenLoc*sizeof(*ham_d)));

			printf("\tGPU %2d done\n", i); fflush(stdout);
		}
	}

	// Set pointer to start of black and white lattice
	black_d = v_d;
	white_d = v_d + llen/2;

	hamB_d = ham_d;
	hamW_d = ham_d + llen/2;

	// Declare two arrays
	float *exp_d[MAX_GPU];
	float  exp_h[2][5];

	// precompute possible exponentials
	// Iterate over all possible spin configurations
	// First loop over spin of interest, either 0 or 1
	// Second loop over all possible up/down configurations of the neighbors
	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 5; j++) {
			if(temp > 0) {
				exp_h[i][j] = expf((i?-2.0f:2.0f)*static_cast<float>(j*2-4)*(1.0f/temp));
			} else {
				if(j == 2) {
					exp_h[i][j] = 0.5f;
				} else {
					exp_h[i][j] = (i?-2.0f:2.0f)*static_cast<float>(j*2-4);
				}
			}
		}
	}

	// Copy exponentials to GPU
	for(int i = 0; i < ndev; i++) {
		CHECK_CUDA(cudaSetDevice(i));
		CHECK_CUDA(cudaMalloc(exp_d+i, 2*5*sizeof(**exp_d)));
		CHECK_CUDA(cudaMemcpy(exp_d[i], exp_h, 2*5*sizeof(**exp_d), cudaMemcpyHostToDevice));
	}

	// Calculate all exp used for weighted summation
	thrust::complex<float> *weighted_exp_d;
	thrust::complex<float> weighted_exp_h[YSL];
	thrust::complex<float> imag = thrust::complex<float>(0,1);

	for(int i = 0; i < YSL; i++) {
		weighted_exp_h[i] = exp(imag*2*M_PI/YSL*i);
	}

	// Copy exponentials to GPU
	for(int i = 0; i < ndev; i++) {
		CHECK_CUDA(cudaSetDevice(i));
		CHECK_CUDA(cudaMalloc(&weighted_exp_d, YSL*sizeof(*weighted_exp_d)));
		CHECK_CUDA(cudaMemcpy(weighted_exp_d, weighted_exp_h, YSL*sizeof(*weighted_exp_d), cudaMemcpyHostToDevice));
	}

	// Calculate sums
	int *d_sum_per_block;
	CHECK_CUDA(cudaMalloc(&d_sum_per_block, grid.x*grid.y*sizeof(*d_sum_per_block)));

	thrust::complex<float> *d_weighted_sum_per_blocks;
	CHECK_CUDA(cudaMalloc(&d_weighted_sum_per_blocks, grid.x*grid.y*sizeof(*d_weighted_sum_per_blocks)));

	int *d_sus_0;
	CHECK_CUDA(cudaMalloc(&d_sus_0, NSLX*NSLY*sizeof(*d_sus_0)));
	CHECK_CUDA(cudaMemset(d_sus_0, 0, NSLX*NSLY*sizeof(*d_sus_0)));

	float *d_sus_k;
	CHECK_CUDA(cudaMalloc(&d_sus_k, NSLX*NSLY*sizeof(*d_sus_k)));
    CHECK_CUDA(cudaMemset(d_sus_k, 0, NSLY*NSLX*sizeof(*d_sus_k)));

	/*
	int *d_store_sum;
	CHECK_CUDA(cudaMalloc(&d_store_sum, NSLX*NSLY*sizeof(*d_store_sum)));
	CHECK_CUDA(cudaMemset(d_store_sum, 0, NSLX*NSLY*sizeof(*d_store_sum)));

	thrust::complex<float> *d_store_weighted_sum;
	CHECK_CUDA(cudaMalloc(&d_store_weighted_sum, NSLX*NSLY*sizeof(*d_store_weighted_sum)));
	CHECK_CUDA(cudaMemset(d_store_weighted_sum, 0, NSLX*NSLY*sizeof(*d_store_weighted_sum)));

	int *d_inc_sus_0;
	CHECK_CUDA(cudaMalloc(&d_inc_sus_0, NSLX*NSLY*sizeof(*d_inc_sus_0)));
	CHECK_CUDA(cudaMemset(d_inc_sus_0, 0, NSLX*NSLY*sizeof(*d_inc_sus_0)));

	float *d_inc_sus_k;
	CHECK_CUDA(cudaMalloc(&d_inc_sus_k, NSLX*NSLY*sizeof(*d_inc_sus_k)));
	CHECK_CUDA(cudaMemset(d_inc_sus_k, 0, NSLX*NSLY*sizeof(*d_inc_sus_k)));
	*/

	// Start and Stop event
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	for(int i = 0; i < ndev; i++) {

		CHECK_CUDA(cudaSetDevice(i));

		// Initialize interaction terms
		hamiltInitB_k<BLOCK_X, BLOCK_Y,
			BMULT_X, BMULT_Y,
			BIT_X_SPIN,
			unsigned long long><<<grid, block>>>(i,
							hamiltPerc1,
							seed+1, // just use a different seed
							i*Y, lld/2,
							reinterpret_cast<ulonglong2 *>(hamB_d));

		hamiltInitW_k<BLOCK_X, BLOCK_Y,
					BMULT_X, BMULT_Y,
					BIT_X_SPIN,
					unsigned long long><<<grid, block>>>((XSL/2)/SPIN_X_WORD/2, YSL, i*Y, lld/2,
									reinterpret_cast<ulonglong2 *>(hamB_d),
									reinterpret_cast<ulonglong2 *>(hamW_d));

		// Init black lattice, lld/2 because of tuples
		latticeInit_k<BLOCK_X, BLOCK_Y,
			      BMULT_X, BMULT_Y,
			      BIT_X_SPIN, C_BLACK,
			      unsigned long long><<<grid, block>>>(i,
								   seed,
								   0, i*Y, lld/2,
								   reinterpret_cast<ulonglong2 *>(black_d));
		CHECK_ERROR("initLattice_k");

		// Init white lattice
		latticeInit_k<BLOCK_X, BLOCK_Y,
			      BMULT_X, BMULT_Y,
			      BIT_X_SPIN, C_WHITE,
			      unsigned long long><<<grid, block>>>(i,
								   seed,
								   0, i*Y, lld/2,
								   reinterpret_cast<ulonglong2 *>(white_d));
		CHECK_ERROR("initLattice_k");
	}


	calculate_average_magnetization<BLOCK_X, BLOCK_Y,
			BMULT_X, BMULT_Y,
			BIT_X_SPIN, unsigned long long><<<grid, block>>>(0,
						XSL, YSL, 0*Y, lld/2,
						reinterpret_cast<ulonglong2 *>(white_d),
						reinterpret_cast<ulonglong2 *>(black_d),
						weighted_exp_d,
						blocks_per_slx,
						blocks_per_sly,
						d_sum_per_block,
						d_weighted_sum_per_blocks);

	clock_t start_timing = clock();

	calculate_incremental_susceptibility<<<1, NSLX*NSLY>>>(blocks_per_slx, blocks_per_sly, NSLX*NSLY, d_sum_per_block, d_weighted_sum_per_blocks, d_sus_0, d_sus_k);

	/*
	for (int i=0; i < NSLX*NSLY; i++){

		if (temp_storage == 0){
			CHECK_CUDA(cub::DeviceReduce::Sum(d_temp, temp_storage, d_sum_per_block + i*blocks_per_slx*blocks_per_sly, &d_store_sum[i], blocks_per_slx*blocks_per_sly));
			CHECK_CUDA(cudaMalloc(&d_temp, temp_storage));
		}

		if (temp_storage_complex == 0){
			CHECK_CUDA(cub::DeviceReduce::Sum(d_temp_complex, temp_storage_complex, d_weighted_sum_per_blocks + i*blocks_per_slx*blocks_per_sly, &d_store_weighted_sum[i], blocks_per_slx*blocks_per_sly));
			CHECK_CUDA(cudaMalloc(&d_temp_complex, temp_storage_complex));
		}

		CHECK_CUDA(cub::DeviceReduce::Sum(d_temp, temp_storage, d_sum_per_block + i*blocks_per_slx*blocks_per_sly, &d_store_sum[i], blocks_per_slx*blocks_per_sly));
		CHECK_CUDA(cub::DeviceReduce::Sum(d_temp_complex, temp_storage_complex, d_weighted_sum_per_blocks + i*blocks_per_slx*blocks_per_sly, &d_store_weighted_sum[i], blocks_per_slx*blocks_per_sly));
	}

	incremental_susceptibility<<<1,NSLX*NSLY>>>(d_store_sum, d_store_weighted_sum, d_inc_sus_0, d_inc_sus_k, NSLX*NSLX);
	*/

	clock_t end_timing = clock();

	double duration = double(end_timing - start_timing) / CLOCKS_PER_SEC;
    std::cout << "Time taken: " << duration << " seconds" << std::endl;

	// Get sum for each sublattice
	int *h_sums_per_block = (int *)malloc(NSLX*NSLY*sizeof(int));
	float *h_weighted_sums_per_block = (float *)malloc(NSLX*NSLY*sizeof(float));

	//CHECK_CUDA(cudaMemcpy(h_sums_per_block, d_sus_0, NSLX*NSLY*sizeof(*d_sus_0), cudaMemcpyDeviceToHost));
	//CHECK_CUDA(cudaMemcpy(h_weighted_sums_per_block, d_sus_k, NSLX*NSLY*sizeof(*d_sus_k), cudaMemcpyDeviceToHost));

 	CHECK_CUDA(cudaMemcpy(h_sums_per_block, d_sus_0, NSLX*NSLY*sizeof(*d_sus_0), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(h_weighted_sums_per_block, d_sus_k, NSLX*NSLY*sizeof(*d_sus_k), cudaMemcpyDeviceToHost));

	for (int i = 0; i < NSLX*NSLY; i++){
		//cout << h_sums_per_block[i] << endl;
		printf("%f\n", h_weighted_sums_per_block[i]);
	}

	// Calculate sum of spins
	//countSpins(ndev, redBlocks, llen, llenLoc, black_d, white_d, sum_d, &cntPos, &cntNeg);

	/*
	printf("\nInitial magnetization: %9.6lf, up_s: %12llu, dw_s: %12llu\n",
	       abs(static_cast<double>(cntPos)-static_cast<double>(cntNeg)) / (llen*SPIN_X_WORD),
	       cntPos, cntNeg);
	*/
	/*
	// Device Synchronize
	for(int i = 0; i < ndev; i++) {
		CHECK_CUDA(cudaSetDevice(i));
		CHECK_CUDA(cudaDeviceSynchronize());
	}

	// Timing
	double __t0;
	if (ndev == 1) {
		CHECK_CUDA(cudaEventRecord(start, 0));
	} else {
		__t0 = Wtime();
	}

	printf("\nPerfom %d Monte Carlo warm up steps \n", nwarmup);
	// Perform Monte Carlo warm-up
	for(int j = 0; j < nwarmup; j++) {
		for(int i = 0; i < ndev; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			// Update black lattice
			spinUpdateV_2D_k<BLOCK_X, BLOCK_Y,
					 BMULT_X, BMULT_Y,
					 BIT_X_SPIN, C_BLACK,
					 unsigned long long><<<grid, block>>>(i,
							 		      seed,
									      j+1,
									      (XSL/2)/SPIN_X_WORD/2, YSL,
									      i*Y,  lld/2,
							 		      reinterpret_cast<float (*)[5]>(exp_d[i]),
									      reinterpret_cast<ulonglong2 *>(hamW_d),
									      reinterpret_cast<ulonglong2 *>(white_d),
									      reinterpret_cast<ulonglong2 *>(black_d));
		}

		// Device Synchronize
		if (ndev > 1) {
			for(int i = 0; i < ndev; i++) {
				CHECK_CUDA(cudaSetDevice(i));
				CHECK_CUDA(cudaDeviceSynchronize());
			}
		}

		// Update white lattice
		for(int i = 0; i < ndev; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			spinUpdateV_2D_k<BLOCK_X, BLOCK_Y,
					 BMULT_X, BMULT_Y,
					 BIT_X_SPIN, C_WHITE,
					 unsigned long long><<<grid, block>>>(i,
							 		      seed,
									      j+1,
									      (XSL/2)/SPIN_X_WORD/2, YSL,
									      i*Y, lld/2,
							 		      reinterpret_cast<float (*)[5]>(exp_d[i]),
									      reinterpret_cast<ulonglong2 *>(hamB_d),
									      reinterpret_cast<ulonglong2 *>(black_d),
									      reinterpret_cast<ulonglong2 *>(white_d));
		}

		// Cuda device Synchronize
		if (ndev > 1) {
			for(int i = 0; i < ndev; i++) {
				CHECK_CUDA(cudaSetDevice(i));
				CHECK_CUDA(cudaDeviceSynchronize());
			}
		}
	}

	printf("\nPerfom %d Monte Carlo steps \n", nsteps);

	// Perform Monte Carlo updates
	for(int j = 0; j < nsteps; j++) {

		for(int i = 0; i < ndev; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			// Update black lattice
			spinUpdateV_2D_k<BLOCK_X, BLOCK_Y,
					 BMULT_X, BMULT_Y,
					 BIT_X_SPIN, C_BLACK,
					 unsigned long long><<<grid, block>>>(i,
							 		      seed,
									      j+1,
									      (XSL/2)/SPIN_X_WORD/2, YSL,
									      i*Y, lld/2,
							 		      reinterpret_cast<float (*)[5]>(exp_d[i]),
									      reinterpret_cast<ulonglong2 *>(hamW_d),
									      reinterpret_cast<ulonglong2 *>(white_d),
									      reinterpret_cast<ulonglong2 *>(black_d));
		}

		// Device Synchronize
		if (ndev > 1) {
			for(int i = 0; i < ndev; i++) {
				CHECK_CUDA(cudaSetDevice(i));
				CHECK_CUDA(cudaDeviceSynchronize());
			}
		}

		// Update white lattice
		for(int i = 0; i < ndev; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			spinUpdateV_2D_k<BLOCK_X, BLOCK_Y,
					 BMULT_X, BMULT_Y,
					 BIT_X_SPIN, C_WHITE,
					 unsigned long long><<<grid, block>>>(i,
							 		      seed,
									      j+1,
									      (XSL/2)/SPIN_X_WORD/2, YSL,
									      i*Y, lld/2,
							 		      reinterpret_cast<float (*)[5]>(exp_d[i]),
									      reinterpret_cast<ulonglong2 *>(hamB_d),
									      reinterpret_cast<ulonglong2 *>(black_d),
									      reinterpret_cast<ulonglong2 *>(white_d));
		}

		// Cuda device Synchronize
		if (ndev > 1) {
			for(int i = 0; i < ndev; i++) {
				CHECK_CUDA(cudaSetDevice(i));
				CHECK_CUDA(cudaDeviceSynchronize());
			}
		}
	}

	// Finish update steps
	if (ndev == 1) {
		CHECK_CUDA(cudaEventRecord(stop, 0));
		CHECK_CUDA(cudaEventSynchronize(stop));
	}
	else {
		for(int i = 0; i < ndev; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			CHECK_CUDA(cudaDeviceSynchronize());
		}
		__t0 = Wtime()-__t0;
	}

	// Calculate final magnetization
	countSpins(ndev, redBlocks, llen, llenLoc, black_d, white_d, sum_d, &cntPos, &cntNeg);
	printf("\nFinal   magnetization: %9.6lf, up_s: %12llu, dw_s: %12llu (iter: %8d)\n\n",
	       abs(static_cast<double>(cntPos)-static_cast<double>(cntNeg)) / (llen*SPIN_X_WORD),
	       cntPos, cntNeg, nwarmup + nsteps);


	if (ndev == 1) {
		CHECK_CUDA(cudaEventElapsedTime(&et, start, stop));
	} else {
		et = __t0*1.0E+3;
	}


	printf("Kernel execution time for %d update steps: %E ms, %.2lf flips/ns (BW: %.2lf GB/s)\n",
		nsteps+nwarmup, et, static_cast<double>(llen*SPIN_X_WORD)*(nsteps+nwarmup) / (et*1.0E+6),
		//(llen*sizeof(*v_d)*2*j/1.0E+9) / (et/1.0E+3));
		(2ull*(nsteps+nwarmup)*
		 	( sizeof(*v_d)*((llen/2) + (llen/2) + (llen/2)) + // src color read, dst color read, dst color write
			  sizeof(*exp_d)*5*grid.x*grid.y ) /
		1.0E+9) / (et/1.0E+3));
	*/

	// Write lattice
	if (dumpOut) {
		char fname[256];
		snprintf(fname, sizeof(fname), "lattice_%dx%d_T_%f_IT_%08d_", Y, X, temp, nsteps + nwarmup);
		dumpLattice(fname, ndev, Y, lld, llen, llenLoc, v_d);
	}

	// free memory for all GPUs and stuff
	CHECK_CUDA(cudaFree(v_d));
	if (useGenHamilt) {
		CHECK_CUDA(cudaFree(ham_d));
	}
	if (ndev == 1) {
		CHECK_CUDA(cudaFree(exp_d[0]));
		CHECK_CUDA(cudaFree(sum_d[0]));
	} else {
		for(int i = 0; i < ndev; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			CHECK_CUDA(cudaFree(exp_d[i]));
			CHECK_CUDA(cudaFree(sum_d[i]));
		}
	}
	for(int i = 0; i < ndev; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaDeviceReset());
    }

	return 0;
}
