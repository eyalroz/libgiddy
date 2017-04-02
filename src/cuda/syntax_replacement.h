#pragma once

#ifdef __CDT_PARSER__
#define USE_SYNTAX_REPLACEMENT 1
#endif

#if USE_SYNTAX_REPLACEMENT == 1
#ifndef CUDA_SYNTAX_REPLACEMENT_H_
#define CUDA_SYNTAX_REPLACEMENT_H_

/**
 * The code in this file is protected by ifdefs so that it is used only when it's
 * included by an IDE's code parser (e.g. Eclipse' CDT, which defines __CDT_PARSER__).
 * Including this file will prevent IDEs from complaining about constructs and defines
 * which nvcc supports but are not available automatically (or at all) with a mere
 * C++ compiler; to put it simply: it prevents all of your code from lighting up as
 * having innumerable syntax errors.
 * 
 * The only exception to the above is the angle-bracket launch syntax for launching
 * CUDA kernels, i.e. some_kernel<<<launch_params>>>(arg1, arg2);  - but this can be
 * avoided as well by using kernel_launch_wrapper.cuh . So, syntax error days are over!
 *
 * TODO: Maybe we could utilize CUDA's include fiels more, e.g. vector_types.h and
 * the like.
 */

#define forceinline

// TODO: Split these off into a separate file, usable for __host__ __device__ functions
#define __global__
#define __shared__
#define __device__
#define __mapped__
#define __host__
#define __device_builtin__
#define __forceinline__

#define __STORAGE__


#define __ffs(x) 1
#define __ffsll(x) 1
#define __brev(x) 1
#define __brevll(x) 1

#ifndef IMPLEMENTING_SHUFFLE_OVERRIDES
#define __shfl_xor(x,y) x
#define __shfl_down(x,y) x
#define __shfl_up(x,y) x
#endif

#ifndef IMPLEMENTING_LDG_OVERRIDES
#define __ldg(x) 0
#endif

#ifndef __VECTOR_TYPES_H__
struct __device_builtin__ uint3
{
	unsigned int x, y, z;
};

typedef struct uint3 uint3;

struct __device_builtin__ dim3 {
	unsigned int x, y, z;

	dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
	dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
	operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
};

typedef struct dim3 dim3;
#endif /* __VECTOR_TYPES_H__ */

extern struct uint3 threadIdx, blockIdx;
extern struct dim3 blockDim, gridDim;
extern int warpSize;

void __syncthreads();
void __threadfence();
void __threadfence_block();
void __threadfence_system();

#define __ballot(...) 0u
#define __any(...) 0u
#define __all(...) 0u
#define __popcll(x) 0
#define __popc(x) 0

#endif /* CUDA_SYNTAX_REPLACEMENT_H_ */
#endif /* USE_SYNTAX_REPLACEMENT == 1 */
