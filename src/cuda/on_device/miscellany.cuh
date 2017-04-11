#pragma once
#ifndef CUDA_ON_DEVICE_MISCELLANY_CUH_
#define CUDA_ON_DEVICE_MISCELLANY_CUH_

/* Utilities for on-device code, e.g. getting global thread index etc. */

#ifndef forceinline
#define forceinline __forceinline__
#endif

#include <cuda_runtime.h>
// Though this is an include file, it should not be visible outside
// the op store's implementation
#include "cuda/syntax_replacement.h"
#include "cuda/api/types.h"
#include "cuda/api/constants.h"

#include "cuda/on_device/grid_info.cuh"

#include <type_traits>

/**
 * Use this type when walking index variables collaboratively among
 * multiple threads along some array. The reason you can't just use
 * the original size type is twofold:
 *
 * 1. Occasionally you might have
 *
 *      pos += blockDim.x * blockDim.y * blockDim.z;
 *
 *    e.g. when you work at block stride on a linear input. Well,
 *    if the type of pos is not large enough (e.g. char) - you would
 *    get into infinite loops.
 *
 * 2. The native integer type on a GPU is 32-bit - at least on CUDA;
 *    so there's not much sense of keeping an index variable in some
 *    smaller type. At best, the compiler will switch it to a 32-bit
 *    value; at worst, you'll waste time putting it into and out of
 *    32-bit variables
 */
template <typename Size>
using promoted_size = typename std::common_type<Size,unsigned>::type;

/**
 * TODO:
 * - There should be some slimmed-down version of the standard template library for
 *   device-side code.
 */
template <typename T>
forceinline __device__ void swap(T& x, T& y) {
   T _x = x;
   T _y = y;
   x = _y;
   y = _x;
}

/*
 ************************************************************************
 * C-standard-library-like functions available on the device
 ************************************************************************
 */

// A single-thread inefficient version of strcmp
forceinline __device__ int strcmp_on_device(const char* src, const char* dst)
{
	int result = 0;

	while (!(result = *(unsigned char *) src - *(unsigned char *) dst) && *dst)
		++src, ++dst;

	if (result < 0)
		result = -1;
	else if (result > 0) result = 1;

	return result;
}

// A single-thread inefficient version of strcmp
forceinline __device__ char* strcpy_on_device(char *dst, const char *src)
{
	while (*dst != '\0') {
		*(dst++) = *(src++);
	}
	*dst = *src;
	return dst;
}

//
//
//
/**
 * Copies some data from one location to antoher - using the native register
 * size for individual elements on CUDA GPUs, i.e. sizeof(int) = 4
 *
 * @todo This is untested.
 *
 * @note unlike memcpy, the size of copied data is num_elements_to_copy times the
 * element size, i.e. 4*num_elements_to_copy.
 *
 * @note CUDA's own general-purpose memcpy() takes void pointers and uses a byte
 *  LD-ST loop; this  LD-ST's the native register size, 4 bytes.
 *
 * @note this function entirely ignores alignment.
 *
 * @param destination Destination of the copy. Must have at least
 * 4 * {@ref num_elements_to_copy} bytes allocated.
 * @param source The beginning of the memory region from which to copy.
 * There must be 4 * {@ref num_elements_to_copy} bytes readable starting with
 * this address.
 * @param num_elements_to_copy the number of int-sized elements of data to copy
 * (regardless of whether the data really consists of integers
 * @return the destination pointer
 */
template <typename T = unsigned int>
forceinline __device__
std::enable_if<sizeof(T) == sizeof(int), T> * copy(
	T*        __restrict__  destination,
	const T*  __restrict__  source,
	size_t                  num_elements_to_copy)
{
	while (num_elements_to_copy-- > 0) {
		*(destination++) = *(source++);
	}
	return destination;
}

/**
 * A replacement for CUDA's iteration-per-byte-loop memcpy()
 * implemntation: This accounts for alignment (although not of
 * incompatible alignment between source and thread); and uses
 * register-sized quanta for copying as much as possible.
 *
 * @note not tested
 *
 * @param destination A location to which to copy data.
 * @param source location from which to copy data
 * @param size number of bytes to copy
 */
template <typename T>
forceinline __device__
void* long_memcpy(
	void*        __restrict__  destination,
	const void*  __restrict__  source,
	size_t                     size)
{
	static_assert(sizeof(int) == 4, "Expecting sizeof(int) to be 4");
	static_assert(sizeof(void*) == sizeof(unsigned long long), "Expecting pointers to be unsigned long long");

	auto misalignment = (unsigned long long) destination % 4;
	// Assumes the source and destination have the same alignment modulo sizeof(int) = 4
	switch((unsigned long long) destination % 4) {
	case 1: *((char* )destination) = *((const char* )source); break;
	case 2: *((short*)destination) = *((const short*)source); break;
	case 3: *((char3*)destination) = *((const char3*)source); break;
	}
	size -= misalignment;

	// These int pointers will be aligned at sizeof(int)
	int*       destination_int = ((int*)       ((char*)      destination - misalignment)) + 1;
	const int* source_int      = ((const int*) ((const char*)source      - misalignment)) + 1;
	auto num_ints_to_copy = size >> 4;
	while (num_ints_to_copy-- > 0) {
		*(destination_int++) = *(source_int++);
	}
	switch(size % 4) {
	case 1: *((char* )destination_int) = *((const char* )source_int); break;
	case 2: *((short*)destination_int) = *((const short*)source_int); break;
	case 3: *((char3*)destination_int) = *((const char3*)source_int); break;
	}
	return destination;
}

/**
 * This gadget is necessary for using dynamically-sized shared memory in
 * templated kernels (i.e. shared memory whose size is set by the launch
 * parameters rather than being fixed. That requires a __shared__ extern
 * variable, but you can't reuse it - which is exactly what the templated
 * code would try to do. So you need a different extern for every type.
 *
 * @note all threads would get the same address when calling this function,
 * so you would need to add different offsets for different threads if
 * you want a warp-specific or thread-specific pointer.
 */
template <typename T>
__device__ T* shared_memory_proxy()
{
	extern __shared__ char memory[];
	return reinterpret_cast<T*>(memory);
}

/**
 * Busy-sleep for (at least) a specified number of GPU cycles.
 *
 * @note GPU clock speed routinely varies significantly, and
 * scheduling is quite hard to predict, so do not expect
 * great accuracy w.r.t. the amount of _time_ this make your
 * warps sleep.
 *
 * @param[in] clock_cycle_ count minimum (and desired) number of
 * clock cycles to sleep.
 * @return the number of clock cycles actually spent busy-sleeping
 * (as well as can be measured)
 */
inline __device__ clock_t busy_sleep(clock_t clock_cycle_count)
{
	clock_t start_clock = clock();
	clock_t clock_offset = 0;
	while (clock_offset < clock_cycle_count)
	{
		clock_offset = clock() - start_clock;
	}
	return clock_offset;
}

forceinline __device__
void* memzero(void* destination, size_t size) { return memset(destination, 0, size); }

/**
 * Some kernels operate on sequences of variable-length data, either
 * at the bit or byte levels. Even if they can determine the lengths
 * of individual elements, locating the element relevant to them
 * would require essentially serial execution, which we wish to avoid.
 * Thus they require some assistance, in the form of the pre-computed
 * positions of elements at fixed intervals. The following is a simple
 * structure for passing those around.
 *
 * @note This cannot be replaced by a span, as that would be the
 * anchors and the number of anchors - which kernels do not need passed
 * to them.
 */
template <typename SourceSize, typename TargetSize, bool UniformPeriod = false>
struct position_anchoring_t;

// TODO: Consider having the same anchor_t as the non-uniform period,
// and just generating it on demand
template <typename SourceSize, typename TargetSize>
struct position_anchoring_t<SourceSize, TargetSize, true> {
	struct anchor_t {
		TargetSize  target_position;
	};
	anchor_t*   anchors; // implicitly, the source position of an anchor is period times its index
	SourceSize  period;
	__host__ __device__
	anchor_t& operator[](SourceSize i) { return anchors[i]; }
};

template <typename SourceSize, typename TargetSize>
using uniform_position_anchoring_t = position_anchoring_t<SourceSize, TargetSize, true>;

template <typename SourceSize, typename TargetSize>
struct position_anchoring_t <SourceSize, TargetSize, false> {
	struct anchor_t {
		SourceSize  source_position;
		TargetSize  target_position;
	};
	anchor_t* anchors;
	__host__ __device__
	anchor_t& operator[](SourceSize i) { return anchors[i]; }
};

template <typename T>
constexpr __device__ __forceinline__
typename std::enable_if<std::is_integral<T>::value, T>::type all_one_bits()
{
	return ~((T)0);
}

namespace detail {

template <typename I>
__forceinline__ __device__ void ensure_index_type_is_valid()
{
	static_assert(
		std::is_same<I, unsigned char >::value or
		std::is_same<I, unsigned short>::value or
		std::is_same<I, unsigned int  >::value or
		std::is_same<I, unsigned long >::value,
		"Invalid index type");
}

template <typename T>
constexpr __device__ bool has_nice_simple_size()
{
	return
		sizeof(T) == 1 or sizeof(T) == 2 or
		sizeof(T) == 4 or sizeof(T) == 8;
}

}

#endif /* CUDA_ON_DEVICE_MISCELLANY_CUH_ */
