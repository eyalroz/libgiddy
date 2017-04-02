#pragma once
#ifndef SRC_CUDA_PRINTING_CUH_
#define SRC_CUDA_PRINTING_CUH_

// Necessary for printf()'ing in kernel code
#include <cstdio>
#include "cuda/on_device/miscellany.cuh"
#include "cuda/on_device/printing.cuh" // deleteme
#include "cuda/api/device_function.hpp"

#ifdef PRINT_NOTHING
#define tprintf(format_str, ... )
#define tprint(str)
#define printf_once(format_str, ... )
#define print_once(str)
#define block_printf(format_str, ... )
#define block_print(str)
#define warp_printf(format_str, ... )
#define warp_print(str)
#define grid_printf(format_str, ... )
#define grid_print(str)
#else
/**
 * A wrapper for the printf function, which prefixes the printed string with
 * the thread's full identification: The In-block thread index, and a
 * block-warp-lane index triplet.
 *
 * Note that only a single printf() call is made for both the prefix id info and
 * the macros' arguments
 *
 * @param format_str the format_string to pass on to printf, for printing after the thread identification
 * @param __VA__ARGS the data to be pluggined into the format string instead of the % tags
 * @return same as printf()
 */
#define thread_printf(format_str, ... )  \
	printf("T %0*llu = (%0*u,%02u,%02u): " format_str "\n", \
		::max(2u,num_digits_required(::grid_info::linear::grid::num_threads() - 1llu)), \
		::grid_info::linear::thread::global_index(), \
		::max(2u,num_digits_required(::grid_info::linear::grid::num_blocks() - 1llu)), \
		blockIdx.x, \
		::grid_info::linear::warp::index(), grid_info::linear::lane::index(), __VA_ARGS__);
#define thread_print(str)  \
	printf("T %0*llu = (%0*u,%02u,%02u): %s\n", \
		::max(2,num_digits_required(::grid_info::linear::grid::num_threads() - 1llu)), \
		::grid_info::linear::thread::global_index(), \
		::max(2,num_digits_required(::grid_info::linear::grid::num_blocks() - 1llu)), \
		blockIdx.x, \
		::grid_info::linear::warp::index(), grid_info::linear::lane::index(), str);
#define tprintf thread_printf
#define tprint thread_print

#define warp_printf(format_str, ... )  \
	do{ \
		if (::grid_info::linear::lane::is_first()) \
			printf("W %0*llu = (%0*u,%02u): " format_str "\n", \
				::max(2u,num_digits_required(::grid_info::linear::grid::num_warps() - 1)), \
				::grid_info::linear::warp::global_index(), \
				::max(2u,num_digits_required(::grid_info::linear::grid::num_blocks() - 1)), \
				blockIdx.x, \
				::grid_info::linear::warp::index(), __VA_ARGS__); \
	} while(0)
#define warp_print(str)  \
	do{ \
		if (::grid_info::linear::lane::is_first()) \
		printf("W %0*llu = (%0*u,%02u): %s\n", \
			::max(2,num_digits_required(::grid_info::linear::grid::num_warps() - 1)), \
			::grid_info::linear::warp::global_index(), \
			::max(2,num_digits_required(::grid_info::linear::grid::num_blocks() - 1)), \
			blockIdx.x, \
			::grid_info::linear::warp::index(), str); \
	} while(0)

#define block_printf(format_str, ... )  \
	do{ \
		if (::grid_info::linear::thread::is_first_in_block()) \
			printf("B %0*u: " format_str "\n", \
				::max(2u,num_digits_required(::grid_info::linear::grid::num_blocks() - 1)), \
				::grid_info::linear::block::index(), __VA_ARGS__); \
	} while(0)
#define block_print(str)  \
	do{ \
		if (::grid_info::linear::thread::is_first_in_block()) \
		printf("B %0*u: %s\n", \
			::max(2u,num_digits_required(::grid_info::linear::grid::num_blocks() - 1)), \
				::grid_info::linear::block::index(), str); \
	} while(0)
#define bprintf block_printf
#define bprint block_print

#define printf_once(format_str, ... )  \
	do { \
		if (::grid_info::linear::thread::global_index() == 0) { \
		    printf("G " format_str "\n", __VA_ARGS__); \
		} \
	} while (false)
#define print_once(str)  \
	do { \
		if (::grid_info::linear::thread::global_index() == 0) { \
		    printf("G %s\n", str); \
		} \
	} while (false)

#define grid_printf  printf_once
#define grid_print   print_once
#define gprintf grid_printf
#define gprint grid_print

#endif /* PRINT_NOTHING */

// Identification is 0-based!
inline __device__ void identify_1d()
{
	printf("Thread %07d - %05d within block %05d - lane %02d of in-block warp %02d)\n",
		::grid_info::linear::thread::global_index(), threadIdx.x, blockIdx.x, threadIdx.x % warpSize, threadIdx.x / warpSize);

}

#define IDENTIFY_FUNCTION() { \
	printf_once("Now executing function \"%s\"", __PRETTY_FUNCTION__); \
	__syncthreads(); \
}

inline __device__ void maybe_identify_function(const char* identifier __attribute__((unused)))
{
#ifdef IDENTIFY_FUNCTION_ON_CUDA_DEVICE
	if (::grid_info::linear::thread::global_index() == 0) {
		printf("Now executing function \"%s\"\n", identifier);
	}
	__syncthreads();
#endif
}


inline __device__ unsigned num_digits_required(unsigned long long extremal_value)
{
	return ceilf(log10f(extremal_value));
}

template <typename T>
inline __device__ unsigned get_bit(T x, unsigned bit_index) { return 0x1 & (x >> bit_index); }

template <typename T>
inline __device__ const char* binary_representation (T x, char *target_buffer, unsigned field_width = 0) {

	unsigned num_bits = (x == 0) ? 0 : ilog_2(x) + 1;

	unsigned leading_zeros = max(field_width - num_bits, 0);
	#pragma unroll
	for(unsigned bit_index = 0; bit_index < leading_zeros; bit_index++) {
		target_buffer[bit_index] = '0';
	}

	#pragma unroll
	for(unsigned bit_index = 0; bit_index < num_bits; bit_index++) {
		target_buffer[leading_zeros + bit_index] = ((x & (1 << bit_index)) == 0) ? '0' : '1';
	}
	target_buffer[leading_zeros + num_bits] = '\0';
	return target_buffer;
}

// This function is "thread-safe" in the sense that execution
// by a each individual thread is in itself serial, so nothing
// can overwrite this array for a thread
template <typename T>
inline __device__ const char *binary_representation (T val, unsigned field_width = 0) {
	char thread_local_buffer[sizeof(T) * CHAR_BIT];
	return binary_representation(val, thread_local_buffer, field_width);
}

inline __device__ const char* true_or_false(bool x)
{
	return x ? "true" : "false";
}
inline __device__ const char* yes_or_no(bool x)
{
	return x ? "yes" : "no";
}

#define BIT_PRINTING_PATTERN_4BIT "%04u"
#define BIT_PRINTING_ARGUMENTS_4BIT(x)\
	(unsigned) get_bit((x),  0) * 1000 + \
	(unsigned) get_bit((x),  1) * 100 +  \
	(unsigned) get_bit((x),  2) * 10 +   \
	(unsigned) get_bit((x),  3) * 1

#define DOUBLE_LENGTH_PATTERN(single_length_pattern) single_length_pattern "'" single_length_pattern
#define DOUBLE_LENGTH_ARGUMENTS(x, single_length, single_length_arguments) \
	single_length_arguments(x), single_length_arguments(x >> single_length)

#define BIT_PRINTING_PATTERN_8BIT    DOUBLE_LENGTH_PATTERN(BIT_PRINTING_PATTERN_4BIT)
#define BIT_PRINTING_PATTERN_16BIT   DOUBLE_LENGTH_PATTERN(BIT_PRINTING_PATTERN_8BIT)
#define BIT_PRINTING_PATTERN_32BIT   DOUBLE_LENGTH_PATTERN(BIT_PRINTING_PATTERN_16BIT)
#define BIT_PRINTING_PATTERN_64BIT   DOUBLE_LENGTH_PATTERN(BIT_PRINTING_PATTERN_32BIT)

#define BIT_PRINTING_ARGUMENTS_8BIT(x)  DOUBLE_LENGTH_ARGUMENTS(x, 4,  BIT_PRINTING_ARGUMENTS_4BIT)
#define BIT_PRINTING_ARGUMENTS_16BIT(x) DOUBLE_LENGTH_ARGUMENTS(x, 8,  BIT_PRINTING_ARGUMENTS_8BIT)
#define BIT_PRINTING_ARGUMENTS_32BIT(x) DOUBLE_LENGTH_ARGUMENTS(x, 16, BIT_PRINTING_ARGUMENTS_16BIT)
#define BIT_PRINTING_ARGUMENTS_64BIT(x) DOUBLE_LENGTH_ARGUMENTS(x, 32, BIT_PRINTING_ARGUMENTS_32BIT)


#define thread_printf_32_bits(x) \
thread_printf( \
	"%04u'%04u'%04u'%04u'%04u'%04u'%04u'%04u", \
	get_bit(x,  0) * 1000 + \
	get_bit(x,  1) * 100 + \
	get_bit(x,  2) * 10 + \
	get_bit(x,  3) * 1, \
	get_bit(x,  4) * 1000 + \
	get_bit(x,  5) * 100 + \
	get_bit(x,  6) * 10 + \
	get_bit(x,  7) * 1, \
	get_bit(x,  8) * 1000 + \
	get_bit(x,  9) * 100 + \
	get_bit(x, 10) * 10 + \
	get_bit(x, 11) * 1, \
	get_bit(x, 12) * 1000 + \
	get_bit(x, 13) * 100 + \
	get_bit(x, 14) * 10 + \
	get_bit(x, 15) * 1, \
	get_bit(x, 16) * 1000 + \
	get_bit(x, 17) * 100 + \
	get_bit(x, 18) * 10 + \
	get_bit(x, 19) * 1, \
	get_bit(x, 20) * 1000 + \
	get_bit(x, 21) * 100 + \
	get_bit(x, 22) * 10 + \
	get_bit(x, 23) * 1, \
	get_bit(x, 24) * 1000 + \
	get_bit(x, 25) * 100 + \
	get_bit(x, 26) * 10 + \
	get_bit(x, 27) * 1, \
	get_bit(x, 28) * 1000 + \
	get_bit(x, 29) * 100 + \
	get_bit(x, 30) * 10 + \
	get_bit(x, 31) * 1 \
);

inline __device__ constexpr const char* ordinal_suffix(int n)
{
	return
		(n % 100 == 1 ? "st" :
		(n % 100 == 2 ? "nd" :
		(n % 100 == 3 ? "rd" :
		"th")));
}

#endif /* SRC_CUDA_PRINTING_CUH_ */
