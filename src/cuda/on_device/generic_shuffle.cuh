/*
 * Generic shuffle variants
 *
 * Originally based on:
 *
 * Bryan Catanzaro's CUDA generics
 * https://github.com/bryancatanzaro/generics/
 * Downloaded on: 2016-04-16
 *
 * Reimplemented by Eyal Rozenberg
 *
 * TODO:
 * - Lots of code duplication; should pass the XOR primitive
 *   as a template, that will cut the code by a factor of about 4
 */

#pragma once
#ifndef SRC_CUDA_ON_DEVICE_GENERIC_SHUFFLE_CUH_
#define SRC_CUDA_ON_DEVICE_GENERIC_SHUFFLE_CUH_

#ifdef __shfl_up
#undef __shfl_up
#undef __shfl_down
#undef __shfl_xor
#endif
#define IMPLEMENTING_SHUFFLE_OVERRIDES
#include <sm_30_intrinsics.h>
#include <vector_types.h>
#include "cuda/stl_array.cuh"
#include "cuda/api/types.h"

namespace detail {

template<typename InputIterator, typename OutputIterator, class UnaryOperation>
__host__ __device__ __forceinline__
OutputIterator transform(
	InputIterator input_it, InputIterator input_sentinel,
	OutputIterator output_it, UnaryOperation unary_op)
{
    while (input_it != input_sentinel ) {
    	*output_it = unary_op(*input_it);
    	++input_it, ++output_it;
    }
    return output_it;
}

template<int s>
__device__ __forceinline__
static void shuffle(
	const cuda::array<int, s>&  in,
	cuda::array<int, s>&        result,
	const int                   source_lane)
{
	transform(
		in.begin(), in.end(), result.begin(),
		[&source_lane](const int& x) { return ::__shfl(x, source_lane, warpSize); }
	);
}

template<int s>
__device__ __forceinline__
static void shuffle_down(
	const cuda::array<int, s>&  in,
	cuda::array<int, s>&        result,
	const unsigned int          delta)
{
	transform(
		in.begin(), in.end(), result.begin(),
		[&delta](const int& x) { return ::__shfl_down(x, delta, warpSize); }
	);
}

template<int s>
__device__ __forceinline__
static void shuffle_up(
	const cuda::array<int, s>&  in,
	cuda::array<int, s>&        result,
	const unsigned int          delta)
{
	transform(
		in.begin(), in.end(), result.begin(),
		[&delta](const int& x) { return ::__shfl_up(x, delta, warpSize); }
	);
}

template<int s>
__device__ __forceinline__
static void shuffle_xor(
	const cuda::array<int, s>&  in,
	cuda::array<int, s>&        result,
	const int                   lane_mask)
{
	transform(
		in.begin(), in.end(), result.begin(),
		[&lane_mask](const int& x) { return ::__shfl_xor(x, lane_mask, warpSize); }
	);
}


} // namespace detail

template<typename T>
__device__ __forceinline__
T __shfl(const T& t, const int& source_lane) {
	static_assert(sizeof(int) == 4 && sizeof(short) == 2 && sizeof(char) == 1, "Sizes sanity check failed");
	T result;
    constexpr auto num_int_shuffles = sizeof(T)/sizeof(int);
    if (num_int_shuffles > 0) {
    	auto& as_array = reinterpret_cast<cuda::array<int, num_int_shuffles>&>(result);
    	auto& t_as_array = reinterpret_cast<const cuda::array<int, num_int_shuffles>&>(t);
    	detail::shuffle<num_int_shuffles>(t_as_array, as_array, source_lane);
    }
    constexpr auto sub_int_remainder_size = sizeof(T) % sizeof(int);
    if (sub_int_remainder_size > 0) {
    	int surrogate;
    	const int& t_remainder = *(reinterpret_cast<const int*>(&t) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (surrogate) = reinterpret_cast<const char&> (t_remainder); break;
		case 2: reinterpret_cast<short&>(surrogate) = reinterpret_cast<const short&>(t_remainder); break;
		case 3: reinterpret_cast<char3&>(surrogate) = reinterpret_cast<const char3&>(t_remainder); break;
		}
    	surrogate = ::__shfl(surrogate, source_lane, warpSize);
    	int& result_remainder = *(reinterpret_cast<int*>(&result) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (result_remainder) = reinterpret_cast<char&> (surrogate); break;
		case 2: reinterpret_cast<short&>(result_remainder) = reinterpret_cast<short&>(surrogate); break;
		case 3: reinterpret_cast<char3&>(result_remainder) = reinterpret_cast<char3&>(surrogate); break;
		}
	}
    return result;
}

template<typename T>
__device__ __forceinline__
T __shfl_down(const T& t, const unsigned int& delta) {
	static_assert(sizeof(int) == 4 && sizeof(short) == 2 && sizeof(char) == 1, "Sizes sanity check failed");
	T result;
    constexpr auto num_int_shuffles = sizeof(T)/sizeof(int);
    if (num_int_shuffles > 0) {
    	auto& as_array = reinterpret_cast<cuda::array<int, num_int_shuffles>&>(result);
    	auto& t_as_array = reinterpret_cast<const cuda::array<int, num_int_shuffles>&>(t);
    	detail::shuffle_down<num_int_shuffles>(t_as_array, as_array, delta);
    }
    constexpr auto sub_int_remainder_size = sizeof(T) % sizeof(int);
    if (sub_int_remainder_size > 0) {
    	int surrogate;
    	const int& t_remainder = *(reinterpret_cast<const int*>(&t) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (surrogate) = reinterpret_cast<const char&> (t_remainder); break;
		case 2: reinterpret_cast<short&>(surrogate) = reinterpret_cast<const short&>(t_remainder); break;
		case 3: reinterpret_cast<char3&>(surrogate) = reinterpret_cast<const char3&>(t_remainder); break;
		}
    	surrogate = ::__shfl_down(surrogate, delta, warpSize);
    	int& result_remainder = *(reinterpret_cast<int*>(&result) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (result_remainder) = reinterpret_cast<char&> (surrogate); break;
		case 2: reinterpret_cast<short&>(result_remainder) = reinterpret_cast<short&>(surrogate); break;
		case 3: reinterpret_cast<char3&>(result_remainder) = reinterpret_cast<char3&>(surrogate); break;
		}
	}
    return result;
}

template<typename T>
__device__ __forceinline__
T __shfl_up(const T& t, const unsigned int& delta) {
	static_assert(sizeof(int) == 4 && sizeof(short) == 2 && sizeof(char) == 1, "Sizes sanity check failed");
	T result;
    constexpr auto num_int_shuffles = sizeof(T)/sizeof(int);
    if (num_int_shuffles > 0) {
    	auto& as_array = reinterpret_cast<cuda::array<int, num_int_shuffles>&>(result);
    	auto& t_as_array = reinterpret_cast<const cuda::array<int, num_int_shuffles>&>(t);
    	detail::shuffle_up<num_int_shuffles>(t_as_array, as_array, delta);
    }
    constexpr auto sub_int_remainder_size = sizeof(T) % sizeof(int);
    if (sub_int_remainder_size > 0) {
    	int surrogate;
    	const int& t_remainder = *(reinterpret_cast<const int*>(&t) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (surrogate) = reinterpret_cast<const char&> (t_remainder); break;
		case 2: reinterpret_cast<short&>(surrogate) = reinterpret_cast<const short&>(t_remainder); break;
		case 3: reinterpret_cast<char3&>(surrogate) = reinterpret_cast<const char3&>(t_remainder); break;
		}
    	surrogate = ::__shfl_up(surrogate, delta, warpSize);
    	int& result_remainder = *(reinterpret_cast<int*>(&result) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (result_remainder) = reinterpret_cast<char&> (surrogate); break;
		case 2: reinterpret_cast<short&>(result_remainder) = reinterpret_cast<short&>(surrogate); break;
		case 3: reinterpret_cast<char3&>(result_remainder) = reinterpret_cast<char3&>(surrogate); break;
		}
	}
    return result;
}

template<typename T>
__device__ __forceinline__
T __shfl_xor(const T& t, const int& lane_mask) {
	static_assert(sizeof(int) == 4 && sizeof(short) == 2 && sizeof(char) == 1, "Sizes sanity check failed");
	T result;
    constexpr auto num_int_shuffles = sizeof(T)/sizeof(int);
    if (num_int_shuffles > 0) {
    	auto& as_array = reinterpret_cast<cuda::array<int, num_int_shuffles>&>(result);
    	auto& t_as_array = reinterpret_cast<const cuda::array<int, num_int_shuffles>&>(t);
    	detail::shuffle_xor<num_int_shuffles>(t_as_array, as_array, lane_mask);
    }
    constexpr auto sub_int_remainder_size = sizeof(T) % sizeof(int);
    if (sub_int_remainder_size > 0) {
    	int surrogate;
    	const int& t_remainder = *(reinterpret_cast<const int*>(&t) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (surrogate) = reinterpret_cast<const char&> (t_remainder); break;
		case 2: reinterpret_cast<short&>(surrogate) = reinterpret_cast<const short&>(t_remainder); break;
		case 3: reinterpret_cast<char3&>(surrogate) = reinterpret_cast<const char3&>(t_remainder); break;
		}
    	surrogate = ::__shfl_xor(surrogate, lane_mask, warpSize);
    	int& result_remainder = *(reinterpret_cast<int*>(&result) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (result_remainder) = reinterpret_cast<char&> (surrogate); break;
		case 2: reinterpret_cast<short&>(result_remainder) = reinterpret_cast<short&>(surrogate); break;
		case 3: reinterpret_cast<char3&>(result_remainder) = reinterpret_cast<char3&>(surrogate); break;
		}
	}
    return result;
}

#endif /* SRC_CUDA_ON_DEVICE_GENERIC_SHUFFLE_CUH_ */
