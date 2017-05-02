#pragma once
#ifndef SRC_CUDA_ON_DEVICE_GRID_INFO_CUH_
#define SRC_CUDA_ON_DEVICE_GRID_INFO_CUH_

#include "imports.cuh"
#include "builtins.cuh" // for lane_index()

#include <cuda_runtime.h>

#define __fd__ __forceinline__ __device__

/*
 ************************************************************************
 * Convenience one-liners relating to grid dimensions, indices within
 * the grid, block or warp, lane functionality etc.
 ************************************************************************
 */


// TODO: Perhaps have functions for strided copy in and out

namespace grid_info {

namespace thread {

template <typename Size = size_t>
__fd__ Size row_major_index_in_grid(uint3 thread_index, uint3 block_dimensions)
{
	return
		((block_dimensions.z == 1) ? 0 : (thread_index.z * block_dimensions.x * block_dimensions.y)) +
		((block_dimensions.y == 1) ? 0 : (thread_index.y * block_dimensions.x)) +
		thread_index.x;
}

template <typename Size = size_t>
__fd__ Size row_major_index_in_grid()
{
	return row_major_index_in_grid<Size>(threadIdx, blockDim);
}


} // namespace thread


// I couldn't use '1d, 2d, 3d since those aren't valid identifiers...
namespace linear {

namespace grid {

__fd__ unsigned num_blocks() {
	return gridDim.x;
}

} // namespace grid

namespace block {

__fd__ unsigned index() {
	return blockIdx.x;
}

__fd__ unsigned index_in_grid() {
	return index();
}

__fd__ bool is_first_in_grid() {
	return block::index_in_grid() == 0;
}

__fd__ bool is_last_in_grid() {
	return block::index_in_grid() == grid::num_blocks() - 1;
}

__fd__ unsigned size() {
	return blockDim.x;
}

// This one should only exist for linear blocks
__fd__ unsigned length() {
	return blockDim.x;
}

__fd__ unsigned num_warps() {
	return (size() + warp_size - 1) >> log_warp_size;
}

__fd__ unsigned num_full_warps() {
	return blockDim.x >> log_warp_size;
}

__fd__ unsigned num_threads() {
	return blockDim.x;
}

__fd__ unsigned last_thread_index() {
	return num_threads() - 1;
}

} // namespace block

namespace thread_block = block;

namespace grid {

template <typename Size = size_t>
__fd__ Size num_warps() {
	return (Size) num_blocks() * block::num_warps();
}

template <typename Size = size_t>
__fd__ Size num_threads() {
	return (Size) num_blocks() * block::size();
}

__fd__ unsigned int num_warps_per_block() {
	// This equals div_rounding_up(blockDim.x / warpSize)
	return (blockDim.x + warp_size - 1) >> log_warp_size;
}


} // namespace grid

namespace thread {

__fd__ unsigned index() {
	return threadIdx.x;
}

__fd__ unsigned index_in_block() {
	return index();
}

__fd__ bool is_first_in_block() {
	return index_in_block() == 0;
}

__fd__ bool is_last_in_block() {
	return index_in_block() == linear::block::size() - 1;
}


/**
 * Returns the global index of the thread - not within the block (the work group), but
 * considering all threads for the current kernel together - assuming a one-dimensional
 * grid.
 */
template <typename Size = size_t>
__fd__ Size index_in_grid() {
	return (Size) threadIdx.x + (Size) blockIdx.x * (Size) blockDim.x;
}

template <typename Size = size_t>
__fd__ Size global_index() {
	return index_in_grid();
}

__fd__ bool is_first_in_grid() {
	return block::is_first_in_grid() && thread::is_last_in_block();
}

__fd__ bool is_last_in_grid() {
	return block::is_last_in_grid() && thread::is_last_in_block();
}


/**
 * Use this for kernels in a 1-dimensional (linear) grid, in which each block of K
 * threads handles K * serialization_factor consecutive elements. That's pretty
 * common... (?)
 *
 * Anyway, each individual thread accesses data with a stride of K.
 *
 * @note Only supporting 2GB elements here
 *
 * @param serialization_factor The number of elements each thread would access
 * @return the initial position for a given thread
 */
template <typename Size=size_t>
__fd__  Size block_stride_start_position(unsigned serialization_factor = 1) {
	return threadIdx.x +
		((Size) serialization_factor * blockIdx.x) * blockDim.x;
}

} // namespace thread

namespace lane {

__fd__ unsigned index(unsigned thread_index) {
	return thread_index & (warp_size - 1);
}

__fd__ unsigned index() {
	return builtins::lane_index();
}

__fd__ unsigned index_in_warp() {
	return index();
}

__fd__ unsigned is_first() {
	return index_in_warp() == 0;
}

__fd__ unsigned is_last() {
	return index_in_warp() == warpSize - 1;
}

} // namespace lane

namespace thread {

__fd__ bool is_first_in_warp() {
	return lane::index() == 0;
}

__fd__ bool is_last_in_warp() {
	return lane::index_in_warp() == warpSize - 1;
}

} // namespace thread

namespace warp {

enum { first_lane = 0, last_lane = warp_size - 1 };

__fd__ int size() {
	return warpSize;
}

__fd__ int length() {
	return warpSize;
}

/**
 * Returns the global index of the warp the calling thread is in - not within the block
 * (the work group), but considering all blocks for the current kernel together -
 * assuming a one-dimensional grid.
 */
template <typename Size = size_t>
__fd__ Size global_index() {
	return thread::global_index<Size>() >> log_warp_size;
}

template <typename Size = size_t>
__fd__ Size index_of_first_lane() {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::index_in_block() & lane_index_mask;
}

template <typename Size = size_t>
__fd__ Size global_index_of_first_lane() {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::global_index<Size>() & lane_index_mask;
}

template <typename Size = size_t>
__fd__ Size index_in_grid() {
	return warp::global_index<Size>();
}

template <typename Size = size_t>
__fd__ Size index_in_grid_of_first_lane() {
	return warp::global_index_of_first_lane<Size>();
}

// TODO: Should all these be "ids" rather than "indices"? Hmmm.
__fd__ unsigned int index() {
	return thread::index() >> log_warp_size;
}

__fd__ unsigned int index_in_block() {
	return warp::index();
}

__fd__ bool is_first_in_block() {
	return warp::index_in_block() == 0;
}

__fd__ bool is_last_in_block() {
	return warp::index_in_block() == block::num_warps() - 1;
}

__fd__ bool is_first_in_grid() {
	return warp::is_first_in_block() && block::is_first_in_grid();
}

__fd__ bool is_last_in_grid() {
	return warp::is_last_in_block() && block::is_last_in_grid();
}

} // namespace warp

} // namespace linear

} // namespace grid_info

#undef __fd__

#endif /* SRC_CUDA_ON_DEVICE_GRID_INFO_CUH_ */
