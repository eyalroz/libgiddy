/*
 * primitives/grid.cuh
 *
 *  Created on: Apr 29, 2016
 *      Author: eyalroz
 */
#pragma once
#ifndef CUDA_ON_DEVICE_PRIMITIVES_GRID_CUH_
#define CUDA_ON_DEVICE_PRIMITIVES_GRID_CUH_

#include "common.cuh"

#define __fd__ __forceinline__ __device__

namespace cuda {
namespace primitives {
namespace grid {
namespace linear {
/**
 * Have all kernel threads perform some action over the linear range
 * of 0..length-1, at strides equal to the grid length, i.e. a thread
 * with index i_t in block with index i_b, where block lengths are n_b,
 * will perform the action on elements i_t, i_t + n_b, i_t + 2*n_b, and
 * so on.
 *
 * Thus, if in the following chart the rectangles represent
 * consecutive segments of n_b integers, the numbers
 * indicate which blocks work on which elements in "block stride":
 *
 *   -------------------------------------------------------
 *  |   1   |  222  |  333  |   1   |  222  |  333  |   1   |
 *  |  11   | 2   2 | 3   3 |  11   | 2   2 | 3   3 |  11   |
 *  |   1   |     2 |    3  |   1   |     2 |    3  |   1   |
 *  |   1   |  222  |    3  |   1   |  222  |    3  |   1   |
 *  |   1   | 2     | 3   3 |   1   | 2     | 3   3 |   1   |
 *  |  111  | 22222 |  333  |  111  | 22222 |  333  |  111  |
 *   -------------------------------------------------------
 *
 * and this is unlike @ref at_block_stride, for which instead
 * of 1, 2, 3, 1, 2, 3, 1 we would have 1, 1, 1, 2, 2, 2, 3
 * (or 1, 1, 2, 2, 3, 3, 4 if the grid has 4 blocks).
 *
 * @note assumes the number of grid threads is fixed (does that
 * always hold? even with dynamic parallelism?)
 *
 * @param length The length of the range (of integers) on which to act
 * @param f The callable to call for each element of the sequence.
 */
template <typename Function, typename Size = size_t>
__fd__ void at_grid_stride(Size length, const Function& f)
{
	auto num_grid_threads = grid_info::linear::grid::num_threads();
	for(promoted_size<Size> pos = ::grid_info::linear::thread::global_index();
		pos < length;
		pos += num_grid_threads)
	{
		f(pos);
	}
}

/**
 * Have all grid threads perform some action over the linear range
 * of 0..length-1, with each thread acting on a fixed number of items
 * (@p the SerializationFactor) at at stride of the block length,
 * i.e. a thread with index i_t in
 * block with index i_b, where block lengths are n_b,
 * will perform the action on elements
 *
 *  n_b * i_b      * serialization_factor + i_t,
 * (n_b * i_b + 1) * serialization_factor + i_t,
 * (n_b * i_b + 2) * serialization_factor + i_t,
 *
 * and so on. For lengths which are not divisible by n_b *
 * serialization_factor, threads in the last block will
 * work on less items.
 *
 * Thus, if in the following chart the rectangles represent
 * consecutive segments of n_b integers, the numbers
 * indicate which blocks work on which elements in "block stride":
 *
 *   -------------------------------------------------------
 *  |   1   |   1   |  222  |  222  |  333  |  333  |    4  |
 *  |  11   |  11   | 2   2 | 2   2 | 3   3 | 3   3 |   44  |
 *  |   1   |   1   |     2 |     2 |    3  |    3  |  4 4  |
 *  |   1   |   1   |  222  |  222  |    3  |    3  | 4  4  |
 *  |   1   |   1   | 2     | 2     | 3   3 | 3   3 | 44444 |
 *  |  111  |  111  | 22222 | 22222 |  333  |  333  |    4  |
 *   -------------------------------------------------------
 *
 * and this is unlike @ref at_grid_stride, for which instead
 * of 1, 1, 2, 2, 3, 3, 4 we would have 1, 2, 3, 1, 2, 3, 1 (if the
 * grid has 3 blocks) or 1, 2, 3, 4, 1, 2 (if the grid has 4 blocks).
 *
 *
 * @note There's a block-level variant of this primitive, but there -
 * each block applies f to the _same_ range of elements, rather than
 * covering part of a larger range.
 *
 * @note Does not handle cases of overflow, i.e. if @param length is
 * very close to the maximum possible value for @tparam Size, this
 * may fail.
 *
 * @note The current implementation avoids an extra condition at each
 * iteration - but at the price of divergence in the last warp; the
 * other trade-off is sometimes more appropriate
 *
 * @param length The length of the range (of integers) on which to act
 * @param serialization_factor the number of elements each thread is to
 * handle (serially)
 * @param f The callable to execute for each element of the sequence.
 */
template <typename Function, typename Size = size_t, typename SerializationFactor = serialization_factor_t>
__fd__ void at_block_stride(
	Size length, const Function& f, SerializationFactor serialization_factor = 1)
{
	Size pos = grid_info::linear::thread::block_stride_start_position(serialization_factor);
	auto block_length = ::grid_info::linear::block::length();
	if (pos + block_length * (serialization_factor - 1) < length) {
		#pragma unroll
		for(SerializationFactor i = 0; i < serialization_factor; i++) {
			f(pos);
			pos += block_length;
		}
	}
	else {
		#pragma unroll
		for(; pos < length; pos += block_length) { f(pos); }
	}
}

} // namespace linear
} // namespace grid

namespace block_to_grid {

/**
 * Accumulates the result of some computation from all
 * the blocks into a single, global (=grid-level) scalar -
 * without writes getting lost due to races etc.
 *
 * @note It is necessarily that at least the first thread
 * in every block calls this function, with the appropriate
 * value, otherwise it will fail. Other threads may either
 * call it or fail to call it, and the value they pass is
 * disregarded.
 *
 * @param accumulator The target in global memory into which
 * block results are accumulated. Typically one should care to
 * initialize it somehow before this primitive is used
 *(probably before the whole kernel is invoked).
 * @param value The result of some block-specific computation
 * (which would be different for threads of different
 * blocks of course)
 */
template <typename BinaryOp>
__fd__ void accumulation_to_scalar(
	typename BinaryOp::result_type* accumulator,
	typename BinaryOp::second_argument_type value)
{
	// TODO: Perhaps make a device-wide primitive out of this bit of code:
	if (grid_info::linear::thread::is_first_in_block()) {
		typename BinaryOp::accumulator::atomic atomic_accumulation_op;
		atomic_accumulation_op(*accumulator, value);
	}
}

} // namespace block_to_grid
} // namespace primitives
} // namespace cuda

#undef __fd__

#endif /* CUDA_ON_DEVICE_PRIMITIVES_GRID_CUH_ */
