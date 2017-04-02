/**
 * This file will contain non-kernel code used
 * in sparse-to-dense and dense-to-sparse multi-reduction kernels;
 * in other words - most/all of the __device__ functions but
 * none of the __global__ functions.
 *
 * see dense_to_dense.cuh and dense_to_dense.cuh for more information

 */
#pragma once
#ifndef SRC_KERNELS_REDUCTION_MULTI_REDUCE_COMMON_CUH_
#define SRC_KERNELS_REDUCTION_MULTI_REDUCE_COMMON_CUH_

#include "cuda/on_device/primitives/warp.cuh"
#include "cuda/on_device/primitives/block.cuh"
#include "cuda/on_device/primitives/grid.cuh"

namespace cuda {
namespace kernels {
namespace reduction {
namespace multi_reduce {
namespace dynamic_num_reductions {

namespace detail {

enum : bool {
	thread_does_reduce_running_sequences_of_same_value = true,
	thread_doesnt_reduce_running_sequences_of_same_value = false,
	do_skip_accumulation_of_neutral_values = true,
	dont_skip_accumulation_of_neutral_values = false,
};

namespace thread_to_unknown_specifity {

/**
 * @note doesn't clear results first. Perhaps it should - but then we would need
 * changes elsewhere
 */
template <unsigned IndexSize, typename ReductionOp, typename InputDatum, typename ReductionIndex,
	bool ThreadReducesRunningSequencesOfSameValue, bool SkipAccumulationOfNeutralValues>
__device__ void sparse_to_dense(
	typename ReductionOp::result_type*  __restrict__  results,
	const InputDatum*                   __restrict__  data,
	const ReductionIndex*               __restrict__  reduction_indices,
	uint_t<IndexSize>                                 data_length,
	ReductionIndex                                    num_reductions,
	serialization_factor_t                            serialization_factor)
{
	using index_type = uint_t<IndexSize>;
	using result_type = typename ReductionOp::result_type;

	// Here (unlike in block-level sparse-to-dense reduction) we assume the
	// global results have some sort of previous value we can accumulate into
	// (e.g. due to other blocks having written there,  or work by a previous
	// kernel); we don't try to coordinate some global initialization.

	typename ReductionOp::accumulator::atomic atomic_acc_op;
	typename ReductionOp::accumulator         acc_op;

	struct { ReductionIndex index; result_type reduction_result; } running = { 0, acc_op.neutral_value() };

	auto f = [&](index_type pos) {
		auto current_reduction_index = reduction_indices[pos];
		auto datum = data[pos];
		if (!ThreadReducesRunningSequencesOfSameValue) {
			atomic_acc_op(results[current_reduction_index], datum);
			return;
		}
		if (current_reduction_index == running.index) { acc_op(running.reduction_result, datum); }
		else {
			if (!SkipAccumulationOfNeutralValues ||
				running.reduction_result != acc_op.neutral_value())
			{
				// Note that this may perform poorly when the input has many elements
				// with the same aggregation index closeby, read by the same warp
				atomic_acc_op(results[running.index], running.reduction_result);
			}
			running.index = current_reduction_index;
			running.reduction_result = datum; // TODO: Perhaps acc_op with the neutral value here?
		}
	};

	primitives::grid::linear::at_block_stride(data_length, f, serialization_factor);

	if (!SkipAccumulationOfNeutralValues ||
		running.reduction_result != acc_op.neutral_value())
	{
		atomic_acc_op(results[running.index], running.reduction_result);
	}
}

} // namespace thread_to_unknown_specifity

namespace thread_to_thread {

/**
 *
 * With this primitive, individual threads perform a sparse-to-dense
 * multi-reduction operation, each on "its own" input data, which is read
 * at block stride.
 *
 *
 * @note doesn't clear the results
 *
 * @tparam IndexSize
 * @tparam ReductionOp
 * @tparam ResultDatum
 * @tparam InputDatum
 * @tparam ReductionIndex
 * @param this_thread_s_results
 * @param data
 * @param indices
 * @param num_reductions
 * @param data_length
 */
template <unsigned IndexSize, typename ReductionOp, typename InputDatum, typename ReductionIndex>
__device__ inline void sparse_to_dense(
	typename ReductionOp::result_type* __restrict__  thread_results,
	const InputDatum*                  __restrict__  data,
	const ReductionIndex*              __restrict__  indices,
	uint_t<IndexSize>                          data_length,
	ReductionIndex                                   num_reductions,
	serialization_factor_t                           serialization_factor)
{
	using namespace grid_info;
	using index_type = uint_t<IndexSize>;

	// TODO: Consider using the only-accumulate-at-end-of-sequence trick
	// like for block-level accumulation

	const auto& f = [&](index_type pos) {
		typename ReductionOp::accumulator op_accumulator;
		ReductionIndex element_index = indices[pos];
		InputDatum element_datum = data[pos];
		op_accumulator(thread_results[element_index], element_datum);
	};

	primitives::grid::linear::at_block_stride(data_length, f, serialization_factor);
}

// There's no thread-to-thread dense-to-dense function, since that wouldn't do anything

} // namespace thread_to_thread

namespace thread_to_warp {


/**
 * Reduces a (short-ish) vector of existing per-thread vectors into a
 * per-warp vector; the results can be going either into a new array
 * or into one of the existing ones - it's guaranteed any overwrites
 * will not interfere with the reads
 *
 * @note Requires the block be made up of full warps
 *
 * @param thread_results
 * @param num_reductions
 */
template <unsigned IndexSize, typename ReductionOp, typename ReductionIndex>
inline __device__ void dense_to_dense(
	typename ReductionOp::result_type*        __restrict__  warp_results,
	const typename ReductionOp::result_type*  __restrict__  thread_results,
	ReductionIndex                                          num_reductions)
{
	using index_type = uint_t<IndexSize>;
	using result_type = typename ReductionOp::result_type;

	struct { ReductionIndex index; result_type result; } saved_for_warp_write;
	auto num_full_write_cycles = num_reductions / warp_size;
	ReductionIndex reduction_index = 0;

	for(promoted_size<ReductionIndex> cycle_index = 0;
		cycle_index < num_full_write_cycles;
		cycle_index ++)
	{
		for(int i = 0; i < warp_size; i++) {
			auto reduction_result =
				primitives::warp::reduce<ReductionOp>(thread_results[reduction_index]);
			if (i == grid_info::linear::lane::index()) {
				saved_for_warp_write = { reduction_index, reduction_result };
			}
			reduction_index++;
		}
		warp_results[saved_for_warp_write.index] = saved_for_warp_write.result;
	}

	// There may be a last, incomplete write cycle to perform

	auto num_participating_in_last_write = num_reductions - reduction_index;
	for(int i = 0; i < num_participating_in_last_write; i++)
	{
		auto reduction_result =
			primitives::warp::reduce<ReductionOp>(thread_results[reduction_index]);
		if (i == grid_info::linear::lane::index()) {
			saved_for_warp_write = { reduction_index, reduction_result };
		}
		reduction_index++;
	}
	if (grid_info::linear::lane::index() < num_participating_in_last_write) {
		warp_results[saved_for_warp_write.index] = saved_for_warp_write.result;
	}
}

/**
 * Peforms a sparse-to-dense reduction from a per-thread pair of input arrays,
 * data and reduction indices, into a dense array of reduction indices -
 * one different such array for each warp in the grid.
 *
 * @note Does NOT require a block made up of full warps
 * @note doesn't clean the results
 */
template <unsigned IndexSize, typename ReductionOp, typename InputDatum, typename ReductionIndex>
__device__ void sparse_to_dense(
	typename ReductionOp::result_type*  __restrict__  warp_results,
	const InputDatum*                   __restrict__  data,
	const ReductionIndex*               __restrict__  reduction_indices,
	uint_t<IndexSize>                                 data_length,
	ReductionIndex                                    num_reductions,
	serialization_factor_t                            serialization_factor)
{
	using index_type = uint_t<IndexSize>;
	using result_type = typename ReductionOp::result_type;

	thread_to_unknown_specifity::sparse_to_dense<
		IndexSize, ReductionOp, InputDatum, ReductionIndex,
		thread_does_reduce_running_sequences_of_same_value,
		do_skip_accumulation_of_neutral_values
	>(warp_results, data, reduction_indices, data_length, num_reductions, serialization_factor);
}

} // namespace thread_to_warp

namespace thread_to_block {

/**
 * @note doesn't clear the block_results first
 */
template <unsigned IndexSize, typename ReductionOp, typename InputDatum, typename ReductionIndex,
	bool ThreadReducesRunningSequencesOfSameValue = true, bool SkipAccumulationOfNeutralValues = false>
__device__ void sparse_to_dense(
	typename ReductionOp::result_type*  __restrict__  block_results,
	const InputDatum*                   __restrict__  data,
	const ReductionIndex*               __restrict__  reduction_indices,
	uint_t<IndexSize>                                 data_length,
	ReductionIndex                                    num_reductions,
	serialization_factor_t                            serialization_factor)
{
	thread_to_unknown_specifity::sparse_to_dense<
		IndexSize, ReductionOp, InputDatum, ReductionIndex,
		ThreadReducesRunningSequencesOfSameValue,
		SkipAccumulationOfNeutralValues
	>(
		block_results, data, reduction_indices, data_length,
		num_reductions, serialization_factor);
}


template <unsigned IndexSize, typename ReductionOp, typename ReductionIndex>
inline __device__ void dense_to_dense(
	typename ReductionOp::result_type*        __restrict__  block_results,
	const typename ReductionOp::result_type*  __restrict__  thread_results,
	ReductionIndex                                          num_reductions)
{
	// This should actually have a much smarter implementation, which makes
	// sure there's the same amount of pressure on all shared memory banks;
	// however, that can also be achieved with this implementation if the
	// per-thread arrays are placed at +1 empty element offsets from each
	// (well, ok, this will only help 8-byte and larger elements to a limited
	// extent; for those we would need individual element read tricks)
	for(ReductionIndex index = 0; index < num_reductions; index ++) {
		block_results[index] =
			cuda::primitives::block::reduce<ReductionOp>(thread_results[index]);
	}
}

template <unsigned IndexSize, typename ReductionOp, typename ReductionIndex>
inline __device__ void dense_to_dense_in_place(
	typename ReductionOp::result_type*        __restrict__  block_results,
	typename ReductionOp::result_type*        __restrict__  thread_results,
	ReductionIndex                                          num_reductions)
{
	// the current implementation of dense_to_dense out of place can actually
	// be done in-place, due to the nature of the block reduce primitive...
	// I think.
	return dense_to_dense_in_place(block_results, thread_results, num_reductions);
}

} // namespace thread_to_block

namespace warp_to_block {


template <unsigned IndexSize, typename ReductionOp, typename ReductionIndex>
inline __device__ void dense_to_dense(
	const typename ReductionOp::result_type*  __restrict__  this_block_s_results,
	const typename ReductionOp::result_type*  __restrict__  this_warp_s_results,
	ReductionIndex                                          num_reductions)
{
	// TODO: Instead of using atomics, there's a (admittedly slight) chance
	// that doing a log-reduction at the warp level through multiple shared memory
	// arrays would work better, i.e. like the warp reduction going from warp_size
	// different values, to warp_size/2 etc. We would be trading atomics for log(warp_size)
	// syncthreads calls - maybe it's worth it?

	using index_type = uint_t<IndexSize>;
	using result_type = typename ReductionOp::result_type;
	// We're allowing just a single block to initialize the shared memory since
	// the number of shared memory banks is equal (on all known GPU microarchitectures)
	// to the warp size, so no more than warp_size elements will be writing at a time
	// anyway. For a mere memsetting we would not do this - since we pay with a
	// syncnthreads - but here it's probably worth it
	if (grid_info::linear::warp::is_first_in_block()) {
		primitives::warp::copy(this_block_s_results, this_warp_s_results, num_reductions);
	}
	__syncthreads();
	// Unfortunately, we don't know how the bank layout of the different warps'
	// results is - and I don't think we want to bother checking that (or do we?)
	// So let's be naive here.
	using atomic_acc_op = typename ReductionOp::accumulator::atomic;
	primitives::warp::elementwise_accumulate<result_type, result_type, atomic_acc_op , index_type>(
		this_block_s_results, this_warp_s_results, num_reductions);
}

/**
 * Same as {@ref dense_to_dense}, but the block's results are placed
 * in the results (shared memory) array previously used for the first
 * warp's results.
 *
 */
template <unsigned IndexSize, typename ReductionOp, typename ReductionIndex>
inline __device__ void dense_to_dense_in_place(
	typename ReductionOp::result_type*        __restrict__  block_results,
	const typename ReductionOp::result_type*  __restrict__  this_warp_s_results,
	ReductionIndex                                          num_reductions)
{
	using index_type = uint_t<IndexSize>;
	using result_type = typename ReductionOp::result_type;
	if (grid_info::linear::warp::is_first_in_block()) {
		// We let each warp take care of accumulating its own results,
		// and this warp has that done for free - its results are the
		// initialization of the block results
		//
		// Note: we're bending the semantics of __restrict__ here,
		// since we do have aliasing for the first warp, but the lack
		// of writes in this branch should makes it ok
		return;
	}
	// Unfortunately, we don't know how the bank layout of the different warps'
	// results is - and I don't think we want to bother checking that (or do we?)
	// So let's be naive here.
	using atomic_acc_op = typename ReductionOp::accumulator::atomic;
	primitives::warp::elementwise_accumulate<result_type, result_type, atomic_acc_op , index_type>(
		block_results, this_warp_s_results, num_reductions);
}


/**
 * Perhaps an in-place variant when num_reductions <= warp_size, with
 * the input
 *
 * @param thread_element_of_this_warp_s_results
 * @param num_reductions
 * @return
 */
template <unsigned IndexSize, typename ReductionOp, typename ReductionIndex>
inline __device__ void dense_to_dense_in_registers(
	typename ReductionOp::result_type&        __restrict__  block_results,
	const typename ReductionOp::result_type&  __restrict__  thread_element_of_this_warp_s_results,
	ReductionIndex                                          num_reductions)
{
	using index_type = uint_t<IndexSize>;

	// TODO: Instead of using atomics, there's a (admittedly slight) chance
	// that doing a log-reduction at the warp level through multiple shared memory
	// arrays would work better, i.e. like the warp reduction going from warp_size
	// different values, to warp_size/2 etc. We would be trading atomics for log(warp_size)
	// syncthreads calls - maybe it's worth it?
	using result_type = typename ReductionOp::result_type;
	if (grid_info::linear::warp::is_first_in_block()) {
		if (grid_info::linear::lane::index() < num_reductions) {
			block_results[grid_info::linear::lane::index()] =
				thread_element_of_this_warp_s_results;
		}
	}
	__syncthreads();
	using atomic_acc_op = typename ReductionOp::accumulator::atomic;
	primitives::warp::elementwise_accumulate<result_type, result_type, atomic_acc_op , index_type>(
		block_results, thread_element_of_this_warp_s_results, num_reductions);
}


// TODO: Implement variants of dense_to_dense and dense_to_dense_in_place which
// take all warps' results as their parameter, and do something smarter than just
// uncoordinated activity of each warp

// Note: No sparse-to-dense here - that can only start at the thread level


} // namespace warp_to_block

namespace block_to_global {

enum : bool {
	atomic_block_to_global = true,
	non_atomic_block_to_global = false
};

/**
 * Invoked when we have exactly one final result per block for every one
 * of the reductions we had to perform (stored at consecutive locations
 * at the beginning of the block's shared memory).
 *
 * @note A 'Parallel for All' post explored the question of whether it is
 * worth it to use a one-per-block atomic operation to finalize the
 * reduction over all blocks, or rather schedule an additional kernel;
 * the conclusions is that, on Kepler anyway, atomics are the better
 * choice.
 *
 * @param global_results
 * @param block_results
 * @param num_reductions
 */
template <unsigned IndexSize, typename ReductionOp, typename ReductionIndex, bool Atomic = true>
__device__ void dense_to_dense(
	typename ReductionOp::result_type*         __restrict__  global_results,
	const typename ReductionOp::result_type*   __restrict__  block_results,
	ReductionIndex                                           num_reductions)
{
	using namespace grid_info;
	using index_type = uint_t<IndexSize>;

	typename std::conditional<Atomic,
		typename ReductionOp::accumulator::atomic,
		typename ReductionOp::accumulator
	>::type acc_op;

	const auto& f = [&](index_type reduction_index) {
		// TODO: Consider randomizing the elements chosen by each warp for copying,
		// or even start reduction_index at warp_size * block::index() and
		// use modulus to advance it
		acc_op(global_results[reduction_index], block_results[reduction_index]);
	};

	primitives::block::at_block_stride(num_reductions, f);
}

// Can't have sparse_to_dense except from the thread level

} // namespace block_to_global

namespace thread_to_global {

template <unsigned IndexSize, typename ReductionOp, typename InputDatum, typename ReductionIndex,
	bool ThreadReducesRunningSequencesOfSameValue = true, bool AvoidAccumulatingNeutralValues = true>
__device__ void sparse_to_dense(
	typename ReductionOp::result_type*  __restrict__  global_results,
	const InputDatum*                   __restrict__  data,
	const ReductionIndex*               __restrict__  reduction_indices,
	uint_t<IndexSize>                           data_length,
	ReductionIndex                                    num_reductions,
	serialization_factor_t                            serialization_factor)
{
	// Here (unlike in block-level sparse-to-dense reduction) we assume the
	// global results have some sort of previous value we can accumulate into
	// (e.g. due to other blocks having written there,  or work by a previous
	// kernel); we don't try to coordinate some global initialization.

	thread_to_unknown_specifity::sparse_to_dense<
		IndexSize, ReductionOp, InputDatum, ReductionIndex,
		ThreadReducesRunningSequencesOfSameValue, AvoidAccumulatingNeutralValues
	>(global_results, data, reduction_indices, data_length, num_reductions, serialization_factor);
}


} // namespace thread_to_global

} // namespace detail
} // namespace dynamic_num_reductions
} // namespace multi_reduce
} // namespace reduction
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_REDUCTION_MULTI_REDUCE_COMMON_CUH_ */
