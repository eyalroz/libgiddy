/**
 * Multi-reduce is the task of performing multiple reductions with the same kernel,
 * or sequence of kernels. That is, one has data intended for different 'bins', or
 * groups if you will, and the result of the multi-reduce is the reduction results
 * for all bins (or all bins with any data).
 *
 * This file contains kernels for the SPARSE TO DENSE variant:
 *
 * SPARSE input: consists of pairs of a value to participate in the reduction,
 * and the bin with which this value is associated. Of course, we prefer a SoA
 * vs AoS layout, so technically the input is two arrays of corresponding
 * lengths, data and indices.
 *
 * DENSE output: This is a single vector of bins, each of which holds
 * the result of a single reduction. The 'reduction index' or 'group index'
 * of each element is implicit - its index in the vector.
 *
 * Another technical caveat is that these kernels do not actually initialize the
 * data, only continue reducing with it, so that it is not their responsibility,
 * for example, to ensure untouched bin reduction results are actualy neutral.
 * For the case of summation, input data will only be added to existing bin
 * results, and it should be a good idea to zero them out in advance.
 */

#include "kernels/common.cuh"
#include "common.cuh" // ... common to multi-reduce operations

namespace cuda {

using namespace grid_info::linear;

namespace detail {

namespace block {

/**
 * Reduces per-thread arrays of values (typically in threads' local memory, and
 * each of length {@ref num_reductions})
 * into BlockSize/warp_size arrays. In other words, performs a factor-of-warp_size
 * reduction for every one of {@ref num_reductions}.
 *
 * @note it might be useful, for the processing down the road, to store
 * the data with a warp_size pitch, i.e. have each sequence of values start
 * at an address a with (a % warp_size == 0).
 *
 * @param[out] results_for_all_warps  The factor-of-warp_size-reduced results
 * @param[in]  thread_results         The thread-specific results to be further reduced
 * @param[in]  num_indices            The number of elements in each thread's result sequence,
 *                                    i.e. the number of reductions to perform
 */
template <typename ReductionOp, typename T>
__device__ void reduceThreadResultsToSharedMemWarpResult(
	T*        __restrict__  results_for_all_warps,
	const T*  __restrict__  thread_reduction_results,
	unsigned short          num_reductions)
{
	/*
	 * The expected caching is such, that each thread
	 */
	for(unsigned reduction_index_base = 0; reduction_index_base < num_reductions; reduction_index_base += warp_size) {
		// Shuffle-reduce warp_size consecutive arrays into a single one
		T reduced_element_saved_for_full_warp_write;
		for(unsigned reduction_index = reduction_index_base;
			reduction_index < reduction_index_base + warp_size;
			reduction_index ++) {
			// Consider interleaving shuffles here (having a warp-level multireduce of
			// up to 32 reductions - to "invert the matrix" of operation ordering so to speak)
			T reduced =
				detail::warp::reduce<T, ReductionOp>(thread_reduction_results[reduction_index]);
			if (reduction_index == lane::index()) {
				reduced_element_saved_for_full_warp_write = reduced;
			}
		}
		results_for_all_warps[reduction_index_base + lane::index()] = reduced_element_saved_for_full_warp_write;
		// No need for synchronization, since we're not going to read from there
		if (lane::index() == 0) {
//		tprintf("for reduction: %4u + %4u + %4u + %4u + %4u + %4u + %4u + %4u + %4u + %4u",
//				local_mem_results[0], local_mem_results[1], local_mem_results[2], local_mem_results[3],
//				local_mem_results[4], local_mem_results[5], local_mem_results[6], local_mem_results[7],
//				local_mem_results[8], local_mem_results[9]);
		}
	}
}

} // namespace block

namespace global {

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
template <typename ReductionOp, typename ResultDatum>
__device__ void multi_reduce_from_block_results(
	ResultDatum*         __restrict__  global_results,
	const ResultDatum*   __restrict__  block_results,
	unsigned short                     num_reductions) // no templated type here, it fits in shared memory
{
	using namespace grid_info;

	// TODO: Consider randomizing the elements chosen by each warp for copying,
	// i.e. not necessarily start 0 by permute
	typename ReductionOp::accumulator::atomic atomic_acc_op;
	for(unsigned reduction_index = linear::thread::index();
		reduction_index < num_reductions;
		reduction_index += linear::block::size())
	{
		atomic_acc_op(global_results[reduction_index], block_results[reduction_index]);
	}
}

} // namespace global

} // namespace detail

namespace kernels {
namespace reduction {

// Please get rid of this!
static const unsigned int reduce_by_indexDefaultSerializationFactor { 32 };


// Everything will use Serialization, except the most naive impl

namespace fixed_max_indices {

namespace initial_acc_per_warp   { }
namespace initial_acc_per_thread {
namespace initial_acc_into_registers { }
namespace initial_acc_into_local_mem {

namespace second_acc_per_warp {

namespace second_acc_into_registers { }
namespace second_acc_into_local_mem {

namespace third_acc_per_block {

namespace third_acc_into_shared_mem {

// Fourth accumulation must (effectively) be global and in global
template <
	unsigned IndexSize, typename ReductionOp, typename ResultDatum, typename InputDatum, typename Index,
	Index NumIndices, unsigned SerializationFactor>
__global__ void reduce_by_index(
	ResultDatum*       __restrict__  target,
	const InputDatum*  __restrict__  data,
	const Index*       __restrict__  indices,
	uint_t<IndexSize>          data_length,
	Index                            num_indices) // We won't use this one, but I want the same signature
{
	maybe_identify_function(__PRETTY_FUNCTION__);
	ResultDatum* warp_results_in_shared_mem = shared_memory_proxy<ResultDatum>();

	typename ReductionOp::accumulator op_accumulator;
	static_assert(NumIndices <= warp_size, "too many indices for this implementation");
	ResultDatum local_results[NumIndices];
	#pragma unroll
	for(Index i = 0; i < NumIndices; i++) {
		local_results[i] = ReductionOp::neutral_value();
	}
	uint_t<IndexSize> pos;
	int i;
	#pragma unroll
	for(i = 0, pos = thread::block_stride_start_position(SerializationFactor);
		i < SerializationFactor && pos < data_length; i++, pos += block::size()) {
		Index element_index = indices[pos];
		InputDatum element_datum = data[pos];
		op_accumulator(local_results[element_index], element_datum);

	}
	// Now we have thread-local results over SerializationFactors element each; let's
	// obtain warp-level results
	#pragma unroll
	for(Index i = 0; i < NumIndices; i++) {
		local_results[i] = detail::warp::reduce<ReductionOp, ResultDatum>(local_results[i]);
	}

	// We could go through shared memory as an intermediary, but let's go global instead
	auto group_index = lane::index();
	auto position_in_shared_mem = thread::index();
		// TODO: Perhaps use something more complex to avoid bank conflicts?
		// perhaps use a pitch of num_banks/2 + 1 instead of NumIndices, to make sure
		// we minimize conflicts?

	if (group_index < NumIndices) {
		warp_results_in_shared_mem[position_in_shared_mem] = local_results[group_index];
	}
	__syncthreads();
	// At this point, warp_results_in_shared_mem has block::num_warps_per_block() sequences
	// of NumIndices elements each, consecutive, with the results of all of the warps; let's
	// reduce them

	if (group_index >= num_indices) { return; }

	// The first element is assigned rather than accumulated into the result variable
	pos = group_index;
	ResultDatum block_accumulator_for_single_group = warp_results_in_shared_mem[pos];
	pos += NumIndices;
	// The rest are accumulated, so they go in the loop
	// TODO: Maybe num_full_wraps would becessary
	#pragma unroll
	for(int j = 1; j < block::num_warps(); j++, pos += NumIndices) {
		block_accumulator_for_single_group =
			op_accumulator(block_accumulator_for_single_group,
				warp_results_in_shared_mem[pos]);
	}

	// Now it's just the first NumIndices threads in the block (all in the first warp)
	// which each have the block-level reduction result for one group; we'll finally go atomic,
	// but this isn't necessary - we could have just written it to main memory, then called
	// a multi-reduce kernel on what's left

	atomic_acc_op(target[group_index], block_accumulator_for_single_group);
}

} //

namespace third_acc_into_global_mem { }

} // namespace third_acc_per_block
namespace third_acc_global {
template <
	unsigned IndexSize, typename ReductionOp, typename ResultDatum, typename InputDatum, typename Index,
	Index NumIndices, unsigned SerializationFactor>
__global__ void reduce_by_index(
	ResultDatum*       __restrict__  target,
	const InputDatum*  __restrict__  data,
	const Index*       __restrict__  indices,
	uint_t<IndexSize>          data_length,
	Index                            num_indices) // We won't use this one, but I want the same signature
{
	maybe_identify_function(__PRETTY_FUNCTION__);
	static_assert(NumIndices <= warp_size, "too many indices for this implementation");
	typename ReductionOp::accumulator op_accumulator;
	ResultDatum local_results[NumIndices];
	#pragma unroll
	for(Index i = 0; i < NumIndices; i++) {
		local_results[i] = ReductionOp::neutral_value();
	}
	uint_t<IndexSize> pos;
	int i;
	#pragma unroll
	for(i = 0, pos = thread::block_stride_start_position(SerializationFactor);
		i < SerializationFactor && pos < data_length; i++, pos += block::size()) {
		Index element_index = indices[pos];
		InputDatum element_datum = data[pos];
		op_accumulator(local_results[element_index], element_datum);
	}

	// Now we have thread-local results over SerializationFactors element each; let's
	// obtain warp-level results
	#pragma unroll
	for(Index i = 0; i < NumIndices; i++) {
		// We assume all warps are full warps
		local_results[i] = detail::warp::reduce<ReductionOp, ResultDatum>(local_results[i]);
	}
	__syncthreads();
	// We could go through shared memory as an intermediary, but let's go global instead
	typename ReductionOp::accumulator::atomic atomic_acc_op;
	auto group_index = lane::index();
	if (group_index >= NumIndices) {
		return;
	}
	atomic_acc_op(target[group_index], local_results[group_index]);
	//if (thread::global_index() > 30)
//	if ((thread::global_index() >= 1020) && (thread::global_index() < 1026)) {
//	tprintf("I have written %10u to index %5u", local_results[group_index], group_index);
//	}

}

// namespace third_acc_global{ }

} // namespace second_acc_into_local_mem
}
namespace second_acc_into_shared_mem { }
namespace second_acc_into_global_mem { }

} // namespace second_acc_per_warp
namespace second_acc_per_block { }
namespace second_acc_global { }

}
namespace initial_acc_into_global_mem { }
namespace initial_acc_into_shared_mem { }
} // namespace initial_acc_per_thread
namespace initial_acc_per_block  {

namespace initial_acc_into_shared_mem  {

namespace second_acc_global {
namespace second_acc_into_global_mem {

template <unsigned IndexSize, typename AccumulatingOp, typename ResultDatum, typename InputDatum,
	typename Index, unsigned SerializationFactor>
__global__ void reduce_by_index(
	ResultDatum*       __restrict__  target,
	const InputDatum*  __restrict__  data,
	const Index*       __restrict__  indices,
	uint_t<IndexSize>          data_length,
	Index                            num_indices)
{
	maybe_identify_function(__PRETTY_FUNCTION__);
	ResultDatum* block_level_results = shared_memory_proxy<ResultDatum>();
	detail::block::memset(block_level_results, num_indices, AccumulatingOp::neutral_value);
	__syncthreads();

	typename AccumulatingOp::accumulator::atomic op;
	// TODO: Template on the type of pos!
	int pos = thread::block_stride_start_position(SerializationFactor);
	#pragma unroll
	for(int i = 0; i < SerializationFactor && pos < data_length; i++) {
		Index      element_index = indices[pos];
		InputDatum element_datum = data[pos];
		op(block_level_results[element_index], element_datum);
		pos += block::size();
	}
	__syncthreads();

	// Now the shared memory region has the results of reduce_by_index for all
	// of this block's input. It remains to...

	detail::block::elementwise_accumulate<ResultDatum, InputDatum, decltype(op), decltype(num_indices)>(
		target, block_level_results, num_indices);
}

} // namespace second_acc_into_global_mem
} // second acc global

} // namespace initial_acc_into_shared_mem

} // namespace initial_acc_per_block
namespace initial_acc_global { }
} // namespace fixed_max_indices

} // namespace reduction
} // namespace kernels
} // namespace cuda
