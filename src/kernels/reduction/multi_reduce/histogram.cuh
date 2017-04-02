/**
 * @note This entire file can be essentially thrown away
 * in favor of using the sparse_to_dense kernel and
 * its device functions in common.cuh - if only we were to:
 *
 * 1. Replace pointers with iterators
 * 2. Make sure the resolution configuration takes
 * into account what kind of effort the reads from the
 * input iterator are going to entail (e.g. read from
 * main memory, constant value, many cycles of computational
 * generation etc.)
 */
#pragma once
#ifndef SRC_KERNELS_REDUCTION_MULTI_REDUCE_HISTOGRAM_CUH
#define SRC_KERNELS_REDUCTION_MULTI_REDUCE_HISTOGRAM_CUH

#include "kernels/common.cuh"
#include "common.cuh" // ... common to multi-reduce operations
#include "cuda/functors.hpp"
#include "cuda/on_device/atomics.cuh"

namespace cuda {
namespace kernels {
namespace reduction {
namespace histogram {

using namespace grid_info::linear;

enum : serialization_factor_t { DefaultSerializationFactor = 32 };

enum : bool {
	thread_does_reduce_running_sequences_of_same_value = true,
	thread_doesnt_reduce_running_sequences_of_same_value = false,
	do_skip_accumulation_of_neutral_values = true,
	dont_skip_accumulation_of_neutral_values = false,
};

namespace detail {

namespace thread_to_unknown_specifity {


/**
 * @note assumes no initialization is necessary and we may
 * just continue accumulation on top of the pre-existing
 * values in the @p histogram array
 */
template <
	unsigned IndexSize, unsigned BinIndexSize,
	bool ThreadReducesRunningSequencesOfSameValue,
	bool SkipAccumulationOfNeutralValues
> __device__ __forceinline__ void accumulate_histogram(
	uint_t<IndexSize>*           __restrict__  histogram,
	const uint_t<BinIndexSize>*  __restrict__  bin_indices,
	uint_t<IndexSize>                          bin_indices_length,
	uint_t<BinIndexSize>                       histogram_length,
	serialization_factor_t                     serialization_factor)
{
	using bin_index_type = uint_t<BinIndexSize>;
	using index_type = uint_t<IndexSize>;

	struct { bin_index_type bin_index; index_type count; } running = { 0, 0 };

	auto f = [&](index_type pos) {
		auto current_bin_index = bin_indices[pos];
		if (!ThreadReducesRunningSequencesOfSameValue) {
			atomic::add(&(histogram[current_bin_index]), (index_type) 1);
			return;
		}
		if (current_bin_index == running.bin_index) {running.count++; }
		else {
			if (running.count != 0) {
				// Note that this may perform poorly when the input has many elements
				// with the same aggregation index closeby, read by the same warp
				atomic::add(&(histogram[running.bin_index]), running.count);
			}
			running.bin_index = current_bin_index;
			running.count = 1;
		}
	};

	primitives::grid::linear::at_block_stride(bin_indices_length, f, serialization_factor);

	atomic::add(&(histogram[running.bin_index]), running.count);
}

} // namespace thread_to_unknown_specifity

namespace thread_to_block {

/**
 * This is very similar to thread-to-block sparse-to-dense in
 * general multi-reductions, except that the bin_indices is
 * implicitly always 1, and the reduction operation is
 * addition (cuda::functors::plus<index_type> if you will) or
 * incrementation when just adding 1 (is that faster)?
 *
 * @todo When we switch to using iterators instead of pointers
 * in the templates, there would be no need to implement
 * the histogram kernel the way that it is now - you could just
 * pass an iterator for the input data which simply yields 1
 * without actually reading from memory.
 *
 * @todo running sum optimization
 *
 * @note the histogram is overwritten, not accumulated into
 *
 */
template <
	unsigned IndexSize,
	unsigned BinIndexSize,
	bool ThreadReducesRunningSequencesOfSameValue = true,
	bool SkipAccumulationOfNeutralValues = false
> __device__ inline void accumulate_histogram(
	uint_t<IndexSize>*           __restrict__  block_histogram,
	const uint_t<BinIndexSize>*  __restrict__  bin_indices,
	uint_t<IndexSize>                          bin_indices_length,
	uint_t<BinIndexSize>                       histogram_length,
	serialization_factor_t                     serialization_factor)
{
	using bin_index_type = uint_t<BinIndexSize>;
	using index_type = uint_t<IndexSize>;

	primitives::block::fill_n(block_histogram, histogram_length, 0);

	__syncthreads();

	thread_to_unknown_specifity::accumulate_histogram<
		IndexSize, BinIndexSize, ThreadReducesRunningSequencesOfSameValue,
		SkipAccumulationOfNeutralValues
	>(block_histogram, bin_indices, bin_indices_length, histogram_length, serialization_factor);
}

} // namespace thread_to_block
} // namespace detail


/**
 *
 * Just like multi_reduce_sparse_to_dense, except that
 * the bin_indices is implicitly always 1, and the
 * reduction operation is addition
 * (cuda::functors::plus<index_type> if you will).
 *
 * Assumptions here:
 *
 * - Relatively high serialization factor (and consider a grid-serialized variant instead)
 */
template <unsigned IndexSize, unsigned BinIndexSize>
__global__ void histogram(
	uint_t<IndexSize>*           __restrict__  histogram,
	const uint_t<BinIndexSize>*  __restrict__  bin_indices,
	uint_t<IndexSize>                          bin_indices_length,
	uint_t<BinIndexSize>                       histogram_length,
	serialization_factor_t                     serialization_factor)
{
	using bin_index_type = uint_t<BinIndexSize>;
	using index_type = uint_t<IndexSize>;
	using ReductionOp = ::cuda::functors::plus<index_type>;

	if (ptx::special_registers::dynamic_smem_size() < IndexSize * histogram_length) {
		detail::thread_to_unknown_specifity::accumulate_histogram<
			IndexSize, BinIndexSize,
			thread_does_reduce_running_sequences_of_same_value,
			do_skip_accumulation_of_neutral_values
		>(histogram, bin_indices, bin_indices_length, histogram_length, serialization_factor);
		return;
	}

	index_type* block_histogram = shared_memory_proxy<index_type>();

	detail::thread_to_block::accumulate_histogram<IndexSize, BinIndexSize>(
		block_histogram, bin_indices, bin_indices_length, histogram_length, serialization_factor);
	__syncthreads();

	multi_reduce::dynamic_num_reductions::detail::block_to_global::dense_to_dense<IndexSize, ReductionOp>(
		histogram, block_histogram, histogram_length);
}

template <unsigned IndexSize, unsigned BinIndexSize>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          bin_indices_length,
		size_t                          histogram_length,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit = nullopt) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(histogram<IndexSize, BinIndexSize>),
			dynamic_shared_mem_limit
		)
	{
		auto size_of_single_full_histogram = histogram_length * IndexSize;
		bool can_fit_a_full_histogram_in_shared_memory =
			size_of_single_full_histogram <= available_dynamic_shared_memory_per_block;

		grid_construction_resolution            = thread;
		length                                  = bin_indices_length;
		serialization_option                    = runtime_specified_factor;
			// TODO: Get it to be auto_maximized!
		default_serialization_factor            = DefaultSerializationFactor;
		if (can_fit_a_full_histogram_in_shared_memory) {
			dynamic_shared_memory_requirement.per_block = histogram_length * IndexSize;
		}
	};
};

} // namespace histogram
} // namespace reduction
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_REDUCTION_MULTI_REDUCE_HISTOGRAM_CUH */
