#pragma once
#ifndef SRC_KERNELS_REDUCTION_SCAN_CUH_
#define SRC_KERNELS_REDUCTION_SCAN_CUH_

#include "kernels/common.cuh"
#include "cuda/on_device/primitives/block.cuh"
#include "cuda/on_device/primitives/warp.cuh"
#include "kernels/reduction/common.cuh"
#include "cuda/on_device/builtins.cuh"

namespace cuda {
namespace kernels {
namespace reduction {
namespace scan {

using primitives::inclusivity_t;

/*
 * The tricky part about a two-phase reduction or scan is how to balance the work
 * between the phases and what to pass from one to the other. Well, here's the key
 * to the matter as I see it: The more of your problem you can break up into
 * embarrassingly parallel work, the better. And in the case of a scan, just think of
 * it of k serial operations on  1/k consecutive sections of the data, where k is
 * the number of blocks necessary to keep the GPU busy all the time. That's not the
 * whole story, since these scans are missing something. What is that they're missing?
 * Just _one_ piece of data each - the reduction result for all data before the
 * beginning of that section.
 *
 * Thus the break-down into two kernels would seem to be:
 *
 * 1. Determine k (and the relevant block parameters)
 * 2. First kernel is a reduction (not a scan) over consecutive (1/k)-fractions of the
 *    data, written to k places in global memory (so, this would need a scratch buffer)
 * 3. In the second kernel, read just one element of the previous kernel's result,
 *    then compute the final prefix sum.
 *
 * This will work. Is it optimal? One could argue that if, during the first kernel,
 * we were to save intermediate results to global memory (e.g. reduction results
 * along the way, say once every block size elements) - later elements in the 1/k
 * section would not have to wait for the computation to conclude for the earlier
 * elements - since their block-initial value would just be computed already.
 * But does this really save anything? While these elements are waiting, other
 * work is being done anyway, and the GPUs are busy. And will the overall work
 * be any less? Well, no, only one extra reduction operation application would
 * be saved for every block (since its as though the block was the first, but its
 * first element had another value. So that's that.
 */

namespace detail {

namespace thread_to_thread {

/**
 * In this phase of a reduction operation, there is no inter-thread,
 * inter-warp or inter-block communication of any kind, and each thread
 * performs a reduction on the input element it reads (after applying
 * a pretransformation to them).
 *
 * Since this is a scan, we make sure all threads in a block only read
 * data within a contiguous segment of the overall input, however.
 *
 * @note we don't pass an "offseted" pointer, to just the segment data,
 * because we may need to absolute position for 'enumerated"-pretransform
 * unary operations
 *
 * @param segment_data
 * @param segment_length
 * @return the scalar reduction result for all elements read by the
 * executing thread
 */
template<
	unsigned IndexSize, typename ReductionOp,
	typename InputDatum,
	typename PretransformOp = functors::identity<InputDatum>
	>
__forceinline__ __device__ typename ReductionOp::result_type reduce_within_segment(
	const InputDatum*  __restrict__     data,
	size_type_by_index_size<IndexSize>  segment_length,
	uint_t<IndexSize>                   segment_start_position = 0)
{
	using index_type        = uint_t<IndexSize>;
	using pretransform_type = typename PretransformOp::result_type;
	using scan_result_type  = typename ReductionOp::result_type;
	using size_type         = size_type_by_index_size<IndexSize>;
	using namespace grid_info::linear;

	static_assert(std::is_same<
		typename PretransformOp::argument_type, InputDatum>::value,
		"The pretransform op must apply to input of the type specified for the input.");

	// single threads reduce independently

	ReductionOp reduction_op;
	scan_result_type thread_result = reduction_op.neutral_value();

	for(size_type pos = segment_start_position + thread::index_in_block();
		pos < segment_start_position + segment_length; pos += block::length())
	{
		typename ReductionOp::accumulator accumulation_op;
		auto pretransformed_datum = reduction::detail::apply_unary_op<size_type, PretransformOp>(
			pos, data[pos]);
		accumulation_op(thread_result, pretransformed_datum);
	}
	return thread_result;
}

} // namespace thread_to_thread

} // namespace detail

namespace reduce_segments {

/**
 * First phase of a scan (= prefix reduction): Perform a reduction rather than
 * a scan, with each block reducing a contiguous segment of the input data,
 * and the results written to global memory - so that in the second phase,
 * blocks (which again are each concerned with a single segment) will have
 * the information they need from other segments.
 *
 * @param segment_results
 * @param data
 * @param total_length
 * @param segment_length
 */
template<unsigned IndexSize, typename ReductionOp, typename InputDatum,
	typename PretransformOp = functors::identity<typename ReductionOp::argument_type> >
__global__ void reduce_segments(
	typename ReductionOp::result_type*  __restrict__  segment_reductions,
	const InputDatum*                   __restrict__  data,
	size_type_by_index_size<IndexSize>                total_length,
	size_type_by_index_size<IndexSize>                full_segment_length)
{
	using namespace grid_info::linear;
	using result_type = typename ReductionOp::result_type;
	using index_type  = uint_t<IndexSize>;
	using size_type   = size_type_by_index_size<IndexSize>;

//  We could have determined the segment length ourselves... but probably
//  it's more important to be flexible and allow callers to set this value
//	size_type full_segment_length =
//		round_up(div_rounding_up(total_length, grid::num_blocks()), block::length());

	static_assert(
		std::is_same<typename ReductionOp::first_argument_type, result_type>::value &&
		std::is_same<typename ReductionOp::second_argument_type, result_type>::value,
		"Only all-same-argument reduction operations are supported at this time.");
		// ... but note these don't have to be the same as InputDatum

	auto block_segment_start_position = full_segment_length * block::index();

	// Note: It is up to the calling code to ensure that
	// block_segment_start_position < total_length! Otherwise we'll have an access
	// violation

	if (total_length < block_segment_start_position) {
		// This can very well happen if the number of blocks is higher than the block
		// size - which is possible, if it takes a lot of blocks to keep the GPU busy,
		// and there are many cores, or perhaps the caller has limited the block size
		// etc.
		segment_reductions[block::index()] = ReductionOp().neutral_value();
//		block_printf("Empty block writing %u to segment_reductions[%u]",
//			(unsigned) ReductionOp().neutral_value(), block::index());
		return;
	}

	auto block_segment_length = builtins::minimum(
		total_length - block_segment_start_position, full_segment_length);

//	grid_printf("total_length = %u", (unsigned) total_length);
//	block_printf(
//		"%d, block_segment_start_position = %7u length_of_this_segment = %8d full_segment_length = %6u",
//		(int) block::is_last_in_grid(),
//		(unsigned) block_segment_start_position,
//		(int) block_segment_size,
//		(unsigned) full_segment_length);


	auto thread_result = detail::thread_to_thread::reduce_within_segment<
		IndexSize, ReductionOp, InputDatum, PretransformOp>(
	 		 data, block_segment_length, block_segment_start_position);
	auto block_result =	primitives::block::reduce<ReductionOp>(thread_result);

	if (thread::is_first_in_block()) {
//		thread_printf("Writing %u, the result of reduction %u elements, to segment_reductions[%u]",
//			block_result, block_segment_length, block::index());
		segment_reductions[block::index()] = block_result;
	}
}

/**
 * @Note: There are two ways to set the launch configuration for this kernel.
 * If you have some sequence of kernels with complex dependencies and need
 * fine-grained control of what your kernel will actually some over, you
 * can specify the length of segments to reduce, so as to get exactly that
 * many results (and have one grid block per one grid subsegment); if you
 * don't care about that - or you're willing to get that figure from the
 * resulting launch_configuration_t, just leave it unset (i.e. nullopt).
 * In the latter case, however, you will still need to pass that value along
 * to your kernel.
 */
template<
	unsigned IndexSize, typename ReductionOp,
	typename InputDatum,
	typename PretransformOp = functors::identity<typename ReductionOp::argument_type>
	>
class launch_config_resolution_params_t final : public kernels::launch_config_resolution_params_t {
public:
	using parent = kernels::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_,
		optional<size_t>                segment_length = nullopt) :
		parent(
			device_properties_,
			device_function_t(reduce_segments<IndexSize, ReductionOp, InputDatum, PretransformOp>)
		)
	{
		if (segment_length) {
			// TODO: This code path has not been tested
			auto num_segments = util::div_rounding_up(length_, segment_length.value());

			grid_construction_resolution        = resolution_t::block;
			length                              = num_segments;
			block_resolution_constraints.max_threads_per_block =
				util::round_up(
					std::min<grid_block_dimension_t>(
						segment_length.value(),
						device_properties.maxThreadsDim[0]
					), warp_size
				);
				// TODO: Perhaps we should just skip this constraint when segment_length is high enough?
		}
		else {
			grid_construction_resolution        = resolution_t::thread;
			length                              = length_;
		}
		serialization_option                    = auto_maximized;
		quanta.threads_in_block                 = warp_size;
		keep_gpu_busy_factor                    = 50;
			// Really? There's a difference between the last part
			// (each thread has one or two elements in registers,
			// and they interact) and the first part which is essentially
			// per-thread-accumulation. The first needs a high keep-busy
			// factor, the second part needs just 1

	};
};


} // namespace reduce_segments


namespace detail {

enum : bool {
	InitialChunkIsBlockAligned   = true,
	InitialChunkIsntBlockAligned = false,
	FinalChunkIsBlockAligned     = true,
	FinalChunkIsntBlockAligned   = false,
	BlockLengthIsAPowerOfTwo     = true,
	BlockLengthIsntAPowerOfTwo   = false,
};

/**
 * @note There is no synchronization w.r.t. the writes to the output,
 * i.e. if you intend to use the {@p results} array later, you'll
 * need some synchronization.
 *
 * @note the actual code for the Scan operation might not need the
 * return value of this function, but other code might.
 *
 * @todo template this for assumptions that segment starts with full
 * chunks, or ends with full chunks
 *
 * @param results
 * @param scratch_area a shared memory area
 * this function is allowed to overwrite. Must have at least warp_size
 * elements allocated, and should be in shared memory.
 * @param data
 * @param segment_start_pos
 * @param segment_length
 * @param preceding_segments_reduction
 */
template<
	unsigned IndexSize, typename ReductionOp, typename InputDatum,
	bool Inclusivity = inclusivity_t::Inclusive,
	bool InitialChunkIsBlockAligned = false,
	bool FinalChunkIsBlockAligned = false,
	bool BlockLengthIsAPowerOfTwo = false,
	typename PretransformOp = functors::identity<typename ReductionOp::argument_type>
	>
__device__ typename ReductionOp::result_type scan_segment(
	typename ReductionOp::result_type*  __restrict__  results,
	typename ReductionOp::result_type*  __restrict__  scratch_area,
	const InputDatum*                   __restrict__  data,
	size_type_by_index_size<IndexSize>                segment_length,
	uint_t<IndexSize>                                 segment_start_pos = 0,
	typename ReductionOp::result_type                 preceding_segments_reduction
		= ReductionOp().neutral_value())
{
	static_assert(BlockLengthIsAPowerOfTwo,
		"Only supporting power-of-two block lengths here, for now");
	using namespace grid_info::linear;
	using index_type = uint_t<IndexSize>;
	using size_type = size_type_by_index_size<IndexSize>;
	using result_type = typename ReductionOp::result_type;

	auto reduction_upto_preceding_chunk = preceding_segments_reduction;

	auto segment_end_pos = segment_start_pos + segment_length;

	// A "chunk" in this kernel is a sequence of data elements of length
	// block::length(), which we process together in each iteration of the loop
	// Actually, only the last block should have non-full chunks

	if (!InitialChunkIsBlockAligned) {
		auto full_chunks_start = round_up_to_multiple_of_power_of_2(segment_start_pos, block::length());
		if (full_chunks_start > segment_start_pos) {
			// TODO: If there's only one warp's worth to work on before the full chunks,
			// don't use a block-primitive; or adjust the block primitive to be able to
			// work on just some of the warps rather than the whole block
			auto pos = segment_start_pos + thread::index();
			result_type chunk_scan_result;
			result_type chunk_reduction_result;
			auto pretransformed_datum = pos < full_chunks_start ?
				reduction::detail::apply_unary_op<decltype(pos), PretransformOp>(pos, data[pos]) :
				ReductionOp().neutral_value();

			primitives::block::scan_and_reduce<ReductionOp, decltype(pretransformed_datum), Inclusivity>(
				scratch_area, pretransformed_datum, chunk_scan_result, chunk_reduction_result);
			if (pos < full_chunks_start) {
				results[pos] = reduction_upto_preceding_chunk +	chunk_scan_result;
			}
			reduction_upto_preceding_chunk += chunk_reduction_result;
			segment_length -= full_chunks_start - segment_start_pos;
			segment_start_pos = full_chunks_start;
			segment_end_pos += full_chunks_start - segment_start_pos;
		}
	}

	// At this point we're guaranteed that the segment start position is a multiple of block::length();
	// (well, assuming the block length is power of 2, at least). We are _not_ sure that it isn't 0 though.

	auto full_chunks_end = round_down_to_multiple_of_power_of_2(segment_end_pos, block::length());

	// TODO: Shouldn't the following become an at_block_stride? Hmm.
	auto pos = segment_start_pos + thread::index();

	for(; pos < full_chunks_end; pos += block::length())
	{
		result_type chunk_scan_result;
		result_type chunk_reduction_result;
		auto pretransformed_datum =
			reduction::detail::apply_unary_op<decltype(pos), PretransformOp>(pos, data[pos]);
		primitives::block::scan_and_reduce<
			ReductionOp, typename PretransformOp::result_type, Inclusivity>(
			scratch_area, pretransformed_datum, chunk_scan_result, chunk_reduction_result);
		results[pos] = reduction_upto_preceding_chunk +	chunk_scan_result;
		reduction_upto_preceding_chunk += chunk_reduction_result;
	}

	if (!FinalChunkIsBlockAligned && full_chunks_end < segment_end_pos) {
		// TODO: If only a warp's worth is left, don't do a block-primitive;
		// or adjust the block primitive to be able to work on just some of the
		// warps rather than the whole block
		result_type chunk_scan_result;
		result_type chunk_reduction_result;
		auto pretransformed_datum = pos < segment_end_pos ?
			reduction::detail::apply_unary_op<decltype(pos), PretransformOp>(pos, data[pos]) :
			ReductionOp().neutral_value();

		primitives::block::scan_and_reduce<ReductionOp, decltype(pretransformed_datum), Inclusivity>(
			scratch_area, pretransformed_datum, chunk_scan_result, chunk_reduction_result);
		if (pos < segment_end_pos) {
			results[pos] = reduction_upto_preceding_chunk + chunk_scan_result;
		}
		reduction_upto_preceding_chunk += chunk_reduction_result;
	}
	return reduction_upto_preceding_chunk;
}

} // namespace detail

namespace scan_using_segment_reductions {

/**
 * Second phase of a scan (= prefix reduction): Each block handles
 * a contiguous segment, with the reduction results for all previous
 * segments present in global memory. Thus there is no need for any
 * block-to-block communication or synchronization. The number of
 * segments is obviously the same as in the first phase (which should
 * be just enough to make sure all of the GPUs processing units are
 * kept busy)
 *
 * @note the segment length _must_ be a multiple of the block size
 * (yes, this means that theoretically the last block might be
 * more than one full block size larger than all the rest)
 *
 * @param results
 * @param segment_reduction_global_results
 * @param data
 * @param total_length
 * @param segment_length
 */
template<
	unsigned IndexSize, typename ReductionOp,
	typename InputDatum,
	bool Inclusivity = inclusivity_t::Inclusive,
	typename PretransformOp = functors::identity<typename ReductionOp::argument_type>
	>
__global__ void scan_using_segment_reductions(
	typename ReductionOp::result_type*         __restrict__  results,
	const typename ReductionOp::result_type*   __restrict__  segment_reductions,
	const InputDatum*                          __restrict__  data,
	size_type_by_index_size<IndexSize>                       total_length,
	size_type_by_index_size<IndexSize>                       segment_length)
{
	using namespace grid_info::linear;
	using result_type = typename ReductionOp::result_type;
	auto num_segments = grid::num_blocks();
	auto segment_index = block::index();
	result_type reduction_of_preceding_segments;
	auto scratch_area = shared_memory_proxy<result_type>();

	// Perhaps this condition below should go into the block reduce primitive anyway?

	if (segment_index > warp_size) {
		// Should have something which does the same as below but with a different name
		// and in the reductions/common.h file, and no pretransform
		auto thread_reduction_result_over_segment_reduction_results =
			detail::thread_to_thread::reduce_within_segment<IndexSize, ReductionOp, result_type>(
				segment_reductions, segment_index);
		reduction_of_preceding_segments =
			primitives::block::reduce<ReductionOp, result_type, true // all threads get result
				>(thread_reduction_result_over_segment_reduction_results);
	}
	else {
		auto single_segment_reduction_result = (lane::index() < segment_index) ?
			segment_reductions[lane::index()] : ReductionOp().neutral_value();
		reduction_of_preceding_segments =
			primitives::warp::reduce<ReductionOp>(single_segment_reduction_result);
	}

	// Now inclusive_reduction_upto_preceding_segment should be correct
	// for all threads in the block, regardless of who read what

	auto this_segment_s_length =
		block::is_last_in_grid() && (total_length % segment_length > 0) ?
		total_length % segment_length : segment_length;

//	block_printf("total_length is %6u, segment length is %6u, this segment's length is %6u",
//		total_length, segment_length, this_segment_s_length);


	// block_printf("Will scan_using_reduction_of_preceding_segments with segment length %4u and preceding reduction %6d",
	//	this_segment_s_length, inclusive_reduction_upto_preceding_segment);
	detail::scan_segment<
		IndexSize, ReductionOp, InputDatum, Inclusivity,
		reduction::scan::detail::InitialChunkIsBlockAligned,
		reduction::scan::detail::FinalChunkIsntBlockAligned,
		reduction::scan::detail::BlockLengthIsAPowerOfTwo,
		PretransformOp>(
		results, scratch_area, data, this_segment_s_length,
		segment_length * block::index(), reduction_of_preceding_segments);
}

template<
	unsigned IndexSize, typename ReductionOp,
	typename InputDatum,
	bool Inclusivity = inclusivity_t::Inclusive,
	typename PretransformOp = functors::identity<typename ReductionOp::argument_type>
	>
class launch_config_resolution_params_t final : public kernels::launch_config_resolution_params_t {
public:
	using parent = kernels::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_,
		size_t                          segment_length,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit = nullopt) :
		parent(
			device_properties_,
			device_function_t(scan_using_segment_reductions<
				IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp>),
			dynamic_shared_mem_limit
		)
	{
		// TODO: This code path has not been tested
		auto num_segments = util::div_rounding_up(length_, segment_length);

		grid_construction_resolution        = resolution_t::block;
		length                              = num_segments;
		block_resolution_constraints.max_threads_per_block =
			util::round_up(
				std::min<grid_block_dimension_t>(segment_length,
					device_properties.maxThreadsDim[0]),
				warp_size);
			// TODO: Perhaps we should just skip this constraint when segment_length is high enough?
		serialization_option                    = auto_maximized;
		dynamic_shared_memory_requirement.per_block =
			sizeof(typename ReductionOp::result_type) * warp_size * 2;
		quanta.threads_in_block                 = warp_size;
		keep_gpu_busy_factor                    = 50; // Really? There's a difference between the last part
													  // (each thread has one or two elements in registers,
													  // and they interact) and the first part which is essentially
													  // per-thread-accumulation. The first needs a high keep-busy
													  // factor, the second part needs just 1
		must_have_power_of_2_threads_per_block  = true;
	}
};


} // namespace scan_using_segment_reductions

namespace scan_single_segment {

template<
	unsigned IndexSize, typename ReductionOp,
	typename InputDatum,
	bool Inclusivity = inclusivity_t::Inclusive,
	typename PretransformOp = functors::identity<typename ReductionOp::argument_type>
	>
__global__ void scan_single_segment(
	typename ReductionOp::result_type*         __restrict__  result,
	const InputDatum*                          __restrict__  data,
	size_type_by_index_size<IndexSize>                       length)
{
	using namespace grid_info::linear;
	using result_type = typename ReductionOp::result_type;
	auto scratch_area = shared_memory_proxy<result_type>();

	detail::scan_segment<IndexSize, ReductionOp, InputDatum, Inclusivity,
		reduction::scan::detail::InitialChunkIsBlockAligned,
		reduction::scan::detail::FinalChunkIsntBlockAligned,
		reduction::scan::detail::BlockLengthIsAPowerOfTwo,
		PretransformOp>(
		result, scratch_area, data, length);
}

template<
	unsigned IndexSize, typename ReductionOp,
	typename InputDatum,
	bool Inclusivity = inclusivity_t::Inclusive,
	typename PretransformOp = functors::identity<typename ReductionOp::argument_type>
	>
class launch_config_resolution_params_t final : public kernels::launch_config_resolution_params_t {
public:
	using parent = kernels::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit = nullopt) :
		parent(
			device_properties_,
			device_function_t(scan_single_segment<
				IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp>),
			dynamic_shared_mem_limit
		)
	{
		grid_construction_resolution            = block;
		length                                  = 1;
		block_resolution_constraints.max_threads_per_block =
			util::round_up_to_power_of_2(
				std::min(
		        	(grid_block_dimension_t) util::round_up(length_, warp_size),
		        	(grid_block_dimension_t) device_properties.maxThreadsDim[0]
				)
			);
		must_have_power_of_2_threads_per_block  = true;
		serialization_option                    = none;
		dynamic_shared_memory_requirement.per_length_unit =
			sizeof(typename ReductionOp::result_type);
		quanta.threads_in_block                 = warp_size;
	};
};

} // namespace scan_single_segment

} // namespace scan
} // namespace reduction
} // namespace kernels
} // namespace cuda


#endif /* SRC_KERNELS_REDUCTION_SCAN_CUH_ */
