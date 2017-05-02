#pragma once
#ifndef SRC_KERNELS_DECOMPRESSION_RUN_LENGTH_CUH
#define SRC_KERNELS_DECOMPRESSION_RUN_LENGTH_CUH

#include "kernels/common.cuh"
#include "cuda/on_device/primitives/warp.cuh"
#include "cuda/on_device/primitives/block.cuh"
#include "cuda/on_device/generic_shuffle.cuh"
#include "cuda/on_device/math.cuh"
#include "kernels/reduction/scan.cuh"
#include "cuda/functors.hpp"

namespace cuda {
namespace kernels {
namespace decompression {
namespace run_length_encoding {

using namespace grid_info::linear;

namespace detail {

// Everything a block needs to know in order
// to perform decompression, which does not
// involve any significant computation and can
// be allowed to be computed redundantly by all threads
//
// Note: We could theoretically use smaller
// types for some of the fields here; particularly,
// the number of runs intersecting a segment is
// no more than the segment size, which may
// be assumed not to need to be extended from, say,
// 2 to 4 bytes. Still, we'll keep it simple and 32-bit'y.
template<unsigned UncompressedIndexSize, unsigned RunLengthSize, unsigned RunIndexSize>
struct segment_decompression_params_t {
	uint_t<RunIndexSize>                            first_run_index; // the first run intersecting the segment
	uint_t<RunLengthSize>                           effective_length_of_first_run;
	size_type_by_index_size<RunIndexSize>           num_runs;
	size_type_by_index_size<UncompressedIndexSize>  uncompressed_start_position;
	size_type_by_index_size<UncompressedIndexSize>  decompressed_length;
};

using namespace grid_info::linear;

namespace block {

/**
 * @note this assumes the beginning of the block output is well-aligned; if it isn't,
 *  the function will work but performance will be sub-optimal.
 */
template<unsigned UncompressedIndexSize, unsigned UncompressedSize, unsigned RunLengthSize, unsigned RunIndexSize>
__forceinline__ __device__ void decompress_a_segment_with_few_runs(
	uint_t<UncompressedSize>*        __restrict__  decompressed_segment,
	const uint_t<RunLengthSize>*     __restrict__  run_lengths, // for this segment only
	const uint_t<UncompressedSize>*  __restrict__  run_data,    // for this segment only
	const segment_decompression_params_t<UncompressedIndexSize, RunLengthSize,  RunIndexSize>&
	                                               params)
{
	using uncompressed_type = uint_t<UncompressedSize>;
	using uncompressed_index_size_type = size_type_by_index_size<UncompressedIndexSize>;
	using run_length_type   = uint_t<RunLengthSize>;
	using index_type        = uint_t<UncompressedIndexSize>;
	using run_index_type    = uint_t<RunIndexSize>;


	constexpr const native_word_t elements_per_thread_write =
		UncompressedSize >= sizeof(native_word_t) ? 1 : sizeof(native_word_t) / UncompressedSize;

	if (params.num_runs == 1) {
		auto uniform_segment_value = run_data[0];
		primitives::block::fill_n(
			decompressed_segment,
			params.effective_length_of_first_run, uniform_segment_value);
		return;
	}

	struct {
		run_index_type                index;  // ... among the runs for this segment
		uncompressed_index_size_type  start;  // ... within the entire decompressed output (
		run_length_type               length; // in elements of type uncompressed_type

		__device__ uncompressed_index_size_type end() const
		{
			return start + length;
		}

		__device__ void advance(const run_length_type* run_lengths)
		{
			index++;
			start += length;
			length = run_lengths[index];
		}

	} current_run = {
		0, // run indices start at 0
		0, // first run starts at the beginning
		params.effective_length_of_first_run
	};

	// Note: We have two loops here, the outer one (the block-stride loop)
	// and the inner one (run-iteration loop). Now, because we synch the
	// warp lanes' knowledge of runs they've seen with each iteration of
	// the outer loop, we have that the _maximum_ number of iterations
	// of a lane on the inner loop is the increase in of
	// first_possible_run_for_next_write for the next outer-loop iteration.
	// In the next iteration of the outer loop, the number of possible
	// iterations of the inner loop decreases by that number.
	//
	// In other words, the total number of iterations of the inner loop
	// (when we take the maximum over the warp due to predicated execution),
	// over all iterators of the outer loop, is
	//
	//   I = R + L / W = L ( R/L + 1/W ) = L ( 1/A + 1/W )
	//
	// where
	// R = num_runs
	// L = params.decompressed_length
	// A = average run length
	//
	// and the number of iterations per element written is
	//
	// I / (L/W) = L/(L/W) ( 1/A + 1/W ) = W ( 1/A + 1/W ) = 1 + W/A
	//
	// This is pretty good, even if runs involve memory reads; after all,
	// other warps have likely already read that, so they'll be mostly
	// reads from L1 cache. The L1 cache will be no less than 16K, so
	// with any bit of fairness in warp scheduling we should not need
	// repeated reads from main memory.

	auto locate_and_perform_next_write = [&](index_type write_start_position) {

		// We assume that current_run is no earlier than the run we need for the next write

		using write_type = promoted_size<uncompressed_type>;
		static_assert(sizeof(write_type) / UncompressedSize == elements_per_thread_write, "Type/size mismatch");


		array<uint_t<UncompressedSize>, elements_per_thread_write> thread_write_buffer;
			// Notes:
			// * The total size of this variable is a multiple of sizeof(native_word_t)
			// * This should be optimized into registers

		#pragma unroll
		for(native_word_t element_within_thread_write = 0;
		    element_within_thread_write < elements_per_thread_write;
		    element_within_thread_write++)
		{
			auto output_position_for_current_element =
				write_start_position + element_within_thread_write;

			while (current_run.end() <= output_position_for_current_element) {
				current_run.advance(run_lengths);
			}
			thread_write_buffer[element_within_thread_write] = run_data[current_run.index];
		}

		*(reinterpret_cast<write_type*>(decompressed_segment + write_start_position)) =
			reinterpret_cast<const write_type&>(thread_write_buffer);

		/*
		{
			// At this point we're done with the current write; but - the
			// different lanes may have been considering different runs (if the lane
			// is not wholly covered by a single run) - with the last lane having
			// examined the farthest run. Since in subsequent runs we will only be
			// writing to farther elements in the output, it's useless for _any_
			// lanes to consider any run before this
			// last-run-considered-by-the-last-lane. So let's get it:
			//
			// Note: this is kind of expensive; for long runs - which is the typical
			// case - it might not be worth it

			auto next_write_start_position =
				write_start_position + grid_info::linear::block::length() * elements_per_thread_write;
			if (next_relative_write_position_for_this_thread < params.decompressed_length) {
				auto last_lanes_run_index =
					primitives::warp::get_from_last_lane(current_run.index);
				if (last_lanes_run_index > current_run.index) {
					primitives::warp::update_from_last_lane(current_run.index);
				}
			}
			// No 'else' here, since in the 'else' case that was the last write
			// for this thread in this segment
		}
		*/

	};

	// I'd thought of making this a separate function, with or without the lambda we call here,
	// but we use the 'innards' of the lambda below for the slack so it won't be much 'cleaner'

	auto truncated_decompressed_length =
		clear_lower_k_bits(params.decompressed_length, log2_constexpr(elements_per_thread_write));
		// in fact, this will round down to a multiple of 4 = sizeof(int)
	index_type write_start_position = thread::index_in_block() * elements_per_thread_write;

	#pragma unroll
	for(;
	    write_start_position  < truncated_decompressed_length;
	    write_start_position  += ::grid_info::linear::block::length() * elements_per_thread_write)
	{
		locate_and_perform_next_write(write_start_position);
	}

	//
	//
	// From here on - just handling the slack
	//
	//
	// We need to write to the last few elements, which can't be grouped
	// into a single 4-byte write. We'll do that with some threads
	// performing single-element writes


	if (elements_per_thread_write > 1) {

		auto num_actually_decompressed = truncated_decompressed_length;
		auto num_slack_elements = params.decompressed_length - num_actually_decompressed;

		if (num_slack_elements == 0) { return; }

		if (truncated_decompressed_length == 0) {
			if (thread::index_in_block() >= num_slack_elements) { return; }
			current_run.index = 0;
			current_run.start = params.uncompressed_start_position;
			current_run.length = run_lengths[current_run.index];
		}
		else {
			auto this_thread_wrote_last_element_so_far =
				(write_start_position == (truncated_decompressed_length - elements_per_thread_write) +
					elements_per_thread_write * grid_info::linear::block::length());
			if (primitives::warp::all_satisfy(not this_thread_wrote_last_element_so_far)) { return; }
			auto lane_which_wrote_the_last_element_so_far =
				primitives::warp::first_lane_satisfying(this_thread_wrote_last_element_so_far);
			auto lane_which_saw_the_farthest_run_so_far = lane_which_wrote_the_last_element_so_far;
			primitives::warp::update_from_lane(current_run, lane_which_saw_the_farthest_run_so_far);
			if (lane::index() >= num_slack_elements) { return; }
		}

		auto slack_write_position = num_actually_decompressed + lane::index();

		while (current_run.end() <= slack_write_position) {
			current_run.advance(run_lengths);
		}
		decompressed_segment[slack_write_position] = run_data[current_run.index];
	}
}

template<unsigned UncompressedIndexSize, unsigned UncompressedSize, unsigned RunLengthSize, unsigned RunIndexSize>
__forceinline__ __device__ void decompress_segment(
	uint_t<UncompressedSize>*        __restrict__   decompressed,
	// this may be a pointer to the absolute beginning of the input data,
	// or just the beginning of the segment - depending on whether PositionsAreRelative
	// or not.
	const uint_t<RunLengthSize>*     __restrict__   run_lengths, // for this segment only
	const uint_t<UncompressedSize>*  __restrict__   run_data,    // for this segment only
	const segment_decompression_params_t<UncompressedIndexSize, RunLengthSize,  RunIndexSize>&
	                                                params)
{

	// TODO: Act differently for blocks with high, low and super-low average
	// run length. (Doing that may require additional segment decompression
	// parameters to be computed.)

	decompress_a_segment_with_few_runs<UncompressedIndexSize, UncompressedSize, RunLengthSize, RunIndexSize>(
		decompressed, run_lengths, run_data, params);
}


// TODO: perhaps convert this into a named constructor idiom for
// segment_decompression_params_t ?
template<unsigned UncompressedIndexSize, unsigned UncompressedSize, unsigned RunLengthSize, unsigned RunIndexSize>
__forceinline__ __device__ segment_decompression_params_t<UncompressedIndexSize, RunLengthSize,  RunIndexSize>
resolve_segment_decompression_params(
	const uint_t<RunLengthSize>*  __restrict__  run_lengths,
	const uint_t<UncompressedIndexSize>*      __restrict__  segment_position_anchors,
	const uint_t<RunLengthSize>*  __restrict__  segment_position_anchor_intra_run_offsets,
	size_type_by_index_size<UncompressedIndexSize>          segment_length,
	size_type_by_index_size<UncompressedIndexSize>          num_segments,
	size_type_by_index_size<UncompressedIndexSize>          num_element_runs,
	size_type_by_index_size<UncompressedIndexSize>          total_uncompressed_length)
{
	using run_length_type = uint_t<RunLengthSize>;

	segment_decompression_params_t<UncompressedIndexSize, RunLengthSize,  RunIndexSize> params;
	// Each block is responsible for a certain number of output elements, not of
	// input elements (the latter option runs too much of a risk of block
	// workload imbalance). The block's output elements range from one
	// position anchor to the next (or to the end if it's the last one)

	auto anchor_index = grid_info::linear::block::index();
	auto next_anchor_index  = grid_info::linear::block::index() + 1;
	params.uncompressed_start_position = anchor_index * segment_length;

	auto is_last_segment = (next_anchor_index == num_segments);
	params.decompressed_length = is_last_segment ?
		total_uncompressed_length - params.uncompressed_start_position :
		segment_length;

	params.first_run_index = segment_position_anchors[anchor_index];

	// For the following code recall we assume the offsets into the runs
	// are less than the entire run lengths

	params.effective_length_of_first_run =
		run_lengths[params.first_run_index] -
		segment_position_anchor_intra_run_offsets[anchor_index];
	if (is_last_segment) {
		params.num_runs = num_element_runs - params.first_run_index;
		params.effective_length_of_first_run =
			(params.num_runs == 1) ?
				total_uncompressed_length - params.uncompressed_start_position :
				run_lengths[params.first_run_index] -
				segment_position_anchor_intra_run_offsets[anchor_index];
	}
	else {
		auto run_index_at_next_anchor = segment_position_anchors[next_anchor_index];
		params.num_runs =  run_index_at_next_anchor - params.first_run_index + 1;
		params.effective_length_of_first_run =
			(params.num_runs == 1) ?
				segment_length :
				run_lengths[params.first_run_index] -
				segment_position_anchor_intra_run_offsets[anchor_index];
	}

	return params;
}

} // namespace block

} // namespace detail

/**
 * Decompress a Run-Length-Encoded (RPE) compressed column,
 * which is made up of consecutive runs of identically-valued
 * elements (typically longer runs), using periodic anchoring
 * information.
 *
 * @note (constraints on the anchoring period?)
 *
 * @note we assume all input arrays are well-aligned.
 */
template<unsigned UncompressedIndexSize, unsigned UncompressedSize, unsigned RunLengthSize, unsigned RunIndexSize = UncompressedIndexSize>
__global__ void decompress(
	uint_t<UncompressedSize>*             __restrict__  decompressed,
	const uint_t<UncompressedSize>*       __restrict__  run_data,
	const uint_t<RunLengthSize>*          __restrict__  run_lengths,
	const uint_t<UncompressedIndexSize>*  __restrict__  segment_position_anchors,
	const uint_t<RunLengthSize>*          __restrict__  segment_position_anchor_intra_run_offsets,
	size_type_by_index_size<UncompressedIndexSize>      segment_length,
	size_type_by_index_size<UncompressedIndexSize>      num_segments,
	size_type_by_index_size<RunIndexSize>               num_element_runs,
	size_type_by_index_size<UncompressedIndexSize>      total_uncompressed_length)
{
	using uncompressed_type    = uint_t<UncompressedSize>;

	static_assert(is_power_of_2(UncompressedIndexSize), "UncompressedIndexSize is not a power of 2");
	static_assert(is_power_of_2(RunIndexSize         ), "RunIndexSize is not a power of 2");
	static_assert(is_power_of_2(RunLengthSize        ), "RunLengthSize is not a power of 2");

	// TODO: For large anchoring periods, consider breaking up segments into consecutive pieces,
	// with several warps handling each of them.

	auto block_params = detail::block::resolve_segment_decompression_params
		<UncompressedIndexSize, UncompressedSize, RunLengthSize, RunIndexSize>(
		run_lengths, segment_position_anchors,
		segment_position_anchor_intra_run_offsets, segment_length,
		num_segments, num_element_runs, total_uncompressed_length);

	detail::block::decompress_segment<UncompressedIndexSize, UncompressedSize, RunLengthSize, RunIndexSize>(
		decompressed + block_params.uncompressed_start_position,
		run_lengths  + block_params.first_run_index,
		run_data     + block_params.first_run_index,
		block_params); // Note we don't pass the anchor arrays
}


template<unsigned UncompressedIndexSize, unsigned UncompressedSize, unsigned RunLengthSize, unsigned RunIndexSize = UncompressedIndexSize>
class launch_config_resolution_params_t final : public kernels::launch_config_resolution_params_t {
public:
	using parent = kernels::launch_config_resolution_params_t;
public:
	static_assert(util::is_power_of_2(UncompressedIndexSize), "UncompressedIndexSize is not a power of 2");
	static_assert(util::is_power_of_2(RunIndexSize         ), "RunIndexSize is not a power of 2");
	static_assert(util::is_power_of_2(RunLengthSize        ), "RunLengthSize is not a power of 2");

	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          uncompressed_length,
		size_t                          segment_length,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit = nullopt) :
		parent(
			device_properties_,
			device_function_t(decompress<UncompressedIndexSize, UncompressedSize, RunLengthSize, RunIndexSize>),
			dynamic_shared_mem_limit
		)
	{
		enum { elements_per_thread_write =
			UncompressedSize >= sizeof(native_word_t) ? 1 : sizeof(native_word_t) / UncompressedSize };

		auto num_anchored_segments =
			util::div_rounding_up(uncompressed_length, segment_length);
		auto num_thread_writes_per_segment = div_rounding_up(segment_length, elements_per_thread_write);

		grid_construction_resolution  = block;
		length                        = num_anchored_segments;
		serialization_option          = none;
		quanta.threads_in_block       = warp_size;
		upper_limits.warps_in_block   = 2;
			// This should probably be architecture-specific, and perhaps
			// even depend on the data distribution if that's known.
			//
			// The point of setting this limit is make the block-length 'jump' between
			// consecutive writes by the same warp shorter, so that it has less work
			// looking for the next positions to write to. With an average run length
			// of 2, the number of iterations over warp_size-long sequences of runs
			// will on the average be less than 2 (i.e. one advance and two checks).
	};
};

} // namespace run_length_encoding
} // namespace decompression
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_DECOMPRESSION_RUN_LENGTH_CUH */
