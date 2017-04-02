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

using namespace grid_info::linear;

namespace warp {

/**
 * After having exactly determined the range of compressed
 * runs which each warp needs to decompress, as well as the range
 * of uncompressed data the warp should fill with these runs,
 * this function "actually performs the decompression, on a
 * warp-by-warp level, and with no inter-warp synch.
 *
 * @param[out] block_decompressed the section of the global results
 * which this warp's block should fill. its starting and ending
 * positions are relative to this array
 * @param[in]  output_start_positions_for_runs the positions
 * within block_decompressed at which every one of the runs
 * provided will carry you
 * @param[in]  warp_decompressed_length the number of elements
 * this warp will decompress
 * @param[in] num_runs The length of {@ref run_data} and
 * {@ref output_start_positions_for_runs} - how many runs will this
 * thread use for decompression
 * @param[in] starting_output_position_for_warp where in the
 * block_decompressed array should the warp start writing
 */
template<unsigned IndexSize, unsigned UncompressedSize, unsigned RunLengthSize>
__device__ void decompress_with_per_warp_info(
	uint_t<UncompressedSize>*        __restrict__  block_decompressed,
	const uint_t<IndexSize>*          __restrict__  output_start_positions_for_runs,
	const uint_t<UncompressedSize>*  __restrict__  run_data,
	uint_t<IndexSize>            warp_decompressed_length,
	uint_t<IndexSize>            num_element_runs,
	uint_t<IndexSize>            starting_output_position_for_warp)
{
	using index_type = uint_t<IndexSize>;
	const index_type num_warp_writes = num_warp_sizes_to_cover(warp_decompressed_length);
	uint_t<IndexSize> first_run_not_all_spent = 0;
		// Note it's possible we've already written some copies
		// of the first unspent run's datum to the decompressed output,
		// but not all of them.

	for(index_type warp_write_index = 0; warp_write_index < num_warp_writes; warp_write_index++)
	{
//		warp_printf("preparing for the %2u%s write (out of %2u). "
//			"The first run which is not yet all spent is %4u",
//			warp_write_index+1,
//			ordinal_suffix(warp_write_index+1),
//			num_warp_writes, first_run_not_all_spent);

		// Now we read the data for (up to) 32 runs, beginning
		// with the first run to be used in this warp write

		auto lane_run_index = first_run_not_all_spent + lane::index();
		auto lane_run_starting_pos = lane_run_index < num_element_runs ?
			output_start_positions_for_runs[lane_run_index] : numeric_limits<index_type>::max();
			// For run indices beyond the last valid run, we pretend
			// they start _very_ far away, i.e. we never get to them,
			// i.e. we'll always try to use the last valid run as
			// though it's effectively infinite. This should not pose
			// a problem, since we do not try to write anything beyond
			// the last decompressed output position


		uint_t<IndexSize> starting_output_pos_for_this_warp_write =
			starting_output_position_for_warp + warp_write_index * warp_size;

		auto lane_run_is_beyond_this_warp_write =
			lane_run_starting_pos >= starting_output_pos_for_this_warp_write + warp_size;
		auto first_lane_with_run_beyond_this_warp_write =
			primitives::warp::first_lane_satisfying(lane_run_is_beyond_this_warp_write);
				// note this may be warp_size if all runs happen to be within
				// the warp write range (in which case the ballot is 0)

//		if (warp_write_index == 2) thread_printf(
//			"thread_output_position = %4u "
//			"run_from_shmem_index = %4u "
//			"run_from_shmem_starting_pos = %5u "
//			"run_from_shmem_is_beyond_this_warp_write = %d "
//			"first_lane_with_run_beyond_end_of_warp = %2u ",
//			thread_output_position,
//			run_from_shmem_index,
//			run_from_shmem_starting_pos,
//			run_from_shmem_is_beyond_this_warp_write,
//			first_lane_with_run_beyond_end_of_warp
//			);


		uint_t<IndexSize> thread_output_position = starting_output_pos_for_this_warp_write + lane::index();

		// Now we know the information regarding all runs usable in this warp is
		// held by the first num_lanes_with_runs_for_this_write ; but which of
		// these runs should this thread use? let's perform a binary
		// search:
		//
		//   left_lane   is the last lane whose run is potentially usable
		//   right_lane  is the first lane whose run is known to be unusable
		//
		// Note that, for the last writing warp, non-writing threads still
		// participate in the search, since (at least one) of them
		// might hold info about a run to be used in the warp write


		unsigned left_lane = 0, right_lane = first_lane_with_run_beyond_this_warp_write;
		while (primitives::warp::some_satisfy(left_lane + 1 < right_lane)) {
//			if (warp_write_index == 2) thread_printf("Searching with l = %2u and r = %2u", left_lane, right_lane);
			// TODO: Consider selecting an intermediary point based on the relative position
			// of the lane within the warp rather thank just always going from the middle
			unsigned mid_lane = (left_lane + right_lane) / 2;
				// Since left and right are at distance at least 2, the mid_lane
				// cannot be left_lane or right_lane
//			if (warp::index() == 3) thread_printf("before shuffle, my run_from_shmem_starting_pos is %u", run_from_shmem_starting_pos);
			auto mid_lane_run_starting_pos = __shfl(lane_run_starting_pos, mid_lane);
			auto mid_lane_s_run_may_be_usable_by_this_thread =
				mid_lane_run_starting_pos <= thread_output_position;
//			thread_printf("mid lane is %2u and its starting pos is %u. It is%s usable as my new left_lane.", mid_lane, mid_lane_run_starting_pos, (mid_lane_s_run_may_be_usable_by_this_thread ? "" :" not"));
			(mid_lane_s_run_may_be_usable_by_this_thread ? left_lane : right_lane) = mid_lane;
		}
		// left_lane is now the single potentially usable lane, and thus:
		auto lane_holding_relevant_run = left_lane;
		auto thread_run_index_for_this_write = __shfl(lane_run_index, lane_holding_relevant_run);
//		thread_printf("The lane for me is %2u , holding run index %4u (within this warp's run range).",
//			lane_holding_relevant_run, thread_run_index_for_this_write);
		if (thread_output_position < starting_output_position_for_warp + warp_decompressed_length) {
			block_decompressed[thread_output_position] = run_data[thread_run_index_for_this_write];
//			thread_printf("Writing run_data[%4u] = %5u to pos %4u in block_decompressed",
//				thread_run_index_for_this_write, run_data[thread_run_index_for_this_write],
//				thread_output_position);
		}

		// What if (left_lane + 1 != right_lane) ? Can that happen?

		if (warp_write_index == num_warp_writes - 1) { return; }

		// In this case we know all lanes must have written,
		// so we can obtain information about the last run from the last lane; and
		// we have to determine what's the first  run for use in the next run

		first_run_not_all_spent += first_lane_with_run_beyond_this_warp_write;
		uint_t<IndexSize> starting_pos_of_first_unused_run =
			first_lane_with_run_beyond_this_warp_write == warp_size ?
				output_start_positions_for_runs[first_run_not_all_spent + warp_size] :
				primitives::warp::get_from_lane(lane_run_starting_pos,
					first_lane_with_run_beyond_this_warp_write);

		if (starting_pos_of_first_unused_run > starting_output_pos_for_this_warp_write + warp_size) {
			// ... then the length of the last used run is beyond what we actually used,
			// so we will continue to use it in for the next write
			first_run_not_all_spent -= 1;
		}
	}
}

} // namespace warp

/**
 * All threads in a warp participate in a search for the
 * index into the uncompressed output at which the warp
 * will eventually write to.
 *
 * @param run_start_positions
 * @param starting_uncompressed_position_for_warp
 * @param num_runs
 * @return
 */
template<unsigned IndexSize, unsigned UncompressedSize, unsigned RunLengthSize>
__device__ uint_t<IndexSize> find_starting_run_index_for_warp(
	const uint_t<IndexSize>* __restrict__  run_start_positions, // uncompressed positions of course
	const uint_t<IndexSize>  __restrict__  starting_uncompressed_position_for_warp,
	uint_t<IndexSize>                      num_element_runs)
{
	using index_type = uint_t<IndexSize>;
	// Note: These search boundaries are common to the entire warp, i.e.
	// individual threads never set them to diverging values
	uint_t<IndexSize> search_l = 0;        // Last possibly-valid run index ('low' starting position)
	uint_t<IndexSize> search_r = num_element_runs; // First certainly-invalid run index ('high' starting position)

	// We perform a kind of a warp_size-ary search to find the consecutive pair of 'low' followed
	// by 'high' position - that's where the warp starts taking its run info from

	while (search_l + warp_size < search_r) {
//		warp_printf("Searching between run indices l = %2u and r = %2u", search_l, search_r);
		// In this phase of the search, we are guaranteed that each thread
		// in the warp makes a different guess, and also that none of the guesses
		// equals search_r - 1
		//
		// Each warp thread makes one sample - although it might not be
		// a different sample from those of other warps, if l and r are relatively
		// close to each other
		auto lane_guess = search_l + (search_r - search_l) * lane::index() / warp_size;
		auto output_pos_for_thread_s_guess_run = run_start_positions[lane_guess];
		auto lane_guess_is_high =
			output_pos_for_thread_s_guess_run > starting_uncompressed_position_for_warp;
		auto lanes_with_high_guesses = builtins::warp_ballot(lane_guess_is_high);
//		thread_printf(
//			"in search of the warp's first run, I checked run index %3u "
//			"which starts at output pos %3u %s %5u (%s w.r.t. warp's starting pos)"
//			"ballot_results = %8X",
//			lane_guess,
//			output_pos_for_thread_s_guess_run,
//			(lane_guess_is_high ? " >" : "<="),
//			starting_uncompressed_position_for_warp,
//			(lane_guess_is_high ? " higher" : "  lower")
//		);

		// The ballot results, from less-significant to most-significant, are
		// a sequence of 1's followed by a sequence of 0', since the predicate is monotone
		// in the sampled_run_start_pos, which is monotone in sampled_run_index,
		// which is monotone in the lane index.
		//
		// Note: it must be the case that at least the first guess is below or equal to
		// to our sought-after index, so we should not get invalid values here (but it
		// isn't necessarily the case

		if (lanes_with_high_guesses == 0) {
			// this will not get us into an infinte loop, since search_l is smaller
			// than the last lane's guess.
			search_l = primitives::warp::get_from_lane(lane_guess, grid_info::linear::warp::last_lane);
			continue;
		}

		auto first_lane_with_high_guess = count_trailing_zeros(lanes_with_high_guesses);
		search_r = __shfl(lane_guess, first_lane_with_high_guess);
		search_l = __shfl(lane_guess, first_lane_with_high_guess - 1);
//		warp_printf("Have updated search_l to %3u (from lane %2u) and search_r to %3u (from lane %2u)", search_l, first_lane_with_high_guess - 1, search_r, first_lane_with_high_guess);
	}

	if (search_l < search_r - 1) {
		// At this point, the search range can be fully covered by warp guesses.
//		warp_printf("Searching between close run indices l = %2u and r = %2u", search_l, search_r);
		auto lane_guess = search_l + lane::index();
		auto output_pos_for_thread_s_guess_run =
			lane_guess < search_r ? run_start_positions[lane_guess] : numeric_limits<index_type>::max();
		auto lane_guess_is_high =
			output_pos_for_thread_s_guess_run > starting_uncompressed_position_for_warp;
		auto lanes_with_high_guesses = builtins::warp_ballot(lane_guess_is_high);
//		thread_printf("find_starting_run_index_for_warp (l close to r): "
//			"I checked run index %3u which starts at output position %3u and is%s as a new search_l; ",
//			lane_guess,
//			output_pos_for_thread_s_guess_run,
//			(lane_guess_is_high ? " high" : " low ")
//		);

		// It _cannot_ be the case that lanes_with_high_guesses is all 0's
		auto first_lane_with_high_guess = count_trailing_zeros(lanes_with_high_guesses);
		search_l = __shfl(lane_guess, first_lane_with_high_guess - 1);
	}
	return search_l;
}

/**
 * This function carries out decompression using block-specific
 * rather than global data; and with the information regarding
 * the runs having adapted to such block-specificity by replacing
 * the run lengths with the starting output position for each run
 * (otherwise a block can't know where to start using just the
 * info regarding the runs it needs to use). With this
 * transformation, the position anchoring is no longer necessary
 * (and is not passed here).
 *
 * @note it is _not_ guaranteed that the first run is aligned with
 * the beginning of the output. It may well have a 'vestige'
 * which is decompressed into earlier output elements by another
 * blocks. It is specifically possible for this vestige to be
 * very large - much larger than the entire range of elements this
 * block should decompress.
 *
 * @param block_decompressed the memory region into which
 * the block is to decompress its relevant runs, and some
 * additional initial elements which correspond to the part of the
 * first run covered by previous blocks
 * @param run_start_positions positions relative to the @ref block_decompressed
 * array at which (block-relevant) run data should start being replicated
 * @param run_data the element to be replicated for each of the
 * runs (same index as for {@ref run_start_positions})
 * @param scratch scratch area for intermediary computation
 * @param block_decompressed_length the number of elements to be
 * decomposed by this thread overall
 * @param num_runs the number of runs to be used for decompressing
 * this block; this is also the length of @ref run_indices and of
 * @ref run_start_positions
 * @param vestige_length The number of elements in the first
 * run, which are 'covered' by another block, i.e. which another
 * block has/will decompress itself.
 */
template<unsigned IndexSize, unsigned UncompressedSize, unsigned RunLengthSize>
__device__ void decompress_with_run_start_positions(
	uint_t<UncompressedSize>*        __restrict__  block_decompressed,
	const uint_t<IndexSize>*         __restrict__  run_start_positions,
	const uint_t<UncompressedSize>*  __restrict__  run_data,
	uint_t<IndexSize>*               __restrict__  scratch,
	uint_t<IndexSize>                              block_decompressed_length,
	uint_t<IndexSize>                              num_element_runs,
	uint_t<RunLengthSize>                          vestige_length)
{
	unsigned num_writing_warps;
	uint_t<IndexSize> starting_output_position_for_warp;
	bool in_last_writing_warp;
	uint_t<IndexSize> warp_decompression_length;

	if (!block::is_last_in_grid()) {
		// In regular blocks, life is much simpler
		num_writing_warps = block::num_warps();
		in_last_writing_warp = grid_info::linear::warp::is_last_in_block();
		warp_decompression_length = block_decompressed_length / block::num_warps();
		starting_output_position_for_warp = warp_decompression_length * grid_info::linear::warp::index() + vestige_length;
	}
	else {
		// In the last block it's a bit messy
		uint_t<IndexSize> decompression_length_in_warp_sizes =
			num_warp_sizes_to_cover(block_decompressed_length);

		// Note this corresponds to directly with the launch configuration resolution code
		uint_t<IndexSize> decompression_length_for_full_warp =
			warp_size * div_rounding_up(decompression_length_in_warp_sizes, block::num_warps());

		num_writing_warps = div_rounding_up(block_decompressed_length, decompression_length_for_full_warp);

		if (grid_info::linear::warp::index() >= num_writing_warps) { return; }

		starting_output_position_for_warp =
			decompression_length_for_full_warp * grid_info::linear::warp::index() + vestige_length;

		in_last_writing_warp = (grid_info::linear::warp::index() + 1== num_writing_warps);

		warp_decompression_length = in_last_writing_warp ?
			block_decompressed_length - ( (num_writing_warps - 1) * decompression_length_for_full_warp) :
			decompression_length_for_full_warp;
	}

//	warp_printf(
//		"This warp is %s warp, so it will decompress %4u elements (in %4u writes) out of the block's total %5u: [ %5u, %5u )",
//		(in_last_writing_warp ? "the last writing" : "          a full"),
//		warp_decompression_length,
//		num_warp_sizes_to_cover(warp_decompression_length),
//		block_decompressed_length,
//		starting_output_position_for_warp,
//		starting_output_position_for_warp + warp_decompression_length
//		);


	// TODO: Figure out why we need this synch
	__syncthreads();


	// Each warp (of those that will actually decompress anything)
	// searches the runs' starting positions array to determine which run
	// covers its first output position
	uint_t<IndexSize> start_run_index_for_warp = find_starting_run_index_for_warp
		<IndexSize, UncompressedSize, RunLengthSize>(
		run_start_positions,
		starting_output_position_for_warp,
		num_element_runs);
//	warp_printf("start_run_index_for_warp = %u", start_run_index_for_warp);
	auto starting_run_indices_for_all_warps = scratch;
	primitives::block::share_warp_datum_with_whole_block(
		start_run_index_for_warp, starting_run_indices_for_all_warps);

	uint_t<IndexSize> end_run_index_for_warp =
		in_last_writing_warp ? num_element_runs : starting_run_indices_for_all_warps[grid_info::linear::warp::index() + 1];

	uint_t<IndexSize> num_runs_in_warp_decompression =
		end_run_index_for_warp  - start_run_index_for_warp + (in_last_writing_warp ? 0 : 1);
		// The final summand allows for the case of the next warp starting in the middle
		// of a run; this warp involves that first run as well - and if it
		// doesn't, no matter, it shouldn't hurt to include it here
//	warp_printf(
//		"This warp will use the %4d runs in the range [ %4d , %4d ] , which ends with the %s.",
//		(int) num_runs_in_warp_decompression, (int) start_run_index_for_warp,
//		(int) (num_runs_in_warp_decompression + start_run_index_for_warp - 1),
//		(in_last_writing_warp ? "last run for the block" : "next warp's first run (which may or may not be used)")
//	);

	warp::decompress_with_per_warp_info<IndexSize, UncompressedSize, RunLengthSize>(
		block_decompressed,
		run_start_positions + start_run_index_for_warp,
		run_data + start_run_index_for_warp,
		warp_decompression_length,
		num_runs_in_warp_decompression,
		starting_output_position_for_warp);
}

namespace block {

template<unsigned IndexSize, unsigned UncompressedSize, unsigned RunLengthSize>
__device__ void resolve_run_start_positions(
	uint_t<IndexSize>*                              __restrict__  run_start_positions,
	uint_t<IndexSize>&                                            uncompressed_position_before_starting_run,
	const uint_t<RunLengthSize>* __restrict__  run_lengths,
	uint_t<IndexSize>*                              __restrict__  block_scan_scratch_area,
	uint_t<RunLengthSize>&                     offset_in_first_run_for_block,
	uint_t<RunLengthSize>                      vestige_length_at_next_anchor,
	uint_t<IndexSize>                          block_anchor_index,
	uint_t<IndexSize>                          starting_uncompressed_position_for_block,
	bool                                             this_block_reaches_end_of_anchors,
	uint_t<IndexSize>                          start_run_index_for_block,
	uint_t<IndexSize>                          num_runs_to_use_in_this_block)
{
	uncompressed_position_before_starting_run =
		starting_uncompressed_position_for_block - offset_in_first_run_for_block;
	using index_type = uint_t<IndexSize>;
	using run_length_type = uint_t<RunLengthSize>;

	// TODO: The run lengths may start with non-128-byte aligned data,
	// making reads somewhat inefficient. Consider making the reads
	// first reach an alignment point, then continue aligned

	const auto &scan_segment = ::cuda::kernels::reduction::scan::detail::scan_segment<
		IndexSize, cuda::functors::plus<index_type>, run_length_type,
		primitives::inclusivity_t::Exclusive,
		reduction::scan::detail::InitialChunkIsntBlockAligned, // TODO: It probably is actually aligned
		reduction::scan::detail::FinalChunkIsntBlockAligned,
		reduction::scan::detail::BlockLengthIsAPowerOfTwo,
		functors::identity<index_type>>;

	// TODO: This could probably be improved some. For example,
	// we could use unsigned offsets from the first output position, so that
	// even when index_type is 64-bit, we would be working here with 32-bit; but
	// note we will need to make sure there can't be any overflow issue with the last
	// run-length value of the run
	uint_t<IndexSize> position_after_all_relevant_runs = scan_segment(
		run_start_positions,
		block_scan_scratch_area,
		run_lengths + start_run_index_for_block,
		num_runs_to_use_in_this_block,
		0, 0);

	once_per_block {
		run_start_positions[num_runs_to_use_in_this_block] = position_after_all_relevant_runs;
	}
	__syncthreads();
	// Note that the positions we've produced...
	//
	// - are NOT relative to the entire kernel's output, and
	// - are NOT relative to this block's output, but
	// - ARE relative to the beginning of the run which populates the first
	//   element of the block's otuput


}

} // namespace block

} // namespace detail

/**
 * Decompress the segment of an RLE (Run-Length Encoded)
 * array between two consecutive position anchors (or the
 * last position anchor and the end of the array).
 *
 *
 * @todo Consider capping the run lengths here, or rather -
 * assuming that they're capped in the input. That may be useful
 * in certain respects.
 *
 * @note we assume all input arrays are well-aligned.
 *
 * @note there's currently an implicit assumption that
 * the Uncompressed type has a size of 4 bytes. If the size
 * is 8, the effect is very minor and we do well enough
 * (relatively to the 4-byte case; each coalesced write we have
 * is in fact two coalesced write, and could have been
 * split into two half-warps). However, if the size is less
 * than 4 then we have about twice as much overhead work
 * per full 128-byte write than we should have.
 *
 * @tparam IndexSize the size in bytes of lengths of the input and
 * output arrays and related data; if it can be limited to 32-bit
 * insigned integers, that may help performance somewhat
 * @tparam UncompressedSize the size of data elements with which we're
 * working; we pretend they're unsigned integers of this size,
 * but they can actually be of any (trivially copyable) type
 * @tparam RunLengthSize The size of the unsigned integer type
 * used for encoding run lengths; try to make this as small as possible.
 * @param[out] decompressed The output array into which runs get
 * unfolded out
 * @param[in] run_data For each same-value run - what is that value?
 * @param[in] run_lengths How many elements are there in each of
 * the same-value runs in the decompressed data?
 * @param[in] position_anchors There is no consistent correspondence
 * between positions in the decompressed output and positions in the
 * input; to avoid having to search the input, we pre-determine the
 * input position at which to find elements of the array, and pass
 * them here
 * @param[in] intra_run_position_anchors Our anchors may well
 * point to the "middle" of a run, not necessary the first element of it;
 * so when anchoring, it is not enough to identify the run, we need to
 * know the position within it
 * @param[in] anchoring_period An anchor appears in the position_anchors
 * array for every anchoring_period uncompressed elements, to the position
 * in the runs arrays of the run containing that uncompressed element
 * @param[in] num_anchors Number of position anchors for the compressed data
 * @param[in] num_runs Number of runs of elements with the same values in
 * the uncompressed data
 * @param[in] anchored_segments_per_block The number of sequences of
 * {@ref anchoring_period} decompressed elements which are to be processed
 * by each block. TODO: Consider making that a template parameter
 * @param[in] uncompressed_length Number of elements to be written into
 * the @ref decompressed buffer
 */
template<unsigned IndexSize, unsigned UncompressedSize, unsigned RunLengthSize>
__global__ void decompress(
	uint_t<UncompressedSize>*        __restrict__  decompressed,
	const uint_t<UncompressedSize>*  __restrict__  run_data,
	const uint_t<RunLengthSize>*     __restrict__  run_lengths,
	const uint_t<IndexSize>*         __restrict__  position_anchors,
	const uint_t<RunLengthSize>*     __restrict__  intra_run_anchor_offsets,
	uint_t<IndexSize>                              anchoring_period,
	uint_t<IndexSize>                              num_anchors,
	uint_t<IndexSize>                              num_element_runs, // = length of the run_* arrays
	uint_t<IndexSize>                              uncompressed_length)
{
	using index_type = uint_t<IndexSize>;
	static_assert( (IndexSize & (IndexSize - 1)) == 0, "Input indices' size in bytes must be a power of 2");
	// Each block is responsible for a certain number of output elements, not of
	// input elements (the latter option runs too much of a risk of block
	// workload imbalance). The block's output elements range from one
	// position anchor to the next (or to the end if it's the last one)

	auto block_anchor_index = block::index();
	auto next_anchor_index  = block::index() + 1;

	auto start_run_index_for_block = position_anchors[block_anchor_index];
	auto this_block_reaches_end_of_anchors = next_anchor_index >= num_anchors;
	auto vestige_length_at_next_anchor =
		this_block_reaches_end_of_anchors ? 0 :
		intra_run_anchor_offsets[next_anchor_index];

	uint_t<IndexSize> starting_uncompressed_position_for_block = block_anchor_index * anchoring_period;
	uint_t<IndexSize> length_to_decompress_in_this_block =
		this_block_reaches_end_of_anchors ?
		uncompressed_length - starting_uncompressed_position_for_block : anchoring_period;
	uint_t<IndexSize> num_runs_to_use_in_this_block =
		(this_block_reaches_end_of_anchors ? num_element_runs  :
		position_anchors[next_anchor_index] + (vestige_length_at_next_anchor > 0 ? 1 : 0))
		- start_run_index_for_block;
	auto offset_in_first_run_for_block = intra_run_anchor_offsets[block_anchor_index];

	index_type* run_start_positions = shared_memory_proxy<index_type>();
		// ... and we had better have enough shared mem!
	index_type* block_scan_scratch_area = run_start_positions +
		round_up_to_multiple_of_power_of_2(num_runs_to_use_in_this_block + 1, warp_size / IndexSize);

	// Variables to be set as part of the run start position resolution
	uint_t<IndexSize> uncompressed_position_before_starting_run;

	detail::block::resolve_run_start_positions<IndexSize, UncompressedSize, RunLengthSize>(
		run_start_positions, uncompressed_position_before_starting_run,
		run_lengths,
		block_scan_scratch_area,
		offset_in_first_run_for_block,
		vestige_length_at_next_anchor,
		block_anchor_index,
		starting_uncompressed_position_for_block,
		this_block_reaches_end_of_anchors,
		start_run_index_for_block,
		num_runs_to_use_in_this_block);

	// At this point, the following holds:
	//
	// For the i'th run our block is to use in decompression, we have in
	// output_start_positions_for_runs[i] its start position and in
	// output_start_positions_for_runs[i+1] its end position, both relative
	// to the beginning of the first run our block uses. Note that this
	// is NOT a position relative to our block's output - since the first
	// run might have a vestige from previous blocks.
	//
	// We would _like_ to now proceed to writing output, but unfortunately
	// that's not so simple. We want to coalesce writes, and an individual
	// doesn't know which run corresponds to the element it needs to write.
	// Also, different warps need to know which writes they're going
	// to be responsible for. So there's actually a lot more work left.


//	if (thread::index_in_grid() == block::size()) {
//		for (int i = 0; i < num_runs; i++) { printf("kernel-wide run_start_positions[%2u] = %4u\n", i, run_start_positions[i]); }
//	}
	detail::decompress_with_run_start_positions<IndexSize, UncompressedSize, RunLengthSize>(
		decompressed         + uncompressed_position_before_starting_run,
		run_start_positions,
		run_data             + start_run_index_for_block,
		block_scan_scratch_area,
		length_to_decompress_in_this_block,
		num_runs_to_use_in_this_block,
		offset_in_first_run_for_block);
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned RunLengthSize>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          uncompressed_length,
		size_t                          position_anchoring_period,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit = nullopt) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(decompress<IndexSize, UncompressedSize, RunLengthSize>),
			dynamic_shared_mem_limit
		)
	{
		auto num_periods = util::div_rounding_up(uncompressed_length, position_anchoring_period);

		// TODO:
		grid_construction_resolution            = block;
		length                                  = num_periods;
		serialization_option                    = none;
		dynamic_shared_memory_requirement.per_block =
			util::round_up((position_anchoring_period + 1) * IndexSize, warp_size);
		dynamic_shared_memory_requirement.per_length_unit = 0;
		quanta.threads_in_block                 = warp_size;
		block_resolution_constraints.max_threads_per_block
		                                        = position_anchoring_period;
	};
};

} // namespace run_length_encoding
} // namespace decompression
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_DECOMPRESSION_RUN_LENGTH_CUH */
